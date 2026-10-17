#!/usr/bin/env python3
"""Sim's predicted centre of pressure, as a reference for the foot FSRs.

Centre of pressure is the one balance quantity the robot cannot compute about
itself. Deriving it from the IMU and joint angles would route it through the
model we are trying to validate, and the known fault is invisible to both of
those sensors: the waist stack deflects forward under load with no servo leaving
position, so the encoders report nothing moved and the pelvis IMU reports a
BACKWARD lean. Pressure at the feet does not care. That makes measured-vs-predicted
CoP an independent test of the model rather than a restatement of it.

Default pose is STRAIGHT (every hinge 0, ctrl 0), which is what `home.py --hold`
puts the real robot in -- so the number printed here is directly comparable to
`scripts/deploy/fsr_monitor.py` running on a heel/toe instrumented foot.

The equilibrium check is the thing to read first. A settled robot must have its
CoP directly under its COM and its contact load equal to its weight; if those
two lines disagree, the contact maths is wrong and every other number is noise.

  python scripts/debug/sim_cop.py
  python scripts/debug/sim_cop.py --pose keyframe --settle 8
  python scripts/debug/sim_cop.py --sweep          # robustness across settling conditions
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = ROOT / "models" / "humanoid_real_v2.xml"
DEFAULT_MAP = ROOT / "config" / "joint_servo_map.yaml"


def act_qposadr(model):
    """qpos address of each actuated hinge, in ACTUATOR order.

    Addressed through the actuator transmission rather than assumed to be qpos[7:], for the
    same reason standing_env.py does it: on a CAD-exported MJCF the hinge order is whatever
    the exporter emitted, and residual_baseline is indexed by ACTION, i.e. by actuator.
    """
    return model.jnt_qposadr[model.actuator_trnid[:, 0]]


def load_pose(model, config=None, trims=None, map_path=DEFAULT_MAP):
    """Joint targets in radians, actuator-indexed: a config's baseline plus per-joint trims.

    The pose the robot balances in is a multi-joint offset, not a trim off straight, so
    comparing sim against hardware requires posing sim at the same whole-body pose the
    hardware held -- exactly what pose_sweep.py --baseline does on the robot. Trims are
    applied in the SAME sign convention as the baseline, matching pose_sweep's
    `u += sign * deg2rad(trim) * units_per_rad`, so a trim quoted from a probe can be
    typed in here unchanged.
    """
    n = model.nu
    q = np.zeros(n)
    if config:
        b = np.asarray(yaml.safe_load(open(config))["standing"]["residual_baseline"], float)
        if b.shape[0] != n:
            raise SystemExit(f"{config}: residual_baseline has {b.shape[0]} values, "
                             f"expected {n}")
        q += b
    if trims:
        dofs = [j["dof"] for j in yaml.safe_load(open(map_path))["joints"]]
        for item in trims.split(","):
            if not item.strip():
                continue
            name, _, deg = item.partition("=")
            name = name.strip()
            if name not in dofs:
                raise SystemExit(f"unknown joint {name!r}; known: {', '.join(dofs)}")
            q[dofs.index(name)] += np.deg2rad(float(deg))
    return q


def settle(model, pose="straight", drop=0.0, yaw=0.0, jitter=0.0, seed=0, seconds=4.0,
           target=None):
    """Stand the model on the floor at `pose` and let the position servos hold it."""
    data = mujoco.MjData(model)
    if pose == "keyframe" and model.nkey:
        data.qpos[:] = model.key_qpos[0]
    else:
        data.qpos[:] = 0
        data.qpos[2] = 0.5
    data.qpos[3] = np.cos(yaw / 2.0)
    data.qpos[6] = np.sin(yaw / 2.0)
    data.ctrl[:] = 0
    if target is not None:
        # Start AT the pose as well as commanding it. Starting straight and letting the
        # servos drive there works, but the swing throws the robot off its feet before it
        # settles, and the question here is what the pose does statically.
        lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        data.ctrl[:] = np.clip(target, lo, hi)
        data.qpos[act_qposadr(model)] = data.ctrl
    mujoco.mj_forward(model, data)

    # drop so the lowest possible foot vertex just clears the floor, then settle onto it
    feet = _foot_geoms(model)
    lowest = min(data.geom_xpos[g][2] - abs(model.geom_size[g]).max() for g in feet)
    data.qpos[2] -= lowest - drop
    if jitter:
        rng = np.random.default_rng(seed)
        data.qpos[7:] += rng.normal(0.0, jitter, model.nq - 7)
    mujoco.mj_forward(model, data)

    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return data


def _foot_geoms(model):
    """Box collision geoms thin in one axis and broad in the others -- the foot plates."""
    out = []
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or not model.geom_contype[g]:
            continue
        s = np.sort(model.geom_size[g])
        if s[0] < 0.01 and s[1] > 0.02 and s[2] > 0.04:
            out.append(g)
    return out


def contact_loads(model, data, feet):
    """Vertical contact load and its moment about the origin, per foot geom."""
    loads = {g: [] for g in feet}
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        pair = (c.geom1, c.geom2)
        g = next((x for x in pair if x in loads), None)
        if g is None or not any(model.geom_type[x] == mujoco.mjtGeom.mjGEOM_PLANE for x in pair):
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        fz = float((c.frame.reshape(3, 3).T @ wrench[:3])[2])
        loads[g].append((fz, c.pos.copy()))
    return loads


def forward_axis(model, data):
    """Pelvis +X flattened into the horizontal plane -- the fore/aft axis."""
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    fwd = data.xmat[pelvis].reshape(3, 3)[:, 0].copy()
    fwd[2] = 0.0
    return fwd / np.linalg.norm(fwd)


def pelvis_attitude(model, data):
    """Pelvis pitch and roll in degrees, from projected gravity.

    Computed exactly as pose_sweep.py computes it from the IMU -- pg = R^T @ (0,0,-1), then
    atan2(pg[0], -pg[2]) -- so the number printed here and the number the robot streams are
    the same quantity and can be subtracted. Anything else (Euler angles off the quaternion,
    say) would agree near upright and diverge exactly where it matters.
    """
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    pg = data.xmat[pelvis].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    return (float(np.degrees(np.arctan2(pg[0], -pg[2]))),
            float(np.degrees(np.arctan2(pg[1], -pg[2]))))


def report(model, data, span_mm=80.0):
    feet = _foot_geoms(model)
    loads = contact_loads(model, data, feet)
    total = sum(f for g in feet for f, _ in loads[g])
    weight = model.body_mass.sum() * 9.81
    com = data.subtree_com[0]
    fwd = forward_axis(model, data)

    print(f"pose settled: tilt cos {data.xmat[1].reshape(3, 3)[:, 2] @ [0, 0, 1]:.4f}, "
          f"root |v| {np.linalg.norm(data.qvel[:3]):.4f} m/s")
    pitch, roll = pelvis_attitude(model, data)
    print(f"  pelvis pitch {pitch:+6.2f} deg   roll {roll:+6.2f} deg   "
          f"(+pitch = leaning FORWARD, same convention as pose_sweep.py)")
    if total <= 1e-6:
        print("NO GROUND CONTACT -- the robot is not standing; nothing to report.")
        return None

    cop = sum(f * p for g in feet for f, p in loads[g]) / total
    centre = np.mean([data.geom_xpos[g] for g in feet], axis=0)
    lateral = np.cross([0.0, 0.0, 1.0], fwd)

    print("\n--- EQUILIBRIUM CHECK (read this first) ---")
    print(f"  contact load {total:6.2f} N   vs weight {weight:6.2f} N   "
          f"(diff {abs(total - weight):.3f})")
    print(f"  CoP vs COM: fore/aft {1000 * ((cop - centre) @ fwd):+.2f} vs "
          f"{1000 * ((com - centre) @ fwd):+.2f} mm, "
          f"lateral {1000 * (cop @ lateral):+.2f} vs {1000 * (com @ lateral):+.2f} mm")
    print("  (a settled robot must have CoP under COM; large gaps invalidate everything below)")

    print("\n--- FORE/AFT CoP  (+ = toward the toe) ---")
    for g in sorted(feet, key=lambda g: data.geom_xpos[g] @ lateral):
        f_g = sum(f for f, _ in loads[g])
        side = "LEFT " if data.geom_xpos[g] @ lateral > com @ lateral else "RIGHT"
        if f_g <= 1e-6:
            print(f"  {side} foot: NO LOAD")
            continue
        c_g = sum(f * p for f, p in loads[g]) / f_g
        print(f"  {side} foot: {1000 * ((c_g - data.geom_xpos[g]) @ fwd):+6.1f} mm from centre, "
              f"{100 * f_g / total:5.1f}% of weight, {len(loads[g])} contact points")
    whole = 1000 * ((cop - centre) @ fwd)
    print(f"  WHOLE ROBOT: {whole:+.1f} mm")
    print(f"\nFSR sensors at +-{span_mm / 2:.0f} mm would read about {whole:+.0f} mm "
          f"if all load passes through them.")
    return whole


def main():
    p = argparse.ArgumentParser(description="Sim centre-of-pressure reference for the foot FSRs")
    p.add_argument("--xml", default=str(DEFAULT_XML))
    p.add_argument("--pose", choices=["straight", "keyframe"], default="straight")
    p.add_argument("--settle", type=float, default=4.0, help="seconds to settle")
    p.add_argument("--span", type=float, default=80.0, help="heel-to-toe sensor spacing, mm")
    p.add_argument("--sweep", action="store_true",
                   help="re-run across settling conditions; CoP should not move")
    p.add_argument("--baseline", default=None, metavar="CONFIG",
                   help="pose at a config's standing.residual_baseline instead of straight, "
                        "so sim holds the same whole-body pose the robot held under "
                        "pose_sweep.py --baseline and the two are directly comparable")
    p.add_argument("--trim", default=None, metavar="JOINT=DEG",
                   help="comma-separated per-joint trims in degrees, applied on top of the "
                        "baseline (e.g. waist_pitch=6). Same sign convention as pose_sweep, "
                        "so a trim quoted from a probe transfers unchanged.")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    target = (load_pose(model, args.baseline, args.trim)
              if (args.baseline or args.trim) else None)
    print(f"{args.xml}\nmass {model.body_mass.sum():.4f} kg, pose={args.pose}")
    if target is not None:
        print(f"posed at {args.baseline or 'straight'}"
              + (f" + {args.trim}" if args.trim else "")
              + f"  (max |joint| {np.degrees(np.abs(target).max()):.1f} deg)")
        lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        if np.any(target < lo - 1e-9) or np.any(target > hi + 1e-9):
            bad = np.where((target < lo - 1e-9) | (target > hi + 1e-9))[0]
            print(f"  !! CLIPPED by ctrlrange at actuator(s) {list(bad)} -- sim is NOT "
                  "holding the pose you asked for")
    print()

    if not args.sweep:
        report(model, settle(model, args.pose, seconds=args.settle, target=target), args.span)
        return

    print(f"{'condition':18s} {'CoP mm':>8s}")
    conds = [("nominal", {}), ("drop +2mm", {"drop": 0.002}), ("drop +5mm", {"drop": 0.005}),
             ("yaw +10deg", {"yaw": np.deg2rad(10)}), ("jitter s1", {"jitter": 0.01, "seed": 1}),
             ("jitter s2", {"jitter": 0.01, "seed": 2}), ("settle 8s", {"seconds": 8.0})]
    vals = []
    for label, kw in conds:
        kw.setdefault("seconds", args.settle)
        data = settle(model, args.pose, target=target, **kw)
        feet = _foot_geoms(model)
        loads = contact_loads(model, data, feet)
        total = sum(f for g in feet for f, _ in loads[g])
        if total <= 1e-6:
            print(f"{label:18s}   no contact")
            continue
        cop = sum(f * pt for g in feet for f, pt in loads[g]) / total
        centre = np.mean([data.geom_xpos[g] for g in feet], axis=0)
        v = 1000 * ((cop - centre) @ forward_axis(model, data))
        vals.append(v)
        print(f"{label:18s} {v:+8.1f}")
    if vals:
        print(f"\nspread {max(vals) - min(vals):.2f} mm -- a stable prediction should be well under 1 mm")


if __name__ == "__main__":
    main()
