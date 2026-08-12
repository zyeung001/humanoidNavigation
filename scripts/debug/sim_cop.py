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

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = ROOT / "models" / "humanoid_real_v2.xml"


def settle(model, pose="straight", drop=0.0, yaw=0.0, jitter=0.0, seed=0, seconds=4.0):
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


def report(model, data, span_mm=80.0):
    feet = _foot_geoms(model)
    loads = contact_loads(model, data, feet)
    total = sum(f for g in feet for f, _ in loads[g])
    weight = model.body_mass.sum() * 9.81
    com = data.subtree_com[0]
    fwd = forward_axis(model, data)

    print(f"pose settled: tilt cos {data.xmat[1].reshape(3, 3)[:, 2] @ [0, 0, 1]:.4f}, "
          f"root |v| {np.linalg.norm(data.qvel[:3]):.4f} m/s")
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
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    print(f"{args.xml}\nmass {model.body_mass.sum():.4f} kg, pose={args.pose}\n")

    if not args.sweep:
        report(model, settle(model, args.pose, seconds=args.settle), args.span)
        return

    print(f"{'condition':18s} {'CoP mm':>8s}")
    conds = [("nominal", {}), ("drop +2mm", {"drop": 0.002}), ("drop +5mm", {"drop": 0.005}),
             ("yaw +10deg", {"yaw": np.deg2rad(10)}), ("jitter s1", {"jitter": 0.01, "seed": 1}),
             ("jitter s2", {"jitter": 0.01, "seed": 2}), ("settle 8s", {"seconds": 8.0})]
    vals = []
    for label, kw in conds:
        kw.setdefault("seconds", args.settle)
        data = settle(model, args.pose, **kw)
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
