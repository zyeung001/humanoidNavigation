#!/usr/bin/env python3
# deploy_arm_reach.py
"""Closed-loop ARM pose-reach on the real robot (RUN ON THE PI). The stack demo.

The policy controls ONLY the arm servos (12-17). Legs + waist are parked at straight
(map centers) with torque on; the robot should be SEATED or otherwise supported -- nothing
here can fall. Demo: the arm drives to the commanded pose and HOLDS it; push the arm away
and it returns. That is the closed loop visibly working.

Mirrors src/environments/arm_reach_env.py exactly:
  obs frame (24) = [target(6) | jpos(6) | jvel(6) | last_action(6)], 4-frame history = 96
  jpos = ABSOLUTE joint angle (0 = straight), 40 Hz, tau=0.3 EMA on the action.

  python3 scripts/deploy/deploy_arm_reach.py --policy-npz models/arm_reach_policy.npz \
      --pose stretch --debug
  python3 scripts/deploy/deploy_arm_reach.py --policy-npz ... --cycle 4   # rotate poses
"""

from __future__ import annotations
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_real_map import SimRealMap, DEFAULT_MAP  # noqa: E402

NJ = 17
N_ARM = 6
ARM_IDX = list(range(11, 17))     # R_sh_pitch, R_sh_roll, R_elbow, L_sh_pitch, L_sh_roll, L_elbow
FRAME = 4 * N_ARM
HISTORY = 4
OBS_DIM = FRAME * HISTORY

# Named demo poses (SIM radians, arm order as above), all verified reachable-and-holdable
# in sim 7/28 on the armfix model (weak servos: forcerange ~0.23 N*m caps how far the arms
# can hold against gravity; sim arm-out = R negative / L positive roll).
POSES = {
    "home":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "stretch": [0.0, -0.42, 0.0, 0.0, 0.42, 0.0],       # both arms out sideways
    "forward": [-0.4, 0.0, 0.0, 0.4, 0.0, 0.0],         # both arms forward ~23 deg
    "bent":    [0.0, 0.0, 0.85, 0.0, 0.0, -0.85],       # both elbows flexed ~49 deg
}


def reach_box(m):
    """Reachable sim-rad box implied by the servo EEPROM limits (same math as the env)."""
    lo = np.empty(N_ARM)
    hi = np.empty(N_ARM)
    for k, i in enumerate(ARM_IDX):
        a = m.signs[i] * (m.lim_lo[i] - m.centers[i]) / m.units_per_rad
        b = m.signs[i] * (m.lim_hi[i] - m.centers[i]) / m.units_per_rad
        lo[k], hi[k] = min(a, b), max(a, b)
    return (lo * 0.9).astype(np.float32), (hi * 0.9).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-npz", default=str(ROOT / "models" / "arm_reach_policy.npz"))
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--pose", default="stretch", choices=sorted(POSES))
    p.add_argument("--cycle", type=float, default=0.0,
                   help="seconds per pose; cycles home->stretch->forward->bent. 0 = hold --pose")
    p.add_argument("--tau", type=float, default=0.3, help="action smoothing (MUST match training)")
    p.add_argument("--hz", type=float, default=40.0)
    p.add_argument("--ramp-secs", type=float, default=2.0)
    p.add_argument("--arm-step-units", type=int, default=8, help="per-step move clamp, arm servos")
    p.add_argument("--jvel-alpha", type=float, default=0.35)
    p.add_argument("--jvel-clamp", type=float, default=2.5)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-every", type=int, default=10)
    args = p.parse_args()

    dt = 1.0 / args.hz
    m = SimRealMap(args.map)
    from numpy_policy import NumpyPolicy  # noqa: E402
    npol = NumpyPolicy.load(args.policy_npz)
    assert npol.obs_dim == OBS_DIM, f"npz obs_dim {npol.obs_dim} != {OBS_DIM} (wrong npz?)"
    assert npol.act_dim == N_ARM, f"npz act_dim {npol.act_dim} != {N_ARM} (wrong npz?)"
    lo, hi = reach_box(m)
    print(f"Policy: {args.policy_npz} (obs {npol.obs_dim}, act {npol.act_dim})")
    print(f"reach box lo={np.round(lo, 2)} hi={np.round(hi, 2)}")

    cycle_poses = ["home", "stretch", "forward", "bent"]
    target = np.clip(np.asarray(POSES[args.pose], dtype=np.float32), lo, hi)

    from hardware import ServoBus  # noqa: E402
    bus = ServoBus().connect()
    arm_ids = [m.servo_ids[i] for i in ARM_IDX]
    straight_units = m.rad_to_units(np.zeros(NJ, dtype=np.float32))

    def full_units(arm_rad):
        """17-vector of servo units: legs/waist straight, arms at arm_rad."""
        v = np.zeros(NJ, dtype=np.float32)
        v[ARM_IDX] = arm_rad
        return m.rad_to_units(v)

    last_action = np.zeros(N_ARM, dtype=np.float32)
    jvel_f = np.zeros(N_ARM, dtype=np.float32)
    prev_jpos = None
    stale = np.zeros(N_ARM, dtype=int)   # consecutive rejected frames, per joint
    hist = deque(maxlen=HISTORY)

    def read_arm_jpos():
        units = bus.read_all(m.servo_ids)
        abs_rad = m.units_to_rad(units)
        return abs_rad[ARM_IDX].astype(np.float32)

    def obs_frame(jpos_in):
        nonlocal prev_jpos, jvel_f, stale
        jpos = np.where(np.isfinite(jpos_in), jpos_in,
                        prev_jpos if prev_jpos is not None else 0.0).astype(np.float32)
        if prev_jpos is None:
            raw = np.zeros(N_ARM, dtype=np.float32)
        else:
            raw = (jpos - prev_jpos) / dt
            bad = np.abs(raw) > args.jvel_clamp
            # Reject a suspicious reading at most twice. WITHOUT this the gate LATCHES:
            # rejecting a sample left prev_jpos unchanged, so the next frame sat exactly as
            # far away, stayed "bad", and jpos froze at a stale value while jvel railed at
            # the clamp forever. Measured on 8/4 (arm_hold3): err locked at 0.359 rad with
            # 0.005 rad of variation and jvel pinned at 2.50 on 93% of frames while the arm
            # sat still -- and the policy could not react to a push at all, because its
            # observation was frozen. A reading that persists 3 frames is real motion (a
            # shove, a fast slew) and is accepted, matching deploy_standing.py's gate.
            stale = np.where(bad, stale + 1, 0)
            hold = bad & (stale < 3)
            jpos = np.where(hold, prev_jpos, jpos)
            raw = np.clip(np.where(hold, 0.0, raw), -args.jvel_clamp, args.jvel_clamp)
        prev_jpos = jpos.copy()
        jvel_f = (1.0 - args.jvel_alpha) * jvel_f + args.jvel_alpha * raw
        return np.concatenate([target, jpos, jvel_f, last_action]).astype(np.float32)

    def obs_of(frame):
        hist.append(frame)
        frames = list(hist)
        if len(frames) < HISTORY:
            frames = [np.zeros(FRAME, dtype=np.float32)] * (HISTORY - len(frames)) + frames
        return np.concatenate(frames)

    try:
        print("Torque on, ramping to straight (legs parked, arms home)...")
        bus.set_torque(m.servo_ids, True)
        cur = bus.read_all(m.servo_ids)
        cur = np.where(np.isfinite(cur), cur, straight_units.astype(float))
        steps = max(1, int(args.ramp_secs / dt))
        for k in range(1, steps + 1):
            u = (cur + (straight_units - cur) * k / steps).round().astype(int)
            bus.write_all(m.servo_ids, u, speed=0)
            time.sleep(dt)

        for _ in range(HISTORY):
            obs_of(obs_frame(read_arm_jpos()))
            time.sleep(dt)

        pose_name = args.pose
        print(f"Closed loop on ARMS (Ctrl-C to stop). pose={pose_name} cycle={args.cycle}s")
        prev_units = full_units(last_action)[ARM_IDX].astype(float)
        t_pose = time.time()
        step = 0
        while True:
            t0 = time.time()
            if args.cycle > 0 and t0 - t_pose > args.cycle:
                pose_name = cycle_poses[(cycle_poses.index(pose_name) + 1) % len(cycle_poses)]
                target[:] = np.clip(np.asarray(POSES[pose_name], dtype=np.float32), lo, hi)
                t_pose = t0
                print(f"  -> pose {pose_name}")
            obs = obs_of(obs_frame(read_arm_jpos()))
            raw = np.clip(npol.predict(obs).ravel(), lo, hi)
            applied = np.clip((1.0 - args.tau) * last_action + args.tau * raw, lo, hi)
            last_action[:] = applied
            units = full_units(applied)[ARM_IDX]
            units = np.clip(units, prev_units - args.arm_step_units,
                            prev_units + args.arm_step_units)
            units = np.clip(units, np.asarray(m.lim_lo)[ARM_IDX],
                            np.asarray(m.lim_hi)[ARM_IDX]).round().astype(int)
            bus.write_all(arm_ids, units, speed=0)
            prev_units = units.astype(float)
            if args.debug and step % args.debug_every == 0:
                err = prev_jpos - target
                print(f"[{step:5d}] pose={pose_name:8s} max|err|={np.abs(err).max():.3f} "
                      f"max|jvel_f|={np.abs(jvel_f).max():.2f} "
                      f"busy={(time.time() - t0) * 1000:4.1f}ms")
            step += 1
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nStopping: arms back to straight.")
        try:
            cur = bus.read_all(m.servo_ids)
            cur = np.where(np.isfinite(cur), cur, straight_units.astype(float))
            steps = max(1, int(1.0 / dt))
            for k in range(1, steps + 1):
                u = (cur + (straight_units - cur) * k / steps).round().astype(int)
                bus.write_all(m.servo_ids, u, speed=0)
                time.sleep(dt)
        except Exception:
            pass
    finally:
        bus.set_torque(m.servo_ids, False)
        bus.close()


if __name__ == "__main__":
    main()
