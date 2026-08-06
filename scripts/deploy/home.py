#!/usr/bin/env python3
# home.py  --  RUN ON THE PI
"""
Standalone "everything back to home" / bench-reset for the servo bus.

Smoothly ramps all 17 servos to the TRUE STRAIGHT (sim-zero) pose — each joint's
per-joint `center` from config/joint_servo_map.yaml, which was re-zeroed on hardware
6/24 — WAITS long enough for them to physically get there, then disables torque so
the robot goes limp. (Previously this drove everything to a flat 512, which is the
wrong straight on most joints; the arms are off by up to ~35 deg from 512.)

Uses hardware.py's ServoBus (raw-serial SCS driver, same one deploy_standing.py
uses) so it works on the Pi without scservo_sdk.

  python3 home.py                 # ramp to straight over 2s, settle, torque off
  python3 home.py --hold          # ramp to straight, settle, KEEP torque on (holds pose)
  python3 home.py --secs 3        # slower 3s ramp
  python3 home.py --settle 2.0    # extra wait after the ramp before relaxing
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware import ServoBus  # noqa: E402
from sim_real_map import SimRealMap  # noqa: E402

_MAP = SimRealMap()
SERVO_IDS = [int(s) for s in _MAP.servo_ids]            # joint-index order
HOME = {int(s): int(c) for s, c in zip(_MAP.servo_ids, _MAP.centers)}  # per-servo straight


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--secs", type=float, default=2.0, help="ramp duration to home")
    p.add_argument("--settle", type=float, default=1.0,
                   help="extra wait after ramp before disabling torque")
    p.add_argument("--hz", type=float, default=50.0, help="ramp update rate")
    p.add_argument("--speed", type=int, default=300, help="per-move servo speed (0=max)")
    p.add_argument("--hold", action="store_true", help="keep torque on after homing")
    p.add_argument("--waist-lean", type=float, default=0.0,
                   help="forward waist lean in DEGREES added to the straight pose (positive = lean "
                        "forward). Use with --hold to find the symmetric near-upright pose the robot "
                        "actually balances at, as a stable baseline.")
    args = p.parse_args()

    # Build the hold target = straight (centers), optionally with a forward waist lean.
    targets = dict(HOME)
    if args.waist_lean != 0.0:
        wp = next(j for j in _MAP.joints if j.dof == "waist_pitch")
        rad = -np.deg2rad(args.waist_lean)   # forward bend = negative waist_pitch
        u = int(round(wp.center + wp.sign * rad * _MAP.units_per_rad))
        targets[wp.servo_id] = int(np.clip(u, wp.servo_limit[0], wp.servo_limit[1]))
        print(f"waist forward-lean {args.waist_lean:+.1f} deg -> servo {wp.servo_id} = {targets[wp.servo_id]} "
              f"(straight {HOME[wp.servo_id]})")

    dt = 1.0 / args.hz
    steps = max(1, int(args.secs / dt))

    bus = ServoBus().connect()
    try:
        bus.set_torque(SERVO_IDS, True)             # so the servos actually drive home

        # Read current positions (fall back to the straight target if a servo doesn't answer).
        start = {}
        for sid in SERVO_IDS:
            try:
                start[sid] = int(bus.read_pos(sid))
            except Exception:
                start[sid] = targets[sid]
        print("Start positions:", start)

        # Smooth linear ramp every servo from its current pos -> its target (straight + lean).
        print(f"Ramping {len(SERVO_IDS)} servos to hold pose over {args.secs:.1f}s...")
        for k in range(1, steps + 1):
            frac = k / steps
            ramp = [int(round(start[sid] + (targets[sid] - start[sid]) * frac)) for sid in SERVO_IDS]
            bus.write_all(SERVO_IDS, np.array(ramp), speed=args.speed)
            time.sleep(dt)

        bus.write_all(SERVO_IDS, np.array([targets[sid] for sid in SERVO_IDS]), speed=args.speed)
        print(f"Settling {args.settle:.1f}s so servos reach home...")
        time.sleep(args.settle)

        if args.hold:
            print("Done. Torque LEFT ON (holding home).")
        else:
            bus.set_torque(SERVO_IDS, False)
            print("Done. Torque OFF (robot is limp at home).")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
