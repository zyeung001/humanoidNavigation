#!/usr/bin/env python3
# home.py  --  RUN ON THE PI
"""
Standalone "everything back to home" / bench-reset for the servo bus.

Smoothly ramps all 17 servos to 512 (raw neutral), WAITS long enough for them to
physically get there, then disables torque so the robot goes limp.

Uses hardware.py's ServoBus (raw-serial SCS driver, same one deploy_standing.py
uses) so it works on the Pi without scservo_sdk.

  python3 home.py                 # ramp to 512 over 2s, settle, torque off
  python3 home.py --hold          # ramp to 512, settle, KEEP torque on (holds pose)
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

SERVO_IDS = list(range(1, 18))   # 1..17
HOME = 512                       # raw neutral (10-bit center)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--secs", type=float, default=2.0, help="ramp duration to home")
    p.add_argument("--settle", type=float, default=1.0,
                   help="extra wait after ramp before disabling torque")
    p.add_argument("--hz", type=float, default=50.0, help="ramp update rate")
    p.add_argument("--speed", type=int, default=300, help="per-move servo speed (0=max)")
    p.add_argument("--hold", action="store_true", help="keep torque on after homing")
    args = p.parse_args()

    dt = 1.0 / args.hz
    steps = max(1, int(args.secs / dt))

    bus = ServoBus().connect()
    try:
        bus.set_torque(SERVO_IDS, True)             # so the servos actually drive home

        # Read current positions (fall back to HOME if a servo doesn't answer).
        start = {}
        for sid in SERVO_IDS:
            try:
                start[sid] = int(bus.read_pos(sid))
            except Exception:
                start[sid] = HOME
        print("Start positions:", start)

        # Smooth linear ramp every servo from its current pos -> 512.
        print(f"Ramping {len(SERVO_IDS)} servos to {HOME} over {args.secs:.1f}s...")
        for k in range(1, steps + 1):
            frac = k / steps
            targets = [int(round(start[sid] + (HOME - start[sid]) * frac)) for sid in SERVO_IDS]
            bus.write_all(SERVO_IDS, np.array(targets), speed=args.speed)
            time.sleep(dt)

        bus.write_all(SERVO_IDS, np.full(len(SERVO_IDS), HOME), speed=args.speed)
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
