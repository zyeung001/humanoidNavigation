#!/usr/bin/env python3
# probe_sign.py  --  RUN ON THE PI
"""
SSH-drivable single-joint sign probe. Unlike verify_signs.py (which holds the whole
turned-out STANDING pose and is interactive), this holds everything STRAIGHT at 512
and wiggles exactly ONE joint, then exits. No prompts -> drivable over a non-interactive
SSH call. The operator watches and reports; signs are recorded separately.

  python3 probe_sign.py --idx 0                 # wiggle action-index 0's servo, +then-
  python3 probe_sign.py --idx 0 --delta 0.3     # amplitude in sim radians
  python3 probe_sign.py --straight              # just hold all at 512 (torque on)
  python3 probe_sign.py --relax                 # torque OFF (limp)

Each probe: hold all at 512, then for the target joint do
  512 -> +delta -> 512 -> -delta -> 512   (slow), torque LEFT ON between calls.
+delta means +sim_rad through the map (units = 512 + sign*delta*units_per_rad).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware import ServoBus          # noqa: E402
from sim_real_map import SimRealMap    # noqa: E402

NEUTRAL = 512
ARM_SERVOS = {12, 13, 14, 15, 16, 17}   # SCS0009 arms; kept limp unless being probed


def ramp(bus, sid, a, b, secs=0.8, hz=50, speed=200):
    steps = max(1, int(secs * hz))
    for k in range(1, steps + 1):
        u = int(round(a + (b - a) * k / steps))
        bus.write_pos(sid, u, speed=speed)
        time.sleep(1.0 / hz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, help="action index to probe")
    ap.add_argument("--delta", type=float, default=0.3, help="amplitude (sim rad)")
    ap.add_argument("--speed", type=int, default=200)
    ap.add_argument("--straight", action="store_true", help="just hold all at 512")
    ap.add_argument("--relax", action="store_true", help="torque off and exit")
    args = ap.parse_args()

    m = SimRealMap()
    bus = ServoBus().connect()

    if args.relax:
        bus.set_torque(m.servo_ids, False)
        print("Torque OFF (limp).")
        bus.close()
        return

    # Arms stay limp (they hunt under torque) unless they're the joint being probed.
    target_sid = m.joints[args.idx].servo_id if args.idx is not None else None
    arms_off = [s for s in m.servo_ids.tolist() if s in ARM_SERVOS and s != target_sid]
    hold_ids = [s for s in m.servo_ids.tolist() if s not in arms_off]
    bus.set_torque(arms_off, False)
    bus.set_torque(hold_ids, True)
    bus.write_all(hold_ids, np.full(len(hold_ids), NEUTRAL), speed=args.speed)
    time.sleep(0.4)
    if args.straight:
        print(f"Holding {len(hold_ids)} servos at 512; arms limp: {arms_off}")
        bus.close()
        return

    if args.idx is None:
        print("give --idx N (or --straight / --relax)")
        bus.close()
        return

    j = m.joints[args.idx]
    # +delta and -delta as raw units via the map sign (clamped to servo_limit).
    u_plus = int(np.clip(NEUTRAL + j.sign * args.delta * m.units_per_rad, j.servo_limit[0], j.servo_limit[1]))
    u_minus = int(np.clip(NEUTRAL - j.sign * args.delta * m.units_per_rad, j.servo_limit[0], j.servo_limit[1]))
    pdir = "UP (toward higher units)" if u_plus > NEUTRAL else "DOWN (toward lower units)"

    print(f"PROBE idx {args.idx}: servo {j.servo_id}  {j.dof}  (sign {j.sign:+d})")
    print(f"  +{args.delta} rad -> units {u_plus}  ({pdir});  -{args.delta} rad -> units {u_minus}")
    print("  sequence: 512 -> +d -> 512 -> -d -> 512")

    ramp(bus, j.servo_id, NEUTRAL, u_plus, speed=args.speed)
    time.sleep(0.6)
    ramp(bus, j.servo_id, u_plus, NEUTRAL, speed=args.speed)
    time.sleep(0.3)
    ramp(bus, j.servo_id, NEUTRAL, u_minus, speed=args.speed)
    time.sleep(0.6)
    ramp(bus, j.servo_id, u_minus, NEUTRAL, speed=args.speed)
    time.sleep(0.3)
    print("  done (held at 512, torque still ON).")
    bus.close()


if __name__ == "__main__":
    main()
