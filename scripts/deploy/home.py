#!/usr/bin/env python3
# home.py  --  RUN ON THE PI
"""
Standalone "everything back to home" / bench-reset for the SCS servo bus.

Smoothly ramps all 17 servos to 512 (raw neutral), WAITS long enough for them to
physically get there, then disables torque so the robot goes limp.

Self-contained: only needs scservo_sdk (no repo imports). Copy-paste onto the Pi.

  python3 home.py                 # ramp to 512 over 2s, settle, torque off
  python3 home.py --hold          # ramp to 512, settle, KEEP torque on (holds pose)
  python3 home.py --secs 3        # slower 3s ramp
  python3 home.py --settle 2.0    # extra wait after the ramp before relaxing
"""

import argparse
import time

from scservo_sdk import PortHandler, scscl

PORT = "/dev/ttyAMA0"
BAUD = 1_000_000
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

    ph = PortHandler(PORT)
    if not ph.openPort():
        raise RuntimeError(f"failed to open {PORT}")
    if not ph.setBaudRate(BAUD):
        raise RuntimeError(f"failed to set baud {BAUD}")
    pk = scscl(ph)

    dt = 1.0 / args.hz
    steps = max(1, int(args.secs / dt))

    # Enable torque so the servos actually drive to home.
    for sid in SERVO_IDS:
        try:
            pk.write1ByteTxRx(sid, 40, 1)  # reg 40 = torque enable
        except Exception as e:
            print(f"  [warn] torque-enable id {sid}: {e}")

    # Read current positions (fall back to HOME if a servo doesn't answer).
    start = {}
    for sid in SERVO_IDS:
        try:
            pos, _, comm, err = pk.ReadPos(sid)
            start[sid] = int(pos) if (comm == 0 and err == 0) else HOME
        except Exception:
            start[sid] = HOME
    print("Start positions:", {k: start[k] for k in SERVO_IDS})

    # Smooth linear ramp every servo from its current pos -> 512.
    print(f"Ramping {len(SERVO_IDS)} servos to {HOME} over {args.secs:.1f}s...")
    for k in range(1, steps + 1):
        frac = k / steps
        for sid in SERVO_IDS:
            tgt = int(round(start[sid] + (HOME - start[sid]) * frac))
            try:
                pk.WritePos(sid, tgt, 0, args.speed)  # WritePos(id, pos, time=0, speed)
            except Exception as e:
                print(f"  [warn] WritePos id {sid}: {e}")
        time.sleep(dt)

    # Make sure they're all commanded exactly to home, then let them finish moving.
    for sid in SERVO_IDS:
        try:
            pk.WritePos(sid, HOME, 0, args.speed)
        except Exception:
            pass
    print(f"Settling {args.settle:.1f}s so servos reach home...")
    time.sleep(args.settle)

    if args.hold:
        print("Done. Torque LEFT ON (holding home).")
    else:
        for sid in SERVO_IDS:
            try:
                pk.write1ByteTxRx(sid, 40, 0)  # torque disable -> limp
            except Exception:
                pass
        print("Done. Torque OFF (robot is limp at home).")

    ph.closePort()


if __name__ == "__main__":
    main()
