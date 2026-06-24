#!/usr/bin/env python3
# servo_deadband.py  --  RUN ON THE PI
"""
Stop SCS servos from buzzing/hunting while torque is ON, by widening their position
DEADBAND (the control-loop compliance margin, NOT the pot-seam dead zone).

Cheap Feetech SCS (SCSCL) servos hold position with a ~1-unit deadband, so they keep
twitching to null sub-degree error -> the hum you hear with torque on but no motion
commanded. Widening CW_DEAD (reg 26) / CCW_DEAD (reg 27) tells the servo "close enough"
so it stops fighting. A few units = ~1 deg of slack; quiet hold, negligible posture loss.
(Side benefit: it also low-passes the policy's micro-chatter at the hardware level.)

These are EEPROM registers protected by LOCK (reg 48): unlock (0) -> write -> relock (1).
This tool READS the current value first (sanity-checks the register) and READS BACK after
writing (confirms it stuck). Does ONE servo by default; pass --all for every joint.

  python3 servo_deadband.py --read-only                 # inspect servo 14, write nothing
  python3 servo_deadband.py --servo 14 --value 8        # set arm servo 14 deadband to 8
  python3 servo_deadband.py --arms --value 8            # all 6 arm servos (12..17)
  python3 servo_deadband.py --all  --value 6            # all 17 (legs a touch smaller)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware import ServoBus  # noqa: E402

REG_CW_DEAD = 26
REG_CCW_DEAD = 27
REG_LOCK = 48          # SCSCL EEPROM lock: 0=unlocked (writable), 1=locked
ARMS = list(range(12, 18))   # SCS0009 arm servos
ALL = list(range(1, 18))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--servo", type=int, default=14, help="single servo id (default 14, an arm)")
    p.add_argument("--arms", action="store_true", help="apply to all arm servos 12..17")
    p.add_argument("--all", action="store_true", help="apply to all 17 servos")
    p.add_argument("--value", type=int, default=8, help="deadband units to set (cw=ccw). ~8 ~= 1 deg")
    p.add_argument("--read-only", action="store_true", help="inspect current deadband, write nothing")
    args = p.parse_args()

    if args.all:
        ids = ALL
    elif args.arms:
        ids = ARMS
    else:
        ids = [args.servo]

    if not 0 <= args.value <= 32:
        print(f"refusing value={args.value}: stay in 0..32 (a few units is plenty).")
        return

    bus = ServoBus().connect()
    try:
        for sid in ids:
            cw = bus.read_reg(sid, REG_CW_DEAD)
            ccw = bus.read_reg(sid, REG_CCW_DEAD)
            if cw is None or ccw is None:
                print(f"servo {sid:2d}: NO REPLY (skipped)")
                continue
            cw, ccw = cw[0], ccw[0]
            print(f"servo {sid:2d}: current deadband cw={cw} ccw={ccw}", end="")
            if cw > 32 or ccw > 32:
                print("  -- value looks wrong for a deadband register; SKIPPING write (verify model).")
                continue
            if args.read_only:
                print("  (read-only)")
                continue

            bus.write_reg(sid, REG_LOCK, 0)           # unlock EEPROM
            bus.write_reg(sid, REG_CW_DEAD, args.value)
            bus.write_reg(sid, REG_CCW_DEAD, args.value)
            bus.write_reg(sid, REG_LOCK, 1)           # relock

            ncw = bus.read_reg(sid, REG_CW_DEAD)
            nccw = bus.read_reg(sid, REG_CCW_DEAD)
            ncw = ncw[0] if ncw else None
            nccw = nccw[0] if nccw else None
            ok = (ncw == args.value and nccw == args.value)
            print(f"  ->  set cw={ncw} ccw={nccw}  {'OK' if ok else 'FAILED (unchanged?)'}")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
