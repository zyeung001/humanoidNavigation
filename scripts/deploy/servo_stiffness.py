#!/usr/bin/env python3
# servo_stiffness.py  --  RUN ON THE PI
"""
Make the load-bearing servos hold POSITION more rigidly under load, to narrow the #1
sim2real gap for standing: the sim joints are perfectly rigid, but the real SCS servos sag
and wobble under the body's weight, so the policy's balance strategy goes unstable (1-2 rad/s
limit-cycle sway). Stiffening the joints moves the real plant toward the rigid sim hinge the
policy was trained on.

This firmware uses a position PID (confirmed by reading the table -- reg 21/22/23 = P/D/I
hold sane values, the AX compliance-slope regs 28/29 are vestigial). STIFFNESS = P (reg 21):
RAISE it to make the joint hold position more firmly under load. Default P here is ~15 (soft);
try 24, then 32. Too high makes the servo buzz/overshoot in its OWN loop, so step up and watch.
Optionally adjust D (reg 22): the stock D=15 is high (sluggish hold) -- lowering D can sharpen
the response, but change ONE thing at a time.

This tool READS the current registers first (always, even when writing) so you can revert,
and READS BACK after writing to confirm it stuck. EEPROM is LOCK-protected (reg 48):
unlock(0) -> write -> relock(1). Defaults to the LEGS+WAIST (the load-bearing joints).

  python3 servo_stiffness.py --read-only                 # inspect legs+waist, write nothing
  python3 servo_stiffness.py --read-only --all           # inspect every servo
  python3 servo_stiffness.py --legs --kp 24 --apply      # stiffen legs+waist: P=24
  python3 servo_stiffness.py --legs --kp 24 --kd 8 --apply # also lower D to 8
  python3 servo_stiffness.py --legs --kp 15 --kd 15 --apply # REVERT to stock
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware import ServoBus  # noqa: E402

REG_P = 21             # some SCS firmware exposes position PID here instead of slope
REG_D = 22
REG_I = 23
REG_CW_MARGIN = 26
REG_CCW_MARGIN = 27
REG_CW_SLOPE = 28      # the stiffness lever (lower = stiffer)
REG_CCW_SLOPE = 29
REG_LOCK = 48          # SCSCL EEPROM lock: 0=unlocked (writable), 1=locked

ARMS = list(range(12, 18))           # SCS0009 arm servos
LEGS = list(range(1, 12))            # legs + 3-DOF waist: the load-bearing joints
ALL = list(range(1, 18))


def _r(bus, sid, reg):
    v = bus.read_reg(sid, reg)
    return v[0] if v else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--servo", type=int, default=None, help="single servo id")
    p.add_argument("--legs", action="store_true", help="legs + waist (servos 1..11), the default target")
    p.add_argument("--arms", action="store_true", help="arm servos 12..17")
    p.add_argument("--all", action="store_true", help="all 17 servos")
    p.add_argument("--kp", type=int, default=None, help="position P gain to set (reg 21; higher=stiffer)")
    p.add_argument("--kd", type=int, default=None, help="position D gain to set (reg 22; optional)")
    p.add_argument("--apply", action="store_true", help="actually write (otherwise read-only)")
    p.add_argument("--read-only", action="store_true", help="inspect only, never write")
    args = p.parse_args()

    if args.servo is not None:
        ids = [args.servo]
    elif args.all:
        ids = ALL
    elif args.arms:
        ids = ARMS
    else:
        ids = LEGS   # default

    write = args.apply and not args.read_only and (args.kp is not None or args.kd is not None)
    if args.apply and args.kp is None and args.kd is None:
        print("--apply given but no --kp/--kd: nothing to write. (read-only shown below)")
    for val, nm in ((args.kp, "kp"), (args.kd, "kd")):
        if val is not None and not 0 <= val <= 100:
            print(f"refusing {nm}={val}: stay in 0..100 (stock P/D are ~15).")
            return

    bus = ServoBus().connect()
    try:
        print(f"{'sid':>3}  {'P':>3} {'D':>3} {'I':>3} | {'cwMrg':>5} {'ccwMrg':>6} | "
              f"{'cwSlp':>5} {'ccwSlp':>6}")
        for sid in ids:
            pP, pD, pI = _r(bus, sid, REG_P), _r(bus, sid, REG_D), _r(bus, sid, REG_I)
            cwm, ccwm = _r(bus, sid, REG_CW_MARGIN), _r(bus, sid, REG_CCW_MARGIN)
            cws, ccws = _r(bus, sid, REG_CW_SLOPE), _r(bus, sid, REG_CCW_SLOPE)
            if cws is None and ccws is None:
                print(f"{sid:>3}  NO REPLY (skipped)")
                continue
            print(f"{sid:>3}  {str(pP):>3} {str(pD):>3} {str(pI):>3} | {str(cwm):>5} {str(ccwm):>6} | "
                  f"{str(cws):>5} {str(ccws):>6}", end="")
            if not write:
                print("  (read-only)")
                continue
            bus.write_reg(sid, REG_LOCK, 0)
            if args.kp is not None:
                bus.write_reg(sid, REG_P, args.kp)
            if args.kd is not None:
                bus.write_reg(sid, REG_D, args.kd)
            bus.write_reg(sid, REG_LOCK, 1)
            nP, nD = _r(bus, sid, REG_P), _r(bus, sid, REG_D)
            ok = ((args.kp is None or nP == args.kp) and (args.kd is None or nD == args.kd))
            print(f"  ->  P={nP} D={nD}  {'OK' if ok else 'FAILED (unchanged?)'}")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
