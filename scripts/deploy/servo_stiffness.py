#!/usr/bin/env python3
# servo_stiffness.py  --  RUN ON THE PI
"""
Make the load-bearing servos hold POSITION more rigidly under load, to narrow the #1
sim2real gap for standing: the sim joints are perfectly rigid, but the real SCS servos sag
and wobble under the body's weight, so the policy's balance strategy goes unstable (1-2 rad/s
limit-cycle sway). Stiffening the joints moves the real plant toward the rigid sim hinge the
policy was trained on.

This firmware uses a position PID (confirmed by reading the table -- reg 21/22/23 = P/D/I
hold sane values, the AX compliance-slope regs 28/29 are vestigial).

P (reg 21) = stiffness: raising it holds position more firmly under load. TRIED AND
REJECTED: P 15->24 was set 6/23 and left on by accident until 7/27; every hardware test in
that window ran on a ~60% stiffer plant than trained, and it made the oscillation WORSE.
Reverted to stock P=15 on servos 1-11. Raising P adds loop gain and SPENDS phase margin --
it is the wrong lever for this robot's failure mode. Leave P at 15 unless you have a new
reason.

D (reg 22) = damping, and this is the lever that matters here (8/5). Every oscillation on
this robot (standing 1.96 Hz, arm 0.9 Hz, offline sensor-free emulation 1.6-1.8 Hz) sits
just under the ~2.9 Hz where the servo's own ~50 ms dead time + ~120 ms tau reaches 180 deg
of phase lag. The loop has almost no derivative damping, and D is the closest available
analogue to the kd term that keeps other position-controlled robots quiet. RAISE it in
steps from stock 15 (try 24, then 32), ARMS FIRST where nothing can fall. Change ONE thing
at a time and re-read before/after.

This tool READS the current registers first (always, even when writing) so you can revert,
and READS BACK after writing to confirm it stuck. EEPROM is LOCK-protected (reg 48):
unlock(0) -> write -> relock(1). Defaults to the LEGS+WAIST (the load-bearing joints).

  python3 servo_stiffness.py --read-only --all           # inspect every servo (do this FIRST)
  python3 servo_stiffness.py --arms --kd 24 --apply      # damping step 1: arms only, D=24
  python3 servo_stiffness.py --arms --kd 32 --apply      # damping step 2 if 24 helped
  python3 servo_stiffness.py --legs --kd 24 --apply      # only after the arm result is in
  python3 servo_stiffness.py --all --kp 15 --kd 15 --apply # REVERT to stock P/D
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
