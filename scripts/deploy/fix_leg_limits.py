#!/usr/bin/env python3
# fix_leg_limits.py  --  RUN ON THE PI
"""
Re-anchor the EEPROM angle limits on the ELEVEN leg and waist servos to the measured
straight pose (9/3 calibration). Sets absolute EEPROM min/max (reg 0x09, big-endian
units), reads them back to confirm, and moves nothing.

  python3 scripts/deploy/fix_leg_limits.py            # DRY RUN: show, write nothing
  python3 scripts/deploy/fix_leg_limits.py --write    # actually write, then verify

WHY. Every leg and waist limit was centred on an ASSUMED straight of 512. On 9/3
calibrate_centers.py measured the real straight pose and six joints were off by more than
posing noise, the worst by 27 units (7.9 deg) -- against a residual clamp half-width of
0.20 rad = 39 units. Limits centred on the wrong point are the silent-clip bug: the servo
stops at a firmware boundary it was never told about, applying zero force and reporting no
error, while the policy and the servo agree they are on target. That is exactly what left
three arm joints unable to reach straight until 8/5.

Each limit is the joint's OLD SPAN re-centred on its MEASURED straight, so the physical
travel arc is preserved exactly and merely relabelled. Spans are deliberately unchanged --
one variable at a time.

  sid  joint            straight   old EEPROM      new EEPROM     shift
    1  waist_yaw            528    411 .. 613     427 .. 629      +16
    2  waist_pitch          488    412 .. 614     387 .. 589      -24
    3  waist_roll           494    414 .. 616     420 .. 622       +9
    4  R_hip_roll           499    412 .. 614     398 .. 600      -13
    5  R_hip_yaw            503    411 .. 705     402 .. 691       -9
    6  R_hip_pitch          513    414 .. 616     412 .. 614       +1
    7  R_knee               510    206 .. 512     204 .. 520       -2   one-sided +10
    8  L_hip_roll           533    412 .. 614     432 .. 634      +21
    9  L_hip_yaw            485    411 .. 740     384 .. 708      -27
   10  L_hip_pitch          496    413 .. 615     395 .. 597      -16
   11  L_knee               520    516 .. 822     510 .. 826       +8   one-sided -10

The two knees are one-sided hinges: straight IS the mechanical stop, so their straight sits
at the END of the new range rather than in the middle. They also get 10 units of margin
PAST that stop, which the arms did not. The arms' stops were measured with a spread of 0,
so placing a limit exactly on one was safe; the knees came back with spreads of 9 and 18
units, so the true stop may lie a little beyond the best round anyone managed. A limit
placed exactly on an under-measured stop would hold the knee short of straight -- the very
silent clip this script exists to prevent. Margin past a MECHANICAL stop costs nothing: the
stop still stops the joint, and the policy cannot command past it anyway, because sim_range
for these joints ends at 0 (straight).

ORDER MATTERS. Run this BEFORE putting the new `center:` values into
config/joint_servo_map.yaml. With the old centre of 512 nothing clips, because 512 sits
inside every old limit; writing the new centres while the OLD limits are still in the
firmware is what would create the clipping this exists to prevent.

AFTER RUNNING. The servos now clamp commands to a different absolute range, so a joint
sitting outside its new range will move the moment torque is enabled. Keep the robot
supported and bring it back with home.py rather than a policy.
"""
import argparse
import time

# pyserial is imported inside main(), only on the write path. The whole point of the dry
# run is to audit the numbers that are about to reach EEPROM, and that review should be
# possible on the dev box -- where there is no robot and no pyserial -- not only on the Pi
# with the robot already on the bench.

PORT = "/dev/ttyAMA0"
BAUD = 1_000_000

# Derived from config/calibration/center_calibration_lower_20260903-172501.yaml and
# center_calibration_idx3-6-7-8_20260903-173239.yaml: new = measured straight + the old
# span measured from the old centre. Written out explicitly so the numbers that reach
# EEPROM can be audited here, not recomputed from files at write time.
LIMITS = {
    1: (427, 629),    # waist_yaw     straight 528, symmetric +/-101
    2: (387, 589),    # waist_pitch   straight 488, symmetric +/-101
    3: (420, 622),    # waist_roll    straight 494, -74/+128 (its centre was 485, not 512)
    4: (398, 600),    # R_hip_roll    straight 499, symmetric +/-101
    5: (402, 691),    # R_hip_yaw     straight 503, -101/+188 (widened 6/21 for the keyframe)
    6: (412, 614),    # R_hip_pitch   straight 513, symmetric +/-101
    7: (204, 520),    # R_knee        straight 510 = the one-sided stop, flexes -; +10 margin
    8: (432, 634),    # L_hip_roll    straight 533, symmetric +/-101
    9: (384, 708),    # L_hip_yaw     straight 485, -101/+223 (widened 6/21)
    10: (395, 597),   # L_hip_pitch   straight 496, symmetric +/-101
    11: (510, 826),   # L_knee        straight 520 = the one-sided stop, flexes +; -10 margin
}
STRAIGHT = {1: 528, 2: 488, 3: 494, 4: 499, 5: 503, 6: 513,
            7: 510, 8: 533, 9: 485, 10: 496, 11: 520}
NAME = {1: "waist_yaw", 2: "waist_pitch", 3: "waist_roll", 4: "R_hip_roll",
        5: "R_hip_yaw", 6: "R_hip_pitch", 7: "R_knee", 8: "L_hip_roll",
        9: "L_hip_yaw", 10: "L_hip_pitch", 11: "L_knee"}
ONE_SIDED = {7, 11}
KNEE_MARGIN = 10   # units of slack past a measured mechanical stop; see above


def checksum(body):
    return (~sum(body)) & 0xFF


def write_reg(ser, sid, reg, data):
    params = [reg] + data
    body = [sid, len(params) + 2, 0x03] + params
    ser.write(bytes([0xFF, 0xFF] + body + [checksum(body)]))
    time.sleep(0.01)
    ser.read(64)


def set_limits(ser, sid, lo, hi):
    lo = max(0, min(1023, lo))
    hi = max(0, min(1023, hi))
    write_reg(ser, sid, 0x30, [0])            # unlock EEPROM
    time.sleep(0.01)
    write_reg(ser, sid, 0x09, [(lo >> 8) & 0xFF, lo & 0xFF,
                               (hi >> 8) & 0xFF, hi & 0xFF])
    time.sleep(0.01)
    write_reg(ser, sid, 0x30, [1])            # lock EEPROM
    time.sleep(0.02)


def read_limits(ser, sid):
    """EEPROM min/max (reg 0x09, 4 bytes) -> (lo, hi), or None if the servo stayed quiet."""
    body = [sid, 4, 0x02, 0x09, 4]
    ser.reset_input_buffer()
    ser.write(bytes([0xFF, 0xFF] + body + [checksum(body)]))
    time.sleep(0.01)
    r = ser.read(10)
    if len(r) != 10 or r[0] != 0xFF or r[1] != 0xFF or r[2] != sid:
        return None
    return ((r[5] << 8) | r[6], (r[7] << 8) | r[8])


def sanity_check():
    """Refuse to write anything that cannot be right, before the bus is even opened."""
    problems = []
    for sid, (lo, hi) in LIMITS.items():
        s = STRAIGHT[sid]
        if not 0 <= lo < hi <= 1023:
            problems.append(f"servo {sid}: {lo}..{hi} is not a valid range")
        if not lo <= s <= hi:
            problems.append(f"servo {sid} ({NAME[sid]}): straight {s} outside {lo}..{hi}")
        if sid in ONE_SIDED and min(abs(s - lo), abs(s - hi)) > KNEE_MARGIN:
            problems.append(f"servo {sid} ({NAME[sid]}): one-sided, but straight {s} is not "
                            f"within {KNEE_MARGIN} units of either end of {lo}..{hi}")
    return problems


def main():
    p = argparse.ArgumentParser(description="Re-anchor leg/waist EEPROM limits")
    p.add_argument("--write", action="store_true",
                   help="actually write. Without this the script only shows what it would "
                        "do; EEPROM writes are permanent, so they are never the default.")
    p.add_argument("--port", default=PORT)
    args = p.parse_args()

    problems = sanity_check()
    if problems:
        print("REFUSING TO RUN -- the table itself is inconsistent:")
        for x in problems:
            print("  " + x)
        return 2

    print(f"{'sid':>3} {'joint':13s} {'straight':>8} {'new limits':>14}   note")
    print("-" * 62)
    for sid in sorted(LIMITS):
        lo, hi = LIMITS[sid]
        note = (f"one-sided: stop at {STRAIGHT[sid]}, +{KNEE_MARGIN} units of margin"
                if sid in ONE_SIDED else "")
        print(f"{sid:>3} {NAME[sid]:13s} {STRAIGHT[sid]:8d} {f'{lo} .. {hi}':>14}   {note}")

    if not args.write:
        print("\nDRY RUN -- nothing was written. Re-run with --write to apply.")
        print("Before you do: the robot should be SUPPORTED and its torque off. Changing a")
        print("limit changes what the servo will accept, so a joint currently outside its")
        print("new range will move as soon as torque comes back.")
        return 0

    import serial  # noqa: PLC0415 -- write path only; see the note at the top

    ser = serial.Serial(args.port, BAUD, timeout=0.1)
    written, bad = [], []
    try:
        for sid in sorted(LIMITS):
            lo, hi = LIMITS[sid]
            before = read_limits(ser, sid)
            set_limits(ser, sid, lo, hi)
            got = read_limits(ser, sid)
            s = STRAIGHT[sid]
            if got is None:
                status, ok = "NO REPLY on read-back", False
            elif got != (lo, hi):
                status, ok = f"FAILED -- reads {got[0]}..{got[1]}", False
            elif not (got[0] <= s <= got[1]):
                status, ok = f"WROTE OK but straight {s} is OUTSIDE it", False
            else:
                status, ok = f"OK  (straight reachable, -{s - lo}/+{hi - s})", True
            print(f"ID {sid:2d} {NAME[sid]:13s} "
                  f"{f'{before[0]}..{before[1]}' if before else '  ?  ':>11} -> "
                  f"{lo:3d}..{hi:3d}   {status}")
            (written if ok else bad).append(sid)
    finally:
        ser.close()

    if bad:
        print(f"\n!! {len(bad)} servo(s) NOT verified: {bad}")
        print("Do NOT put the new `center:` values into the map until this reads clean --")
        print("those joints would clip silently, with zero force and no error.")
        if written:
            print(f"({len(written)} DID write and verify: {written}. The bus is in a mixed "
                  "state; re-run to finish.)")
        return 1
    print(f"\nAll {len(written)} verified. Now apply the new `center:` values to "
          "config/joint_servo_map.yaml,")
    print("then bring the robot back with home.py before running anything else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
