#!/usr/bin/env python3
# widen_hip_limits.py  --  RUN ON THE PI
"""
Widen the EEPROM angle limits on the two HIP-YAW servos so they can reach the
turned-out standing keyframe pose (and be bench-tested by verify_signs.py).

Background: set_limits.py wrote a symmetric +/-30deg (units 411..613) window on
every non-hinge joint. But the standing keyframe needs:
    R_hip_yaw (servo 5) -> +55deg  (map units ~699)
    L_hip_yaw (servo 9) -> +65deg  (map units ~733)
so both are pinned at 613 and can't reach the pose or move during sign-testing.

This sets ABSOLUTE EEPROM limits (reg 0x09=min, 0x0B=max, big-endian units) on
servos 5 and 9 ONLY, capped just above the pose so a wrong sign during verify
can't overswing past ~66deg. Matches config/joint_servo_map.yaml's widened
software servo_limit (EEPROM is set slightly wider so the software clamp binds).

  python3 widen_hip_limits.py            # write the limits
  python3 widen_hip_limits.py --revert    # restore symmetric 411..613
"""
import argparse
import serial
import time

PORT = "/dev/ttyAMA0"
BAUD = 1_000_000

# servo_id -> (min_units, max_units).  EEPROM slightly >= map servo_limit.
WIDENED = {5: (411, 705), 9: (411, 740)}   # R_hip_yaw, L_hip_yaw
SYMMETRIC = {5: (411, 613), 9: (411, 613)}  # original placeholder


def checksum(body):
    return (~sum(body)) & 0xFF


def write_reg(ser, sid, reg, data):
    params = [reg] + data
    length = len(params) + 2
    body = [sid, length, 0x03] + params      # 0x03 = WRITE
    ser.write(bytes([0xFF, 0xFF] + body + [checksum(body)]))
    time.sleep(0.01)
    ser.read(64)                              # drain status packet


def set_limits(ser, sid, lo, hi):
    lo = max(0, min(1023, lo))
    hi = max(0, min(1023, hi))
    write_reg(ser, sid, 0x30, [0])            # unlock EEPROM
    time.sleep(0.01)
    write_reg(ser, sid, 0x09, [(lo >> 8) & 0xFF, lo & 0xFF,
                               (hi >> 8) & 0xFF, hi & 0xFF])  # min(0x09)+max(0x0B)
    time.sleep(0.01)
    write_reg(ser, sid, 0x30, [1])            # lock EEPROM
    time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true", help="restore 411..613")
    args = ap.parse_args()
    table = SYMMETRIC if args.revert else WIDENED

    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    try:
        for sid, (lo, hi) in table.items():
            set_limits(ser, sid, lo, hi)
            print(f"ID {sid:2d}: EEPROM limits -> {lo} .. {hi}")
    finally:
        ser.close()
    print("Done. Run diag.py (or verify_signs.py --dry-run) to confirm.")


if __name__ == "__main__":
    main()
