#!/usr/bin/env python3
# fix_arm_limits.py  --  RUN ON THE PI
"""
Fix the EEPROM angle limits on the 4 arm servos after the 6/22 roll<->elbow
servo-ID reconciliation. The original set_limits.py put the one-sided ELBOW
range on servos 13 & 16, but those are actually the SHOULDER-ROLL servos; the
real elbows are servos 14 & 17. This swaps the limits to match:

  servo 13  R_shoulder_roll : [411, 613]  symmetric   (was 512..818 elbow)
  servo 14  R_elbow         : [512, 818]  one-sided +  (was 411..613)
  servo 16  L_shoulder_roll : [411, 613]  symmetric   (was 206..512 elbow)
  servo 17  L_elbow         : [206, 512]  one-sided -  (was 411..613)

Sets ABSOLUTE EEPROM min/max (reg 0x09/0x0B, big-endian units). No motion.

  python3 fix_arm_limits.py
"""
import serial
import time

PORT = "/dev/ttyAMA0"
BAUD = 1_000_000

LIMITS = {13: (411, 613), 14: (512, 818), 16: (411, 613), 17: (206, 512)}


def checksum(body):
    return (~sum(body)) & 0xFF


def write_reg(ser, sid, reg, data):
    params = [reg] + data
    length = len(params) + 2
    body = [sid, length, 0x03] + params
    ser.write(bytes([0xFF, 0xFF] + body + [checksum(body)]))
    time.sleep(0.01)
    ser.read(64)


def set_limits(ser, sid, lo, hi):
    lo = max(0, min(1023, lo))
    hi = max(0, min(1023, hi))
    write_reg(ser, sid, 0x30, [0])           # unlock EEPROM
    time.sleep(0.01)
    write_reg(ser, sid, 0x09, [(lo >> 8) & 0xFF, lo & 0xFF,
                               (hi >> 8) & 0xFF, hi & 0xFF])
    time.sleep(0.01)
    write_reg(ser, sid, 0x30, [1])           # lock EEPROM
    time.sleep(0.02)


def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    try:
        for sid, (lo, hi) in LIMITS.items():
            set_limits(ser, sid, lo, hi)
            print(f"ID {sid:2d}: EEPROM limits -> {lo} .. {hi}")
    finally:
        ser.close()
    print("Done. Re-read with a position check to confirm.")


if __name__ == "__main__":
    main()
