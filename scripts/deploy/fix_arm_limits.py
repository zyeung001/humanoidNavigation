#!/usr/bin/env python3
# fix_arm_limits.py  --  RUN ON THE PI
"""
Set the EEPROM angle limits on all SIX arm servos, RE-ANCHORED to the measured
straight pose (8/5 calibration). Sets ABSOLUTE EEPROM min/max (reg 0x09/0x0B,
big-endian units), reads them back to confirm, and moves nothing.

  python3 fix_arm_limits.py

WHY THIS CHANGED 8/5. Every previous version of this table assumed the arms are
straight at 512. `calibrate_arm_centers.py` measured the real straight pose and
that assumption was wrong by up to 40 deg, so the limits were centred on the
wrong point and THREE joints could not reach straight at all:

  servo 12 R_shoulder_pitch  straight 380 vs min 412  -- short  32 units ( 9.4 deg)
  servo 13 R_elbow           straight 418 vs min 512  -- short  94 units (27.6 deg)
  servo 15 L_shoulder_pitch  straight 648 vs max 612  -- over   36 units (10.6 deg)

i.e. the shoulder pitches could only travel in a +/-30 deg window centred ~39 deg
away from straight, so the arm physically could not straighten -- which is exactly
the reported symptom (arms hang visibly bent while the policy reports it is on
target). Same silent-clip bug class as the 7/28 L_shoulder_roll ctrlrange fault.

Each limit is the joint's OLD span shifted by that joint's measured centre offset,
so the physical travel arc is preserved exactly and is now correctly labelled --
the same operation the 6/24 re-zero did for the legs and waist. Spans are
deliberately unchanged (one variable at a time); the shoulder pitches keep the
+/-101-unit placeholder span from 6/18, which still covers every demo pose.

  sid  joint             offset   old span      new span
   12  R_shoulder_pitch   -132   411 .. 613    279 .. 481   symmetric
   14  R_shoulder_roll     +18   411 .. 613    429 .. 631   symmetric
   13  R_elbow             -94   512 .. 818    418 .. 724   one-sided +
   15  L_shoulder_pitch   +136   411 .. 613    547 .. 749   symmetric
   17  L_shoulder_roll      -3   411 .. 613    408 .. 610   symmetric
   16  L_elbow            -127   206 .. 512     79 .. 385   one-sided -

ORDER MATTERS: run this BEFORE putting the new `center:` values into
config/joint_servo_map.yaml. With the old centre of 512 nothing clips (512 sits
inside every old limit); writing the new centres while the old limits are still
in the firmware is what would create the clipping.
"""
import serial
import time

PORT = "/dev/ttyAMA0"
BAUD = 1_000_000

# Limits follow the PHYSICAL SLOT (8/4: the rebuild swapped roll<->elbow on BOTH arms --
# servo 13 sits in the R elbow, 16 in the L elbow; bus IDs live in servo EEPROM, not the
# cabling), re-anchored 8/5 to the measured straight pose. See the module docstring.
LIMITS = {
    12: (279, 481),   # R_shoulder_pitch  straight 380, symmetric +/-101
    14: (429, 631),   # R_shoulder_roll   straight 530, symmetric +/-101
    13: (418, 724),   # R_elbow           straight 418 = the one-sided stop, flexes +
    15: (547, 749),   # L_shoulder_pitch  straight 648, symmetric +/-101
    17: (408, 610),   # L_shoulder_roll   straight 509, symmetric +/-101
    16: (79, 385),    # L_elbow           straight 385 = the one-sided stop, flexes -
}
STRAIGHT = {12: 380, 14: 530, 13: 418, 15: 648, 17: 509, 16: 385}


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


def read_limits(ser, sid):
    """Read back EEPROM min/max (reg 0x09, 4 bytes) -> (lo, hi) or None."""
    body = [sid, 4, 0x02, 0x09, 4]
    ser.reset_input_buffer()
    ser.write(bytes([0xFF, 0xFF] + body + [checksum(body)]))
    time.sleep(0.01)
    r = ser.read(10)
    if len(r) != 10 or r[0] != 0xFF or r[1] != 0xFF or r[2] != sid:
        return None
    return ((r[5] << 8) | r[6], (r[7] << 8) | r[8])


def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    bad = []
    try:
        for sid, (lo, hi) in LIMITS.items():
            set_limits(ser, sid, lo, hi)
            got = read_limits(ser, sid)
            s = STRAIGHT[sid]
            if got is None:
                status, ok = "NO REPLY on read-back", False
            elif got != (lo, hi):
                status, ok = f"FAILED -- reads {got[0]}..{got[1]}", False
            elif not (got[0] <= s <= got[1]):
                status, ok = f"WROTE OK but straight {s} is OUTSIDE the range", False
            else:
                status, ok = f"OK (straight {s} reachable, -{s - lo}/+{hi - s})", True
            print(f"ID {sid:2d}: EEPROM limits -> {lo:3d} .. {hi:3d}   {status}")
            if not ok:
                bad.append(sid)
    finally:
        ser.close()
    if bad:
        print(f"\n!! {len(bad)} servo(s) NOT verified: {bad}. Do NOT put the new `center:` "
              "values in the map until this reads clean -- the joints would clip silently.")
        return 1
    print("\nAll six verified. Safe to apply the new `center:` values to "
          "config/joint_servo_map.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
