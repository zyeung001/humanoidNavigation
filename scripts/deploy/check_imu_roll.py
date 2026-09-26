#!/usr/bin/env python3
# check_imu_roll.py  --  RUN ON THE PI.  Reads the IMU only; no servo moves.
"""Does the IMU report sideways lean in the same direction sim does?

WHY. On 9/26 the 195M policy fell sideways within a second of starting. Both hip rolls
pushed the weight the same way, the IMU read +8.9 deg of roll, and the policy kept pushing
in the direction it was already falling -- the signature of a controller whose sense of
"which way am I leaning" is reversed. The previous model (175M) barely steered sideways,
which would hide a reversed roll entirely.

Sim's convention, measured on the model rather than assumed: tilt the robot so ITS RIGHT
side goes down and the policy sees NEGATIVE roll (pg_y < 0), with a POSITIVE roll rate
(gyro x) while it is tipping that way. This checks the real IMU against that. Words cannot
be trusted here -- "right" depends on which side you stand -- so the test asks for one
specific physical motion and lets the sensor answer.

  python3 scripts/deploy/check_imu_roll.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware import IMU  # noqa: E402

SECONDS = 10.0


def main():
    import yaml  # noqa: PLC0415
    root = Path(__file__).resolve().parents[2]
    remap = None
    cal = root / "config" / "imu_calib.yaml"
    if cal.exists():
        remap = np.array(yaml.safe_load(open(cal))["axis_remap"], dtype=float)
    imu = IMU(axis_remap=remap).connect()

    print("\nHold the robot upright, feet off the table or lightly on it.")
    print("Its RIGHT side is the side with the right-leg servos (IDs 4-7).")
    print("  -> If you are FACING the robot, its right side is on YOUR LEFT.\n")
    print("When you see GO: tilt it so ITS RIGHT side goes DOWN by about 15 deg,")
    print("hold it there for 2 seconds, then bring it back upright.\n")
    for n in (3, 2, 1):
        print(f"  {n}...", flush=True)
        time.sleep(1.0)
    print("  GO\n", flush=True)

    t0, rows = time.time(), []
    while time.time() - t0 < SECONDS:
        pg = imu.projected_gravity()
        av = imu.angular_velocity()
        rows.append((time.time() - t0,
                     float(np.degrees(np.arctan2(pg[1], -pg[2]))),
                     float(np.degrees(np.arctan2(pg[0], -pg[2]))), float(av[0])))
        time.sleep(0.01)
    imu.close()

    t, roll, pitch, avx = (np.array(c) for c in zip(*rows))
    k = int(np.argmax(np.abs(roll)))
    print(f"peak roll {roll[k]:+.1f} deg at t={t[k]:.1f}s   (pitch then {pitch[k]:+.1f} deg)")
    if abs(roll[k]) < 5.0:
        print("\nINCONCLUSIVE: the tilt never reached 5 deg. Run it again with a bigger tilt.")
        return 2
    # roll rate while tipping INTO the peak, not while coming back out of it
    going = (t < t[k]) & (np.abs(roll) > 0.3 * abs(roll[k]))
    rate = float(np.median(avx[going])) if going.any() else float("nan")
    print(f"roll rate while tipping in: {rate:+.2f} rad/s")

    roll_ok = roll[k] < 0
    rate_ok = rate > 0 if np.isfinite(rate) else None
    print("\nSim expects, for ITS RIGHT side down: roll NEGATIVE, roll rate POSITIVE.")
    print(f"  roll sign : {'MATCHES sim' if roll_ok else 'REVERSED vs sim'}")
    if rate_ok is not None:
        print(f"  rate sign : {'MATCHES sim' if rate_ok else 'REVERSED vs sim'}")
    if roll_ok and rate_ok is not False:
        print("\nVERDICT: the IMU agrees with sim on sideways lean. The 9/26 fall is not a "
              "sensor-sign problem.")
    elif not roll_ok and rate_ok is False:
        print("\nVERDICT: sideways lean is REVERSED between the robot and sim -- both the "
              "angle and the rate. A policy that steers sideways will push itself over.")
    else:
        print("\nVERDICT: the angle and the rate DISAGREE with each other. Paste this output; "
              "the accelerometer and gyro remaps need checking separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
