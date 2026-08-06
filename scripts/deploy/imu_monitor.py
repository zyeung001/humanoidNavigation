#!/usr/bin/env python3
# imu_monitor.py  --  RUN ON THE PI
"""
Live IMU readout for confirming the projected-gravity POLARITY on every axis -- the one
sim-to-real seam check we never closed (upright only confirms the vertical axis).

NO servos are touched. Just streams projected_gravity (obs frame) + angular velocity so
you can tilt the robot by hand and confirm the signs move the RIGHT way:

  - Hold upright:        pg ~ [ 0.00,  0.00, -1.00]
  - Tip FORWARD (face down a bit):   pg[0] should go POSITIVE (toward +x = forward lean)
  - Tip BACKWARD:                    pg[0] should go NEGATIVE
  - Tip to its RIGHT:                pg[1] should move one way, LEFT the other (note which)

If a lateral axis moves the WRONG way (or the wrong axis responds), the axis_remap in
config/imu_calib.yaml has a sign/permutation error -> the policy corrects the wrong
direction the instant it leans, which looks exactly like a seizure. Fix the remap, not the
policy. If every axis tracks correctly, the loop is clean and the problem is plant/mechanics.

  python3 scripts/deploy/imu_monitor.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware import IMU  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    calib = ROOT / "config" / "imu_calib.yaml"
    axis_remap = None
    if calib.exists():
        import yaml
        axis_remap = np.asarray(yaml.safe_load(open(calib))["axis_remap"], dtype=np.float32)
        print(f"axis_remap from {calib}:\n{axis_remap}")
    imu = IMU(axis_remap=axis_remap).connect()
    print("\nTilt the robot by hand and watch the signs. Ctrl-C to stop.")
    print("upright -> pg~[0,0,-1]; lean FORWARD -> pg[0] should go POSITIVE.\n")
    try:
        while True:
            pg = imu.projected_gravity()
            av = imu.angular_velocity()
            lean = "FWD " if pg[0] > 0.15 else ("BACK" if pg[0] < -0.15 else "    ")
            side = "RIGHT" if pg[1] > 0.15 else ("LEFT " if pg[1] < -0.15 else "     ")
            print(f"\rpg=[{pg[0]:+.2f},{pg[1]:+.2f},{pg[2]:+.2f}]  "
                  f"|av|={np.linalg.norm(av):.2f}   lean={lean} side={side}   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
