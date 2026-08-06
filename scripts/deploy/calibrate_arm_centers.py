#!/usr/bin/env python3
"""calibrate_arm_centers.py  --  RUN ON THE PI

Measure the true STRAIGHT position of the six arm joints and emit the `center:` values for
config/joint_servo_map.yaml.

WHY (8/4): the arm rebuild put servos 13<->14 and 16<->17 back in each other's slots. The
map's servo_id fields were corrected, but a joint's zero is a property of how its HORN sits
on the spline, so swapping servos between slots invalidates the zeros too. Symptom: at the
`home` pose the policy reports only 0.06-0.11 rad of error while the arms visibly hang bent
-- servos and policy agree they are on target, and the target is not straight.

Torque is OFF the whole time; nothing is driven. You pose the arms by hand, this reads the
encoders, medians several samples per joint to reject bus noise, and prints YAML to paste.

    python3 scripts/deploy/calibrate_arm_centers.py
    python3 scripts/deploy/calibrate_arm_centers.py --samples 15
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import yaml  # noqa: E402

from hardware import ServoBus  # noqa: E402

DEFAULT_MAP = Path(__file__).parent.parent.parent / "config" / "joint_servo_map.yaml"
ARM_IDX = range(11, 17)          # action indices of the six arm joints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(DEFAULT_MAP))
    ap.add_argument("--samples", type=int, default=9, help="reads per joint (median)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.map))
    joints = [j for j in cfg["joints"] if j["idx"] in ARM_IDX]
    ids = [int(j["servo_id"]) for j in joints]
    default_center = int(cfg.get("servo_center", 512))

    bus = ServoBus().connect()
    try:
        bus.set_torque(ids, False)
        print(f"Torque OFF on arm servos {ids}. Nothing will be driven.\n")
        print("Pose BOTH arms hanging straight down, relaxed and symmetric, then press Enter.")
        input("  [Enter] when the arms are straight: ")

        print(f"\nreading {args.samples} samples per joint...")
        rows = []
        for j in joints:
            sid = int(j["servo_id"])
            vals = []
            for _ in range(args.samples):
                p = bus.read_pos(sid)
                if p is not None:
                    vals.append(int(p))
                time.sleep(0.01)
            if not vals:
                print(f"  !! servo {sid} ({j['dof']}): NO valid reads -- check the bus")
                rows.append((j, None, None))
                continue
            med = int(statistics.median(vals))
            spread = max(vals) - min(vals)
            rows.append((j, med, spread))
            flag = "  <<NOISY, re-run" if spread > 4 else ""
            print(f"  {j['dof']:<18} servo {sid:2d}: median {med:4d}  "
                  f"spread {spread}{flag}")

        print("\n--- paste these `center:` values into config/joint_servo_map.yaml ---")
        print("(a joint whose median is already within ~2 units of "
              f"{default_center} needs no override)\n")
        for j, med, _ in rows:
            if med is None:
                continue
            off = med - default_center
            note = "  # no override needed" if abs(off) <= 2 else \
                   f"  # {off:+d} units = {off/195.0*57.2958:+.1f} deg off nominal"
            print(f"  idx {j['idx']:2d} {j['dof']:<18} center: {med}{note}")
        print("\nAlso SHIFT that joint's servo_limit by the same offset, so its sim-radian "
              "range is preserved (this is what the 6/24 re-zero did for the legs/waist).")
    finally:
        # Guarded: if the body raised, a throwing cleanup would mask the real error
        # (that is what buried the missing-connect() bug on the first run, 8/5).
        try:
            bus.set_torque(ids, False)
        except Exception as e:      # noqa: BLE001 - cleanup must not mask the original
            print(f"(cleanup: could not re-assert torque-off: {e})")
        try:
            bus.close()
        except Exception:           # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
