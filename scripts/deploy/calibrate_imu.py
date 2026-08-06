#!/usr/bin/env python3
# calibrate_imu.py  --  RUN ON THE PI
"""
Derive the IMU axis_remap (raw sensor axes -> robot PELVIS/base frame) by measuring
gravity in three known poses, then write it to config/imu_calib.yaml.

The pelvis frame (matches the sim base_link the policy obs is defined in):
    +X = forward (robot's front)
    +Y = left
    +Z = up
proj_grav (gravity direction in pelvis coords) is therefore:
    upright            -> [ 0,  0, -1]   (gravity points down = -Z)
    front facing floor -> [ 1,  0,  0]   (gravity points along +X)
    left  facing floor -> [ 0,  1,  0]   (gravity points along +Y)

We want:  proj_grav = axis_remap @ (-normalize(accel))
The accel reads +1g UP at rest, so -normalize(accel) is "down" in the raw sensor frame.
With three poses:  axis_remap = G @ inv(M),  where the columns of
    M = the measured down-vectors (sensor frame), one per pose
    G = the known proj_grav targets above
rounded to the nearest signed permutation (clean +/-1 entries).

Two ways to run:

  Interactive (at a real terminal, all three poses in one go):
    python3 calibrate_imu.py            # measure + print
    python3 calibrate_imu.py --apply    # also write the yaml

  Pose-by-pose (drivable over a non-interactive SSH session): pose the robot, then
  run ONE capture per pose; finally solve:
    python3 calibrate_imu.py --capture UPRIGHT
    python3 calibrate_imu.py --capture FRONT-DOWN
    python3 calibrate_imu.py --capture LEFT-DOWN
    python3 calibrate_imu.py --solve --apply
    python3 calibrate_imu.py --list     # show what's captured
    python3 calibrate_imu.py --reset    # clear captures
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware import IMU  # noqa: E402

# label -> (instruction, proj_grav target in pelvis frame)
POSES = {
    "UPRIGHT":    ("Stand the robot upright, normally (pelvis +Z pointing UP).",        np.array([0.0, 0.0, -1.0])),
    "FRONT-DOWN": ("Tip it forward so the FRONT faces straight down (pelvis +X DOWN).", np.array([1.0, 0.0,  0.0])),
    "LEFT-DOWN":  ("Lay it on its LEFT side so the LEFT faces down (pelvis +Y DOWN).",  np.array([0.0, 1.0,  0.0])),
}
DEFAULT_STATE = "/tmp/imu_calib_poses.json"


def measure_down(secs=1.5, hz=100.0):
    """Average raw accel while held still, return unit 'down' vector in sensor frame."""
    imu = IMU().connect()                 # identity remap -> RAW sensor axes
    n = max(1, int(secs * hz))
    acc = np.zeros(3, dtype=np.float64)
    for _ in range(n):
        acc += imu._read_accel_g()
        time.sleep(1.0 / hz)
    imu.close()
    acc /= n
    up = acc / np.linalg.norm(acc)        # accel points UP at rest
    return (-up).tolist()                 # down in sensor frame


def load_state(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(path, state):
    Path(path).write_text(json.dumps(state, indent=2))


def solve(state):
    """Return (remap int 3x3, remap_f float, det, resid, valid). Raises if poses missing."""
    missing = [k for k in POSES if k not in state]
    if missing:
        raise ValueError(f"missing captures: {missing} (have {list(state)})")
    M_cols, G_cols = [], []
    for label, (_, g) in POSES.items():
        M_cols.append(np.asarray(state[label], dtype=np.float64))
        G_cols.append(g)
    M = np.column_stack(M_cols)
    G = np.column_stack(G_cols)
    remap_f = G @ np.linalg.inv(M)
    remap = np.rint(remap_f).astype(int)

    valid = True
    if not np.array_equal(np.sort(np.abs(remap).sum(axis=1)), [1, 1, 1]):
        valid = False
    if not np.array_equal(np.sort(np.abs(remap).sum(axis=0)), [1, 1, 1]):
        valid = False
    det = int(round(np.linalg.det(remap)))
    if det != 1:
        valid = False
    resid = float(np.max(np.abs(remap_f - remap)))
    return remap, remap_f, det, resid, valid


def yaml_for(remap):
    rows = "\n".join(f"  - [{r[0]}, {r[1]}, {r[2]}]" for r in remap)
    return (
        "# IMU mounting calibration for the real robot.\n"
        "# axis_remap maps raw IMU sensor axes -> the PELVIS/base frame:\n"
        "#     proj_grav = axis_remap @ (-accel_normalized)\n"
        "#     ang_vel   = axis_remap @ gyro_rads - gyro_bias\n"
        f"# Derived {time.strftime('%Y-%m-%d')} by calibrate_imu.py (3-pose gravity method).\n"
        "# IMU mounted on the PELVIS/base block (obs frame), so no waist correction is needed.\n"
        "axis_remap:\n" + rows + "\n"
    )


def report(remap, remap_f, det, resid, valid):
    print("Raw solved matrix (pre-rounding):")
    print(np.array2string(remap_f, precision=2, suppress_small=True))
    print("\nSnapped axis_remap (raw IMU axes -> pelvis frame):")
    print(remap)
    print(f"\nrounding residual (max off-integer): {resid:.3f}  (want < ~0.25)")
    print(f"determinant: {det}  (want +1; -1 => an axis is flipped / wrong pose)")
    if not valid:
        print("\n*** INVALID remap: a pose was off or front/left was reversed. Re-capture.")
    elif resid > 0.3:
        print("\n*** Poses were sloppy (large residual). Re-capture more level.")
    else:
        print("\nVALID signed-permutation remap.")
    return valid and resid <= 0.3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "config" / "imu_calib.yaml"))
    p.add_argument("--state", default=DEFAULT_STATE, help="where pose captures are stored")
    p.add_argument("--secs", type=float, default=1.5, help="averaging window per pose")
    p.add_argument("--capture", metavar="POSE", help=f"capture one pose: {list(POSES)}")
    p.add_argument("--solve", action="store_true", help="solve from captured poses")
    p.add_argument("--apply", action="store_true", help="write the result to --out")
    p.add_argument("--list", action="store_true", help="show captured poses")
    p.add_argument("--reset", action="store_true", help="clear captured poses")
    args = p.parse_args()

    # ---- pose-by-pose mode ----
    if args.reset:
        Path(args.state).unlink(missing_ok=True)
        print(f"cleared {args.state}")
        return
    if args.list:
        print(json.dumps(load_state(args.state), indent=2))
        return
    if args.capture:
        label = args.capture.upper()
        if label not in POSES:
            print(f"unknown pose {label!r}; valid: {list(POSES)}")
            return
        print(f"Capturing {label}: {POSES[label][0]}")
        d = measure_down(secs=args.secs)
        dom = int(np.argmax(np.abs(d)))
        print(f"  measured down (sensor): [{d[0]:+.3f} {d[1]:+.3f} {d[2]:+.3f}]  "
              f"-> dominant {'XYZ'[dom]}{'+' if d[dom] > 0 else '-'}")
        state = load_state(args.state)
        state[label] = d
        save_state(args.state, state)
        print(f"  saved. captured so far: {list(state)}")
        return
    if args.solve:
        remap, remap_f, det, resid, valid = solve(load_state(args.state))
        ok = report(remap, remap_f, det, resid, valid)
        if ok and args.apply:
            Path(args.out).write_text(yaml_for(remap))
            print(f"\nWrote {args.out}")
        elif ok:
            print("\n(re-run with --apply to write the yaml)")
        return

    # ---- interactive mode (default) ----
    state = {}
    for label, (instr, _) in POSES.items():
        print(f"\n== Pose: {label} ==\n   {instr}")
        input("   Hold it steady, then press ENTER...")
        d = measure_down(secs=args.secs)
        print(f"   measured down (sensor): [{d[0]:+.3f} {d[1]:+.3f} {d[2]:+.3f}]")
        state[label] = d
    remap, remap_f, det, resid, valid = solve(state)
    ok = report(remap, remap_f, det, resid, valid)
    if ok:
        print("\n--- imu_calib.yaml ---\n" + yaml_for(remap))
        if args.apply:
            Path(args.out).write_text(yaml_for(remap))
            print(f"Wrote {args.out}")
        else:
            print("(re-run with --apply to write the file)")


if __name__ == "__main__":
    main()
