#!/usr/bin/env python3
"""Make the leg pair concentric with the hip pair, which on the real robot it is.

THE DEFECT. In models/humanoid_real_v2.xml the two hip-roll pivots and the two feet are
each correctly spaced -- 178.9 mm hip to hip, 163.9 mm foot to foot -- but the two pairs
are not concentric. The hip pair sits 6.92 mm sideways of the foot pair, which splits as
0.6 mm of inboard offset on the right leg and 14.4 mm on the left. A symmetric robot has
7.5 mm on both.

MEASURED ON THE ROBOT (9/19): torque off, legs straight, the two sides come out the same
to about 1 mm. So the robot is symmetric here and the model is not. This is a CAD export
artefact, not a property of the machine.

WHAT IT IS NOT. It is not the cause of the hip-roll authority asymmetry reported earlier
in that session. Kinematically the two hip rolls are already mirror images (-16.39 and
+16.46 mm of COM shift per 5 deg); the 14x figure came from measuring through a settle,
which lets the robot rebalance and tip between feet and so mixes geometry with balance.
Nor does this explain the lateral COM offset: correcting the skew moves it from +13.2 mm
to +10.8 mm only, and 10.8 mm on a 414.7 mm COM height is 1.49 deg of lean against the
+1.08 deg the robot actually measures -- so most of that offset looks real.

THE CORRECTION (--mode pivot, the default). Both hip-roll AXES move laterally by the skew.
A joint's pos relocates the rotation axis and moves no geometry, so it cannot introduce
interference; what it changes is the moment arm from each roll axis to its own foot, which
is exactly the quantity the ruler measured. Both separations are preserved and the model
still passes test_model_integrity 13/13.

THE CORRECTION THAT WAS TRIED FIRST AND REJECTED (--mode leg). Moving the lower legs to
the hips instead is geometrically equivalent on paper, and it breaks the model: the right
hip-yaw bracket has only about 5.7 mm of clearance to the pelvis, so a 6.92 mm inboard
shift buries it 1.24 mm inside base_link and integrity drops to 12/13 on resting
self-collision. It stays behind the flag because that failure is the evidence for moving
the axes rather than the legs.

Deltas are rotated into the owning frame, since pos is expressed there and these CAD
frames are not axis-aligned.

  python scripts/debug/fix_leg_symmetry.py                 # report only, writes nothing
  python scripts/debug/fix_leg_symmetry.py --write         # correct models/humanoid_real_v2.xml
  python scripts/debug/make_compliant_waist.py             # then regenerate the compliant model
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE_XML = ROOT / "models" / "humanoid_real_v2.xml"
# First body under each hip-roll joint. Shifting these moves the lower leg and leaves the
# roll pivot untouched; shifting the roll body itself would move the pivot with it and
# change nothing.
HIP_YAW_BODIES = ("0003_11", "0003_3")
# The hip-roll joints themselves, for the default pivot correction.
HIP_ROLL_JOINTS = {"R_hip_roll": "Revolute 18", "L_hip_roll": "Revolute 24"}


def measure(path):
    """Skew, separations and per-leg inboard offsets, in mm, from a settled-free pose."""
    import mujoco  # noqa: PLC0415
    import yaml  # noqa: PLC0415
    sys.path.insert(0, str(ROOT / "scripts" / "debug"))
    from sim_cop import _foot_geoms, forward_axis  # noqa: PLC0415

    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    d.qpos[:] = 0
    d.qpos[3] = 1.0
    mujoco.mj_forward(m, d)
    fwd = forward_axis(m, d)
    lat = np.cross([0.0, 0.0, 1.0], fwd)
    dofs = [j["dof"] for j in
            yaml.safe_load(open(ROOT / "config" / "joint_servo_map.yaml"))["joints"]]
    anchor = {n: d.xanchor[m.actuator_trnid[i, 0]] for i, n in enumerate(dofs)}
    feet = sorted(_foot_geoms(m))
    P = lambda v: 1000.0 * (v @ lat)  # noqa: E731

    hips = (P(anchor["R_hip_roll"]), P(anchor["L_hip_roll"]))
    # Each foot belongs to whichever knee sits above it. Pairing them by index would be a
    # coin flip on a CAD export, and the whole point here is a left-right comparison.
    foot = tuple(P(d.geom_xpos[min(feet, key=lambda g: abs(
        P(d.geom_xpos[g]) - P(anchor[knee])))]) for knee in ("R_knee", "L_knee"))
    hip_mid, foot_mid = 0.5 * sum(hips), 0.5 * sum(foot)
    return {
        "skew": hip_mid - foot_mid,
        "hip_sep": abs(hips[1] - hips[0]),
        "foot_sep": abs(foot[1] - foot[0]),
        # Each leg against ITS OWN foot, not against a midpoint -- measuring both sides
        # from their own midpoints cancels the skew by construction and reports 7.52 on
        # both legs however badly skewed the model is. These are the two numbers a ruler
        # checks on the robot, and mirror symmetry means equal magnitude, opposite sign.
        "inboard_R": hips[0] - foot[0],
        "inboard_L": hips[1] - foot[1],
        "lat": lat, "model": m, "data": d,
    }


def show(tag, r):
    print(f"  {tag:8s} skew {r['skew']:+6.2f} mm   hip sep {r['hip_sep']:6.1f}   "
          f"foot sep {r['foot_sep']:6.1f}   inboard R {r['inboard_R']:+5.2f} / "
          f"L {r['inboard_L']:+5.2f}")


def main():
    import mujoco  # noqa: PLC0415

    p = argparse.ArgumentParser(description="Make the leg pair concentric with the hips")
    p.add_argument("--xml", default=str(BASE_XML))
    p.add_argument("--out", default=None, help="default: overwrite --xml")
    p.add_argument("--mode", choices=("pivot", "leg"), default="pivot",
                   help="pivot (default): move the hip-roll AXES to the legs, moving no "
                        "geometry. leg: move the lower legs to the hips -- rejected 9/19, "
                        "it buries the right hip bracket in the pelvis.")
    p.add_argument("--write", action="store_true",
                   help="actually write. Without it this only reports what it would do.")
    args = p.parse_args()

    src = Path(args.xml)
    before = measure(src)
    print(f"\n{src}")
    show("BEFORE", before)

    if abs(before["skew"]) < 0.05:
        print("\n  already concentric -- nothing to do.")
        return 0

    m, d = before["model"], before["data"]
    tree = ET.parse(src)
    moved = {}

    if args.mode == "leg":
        # Move the lower legs to the hips. REJECTED on 9/19: the right hip-yaw bracket has
        # only ~5.7 mm of clearance to the pelvis, so shifting it 6.92 mm inboard buries it
        # 1.24 mm inside base_link and test_model_integrity drops to 12/13 on resting
        # self-collision. Kept behind the flag because the failure is the evidence.
        delta_world = (before["skew"] / 1000.0) * before["lat"]
        for name in HIP_YAW_BODIES:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise SystemExit(f"{src}: no body {name!r} -- has the model been rebuilt?")
            parent_R = d.xmat[m.body_parentid[bid]].reshape(3, 3)
            new_pos = m.body_pos[bid] + parent_R.T @ delta_world
            moved[name] = ("body pos", m.body_pos[bid].copy(), new_pos)
            node = next((b for b in tree.iter("body") if b.get("name") == name), None)
            if node is None:
                raise SystemExit(f"{src}: body {name!r} compiled but not in the XML")
            node.set("pos", " ".join(f"{v:.9g}" for v in new_pos))
        print(f"\n  shifting the lower legs {before['skew']:+.2f} mm laterally:")
    else:
        # Move the hip-roll AXES to the legs. A joint's pos relocates the rotation axis and
        # moves no geometry at all, so it cannot introduce interference -- which is exactly
        # why this is the default. What it changes is the moment arm from each roll axis to
        # its own foot, and that arm is the quantity the ruler measured.
        delta_world = -(before["skew"] / 1000.0) * before["lat"]
        for dof, jname in HIP_ROLL_JOINTS.items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise SystemExit(f"{src}: no joint {jname!r} -- has the model been rebuilt?")
            body_R = d.xmat[m.jnt_bodyid[jid]].reshape(3, 3)
            new_pos = m.jnt_pos[jid] + body_R.T @ delta_world
            moved[dof] = ("joint pos", m.jnt_pos[jid].copy(), new_pos)
            node = next((j for b in tree.iter("body") for j in b.findall("joint")
                         if j.get("name") == jname), None)
            if node is None:
                raise SystemExit(f"{src}: joint {jname!r} compiled but not in the XML")
            node.set("pos", " ".join(f"{v:.9g}" for v in new_pos))
        print(f"\n  moving the hip-roll axes {-before['skew']:+.2f} mm laterally "
              "(no geometry moves):")
    for name, (what, old, new) in moved.items():
        print(f"    {name:12s} {what} {np.array2string(old, precision=6)} -> "
              f"{np.array2string(new, precision=6)}")

    if not args.write:
        print("\n  DRY RUN -- nothing written. Re-run with --write to apply, then")
        print("  regenerate the compliant model: python scripts/debug/make_compliant_waist.py")
        return 0

    out = Path(args.out) if args.out else src
    tree.write(out, encoding="unicode")
    after = measure(out)
    show("AFTER", after)
    ok = (abs(after["skew"]) < 0.05
          and abs(abs(after["inboard_R"]) - abs(after["inboard_L"])) < 0.1)
    print(f"\n  wrote {out}")
    print("  " + ("symmetric now." if ok else "!! STILL ASYMMETRIC -- inspect before using"))
    print("  Regenerate the compliant model: python scripts/debug/make_compliant_waist.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
