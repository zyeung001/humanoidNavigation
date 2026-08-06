#!/usr/bin/env python3
"""calibrate_centers.py  --  RUN ON THE PI

Measure the true STRAIGHT encoder position of a group of joints and emit the `center:`
values (with the matching shifted `servo_limit`s) for config/joint_servo_map.yaml.

READ-ONLY. Torque is switched OFF on the measured servos and NOTHING is ever driven or
written -- not the servos, not their EEPROM, not the map. You pose the robot by hand; this
reads present position and the EEPROM angle limits, then prints numbers and saves them.

WHY THIS MATTERS. A joint's zero is a property of how its horn sits on the spline, so it is
a measured constant, not a chosen one. The map converts both directions through it --

    units   = center + sign * sim_rad * units_per_rad
    sim_rad = sign * (units - center) / units_per_rad

-- so a wrong `center` corrupts the COMMAND and the joint-angle OBS by the same amount, in
opposite senses, and the servo and the policy will agree they are on target while the joint
is visibly not straight. On 8/5 the six arm joints measured up to 136 units (40 deg) off
nominal, and three of them could not physically reach straight at all, because every
previous limit table had assumed straight = 512.

Legs and waist have never been measured this way. (The 6/24 session did measure all 17, but
that calibration was left uncommitted on the Pi and was lost -- only waist_roll: 485 from
6/18 survives in the map.) That is why this matters beyond tidiness: standing runs J-O drove
a residual policy whose baseline is "all zeros = straight" inside a +/-0.20 rad clamp box.
If a leg or waist centre is off by, say, 40 units = 0.205 rad, that box is displaced by more
than its own width and the standing negative result was measured on the wrong pose.

    python3 scripts/deploy/calibrate_centers.py --group lower          # legs + waist
    python3 scripts/deploy/calibrate_centers.py --group lower --repeat 3
    python3 scripts/deploy/calibrate_centers.py --group arms           # what 8/5 did
    python3 scripts/deploy/calibrate_centers.py --idx 8,9,10           # waist only

--repeat asks you to disturb and re-pose between rounds. Do it. The within-round spread only
measures bus/encoder noise (it was 0 on every arm joint on 8/5); the BETWEEN-round spread
measures how repeatably a human can pose the joint, which is the error that actually limits
this procedure and the only honest scale bar for judging whether an offset is real.
"""
import argparse
import datetime
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import yaml  # noqa: E402

from hardware import ServoBus  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO / "config" / "joint_servo_map.yaml"

GROUPS = {
    "legs":  list(range(0, 8)),      # both legs: roll, yaw, pitch, knee
    "waist": list(range(8, 11)),     # roll, pitch, yaw
    "lower": list(range(0, 11)),     # legs + waist -- the untried half as of 8/5
    "arms":  list(range(11, 17)),    # measured 8/5
    "all":   list(range(0, 17)),
}

REG_ANGLE_LIMIT = 0x09               # 4 bytes, big-endian: min_hi min_lo max_hi max_lo
DEG = 57.29577951308232

# How to pose each group. Printed before torque comes off; advisory, not enforced.
POSE_HINT = {
    "legs": (
        "Lay the robot on its BACK on a flat table -- do NOT try this standing, the legs go\n"
        "  limp the moment torque drops. Then, gently:\n"
        "    - extend both knees fully, until they stop. Do not force past the stop; the\n"
        "      knees are one-sided hinges, so that stop IS the straight reference.\n"
        "    - lay both legs flat and parallel: thighs in line with the torso, kneecaps\n"
        "      facing straight up (this is what kills hip roll and hip yaw error), feet\n"
        "      flat on the table.\n"
        "    - hold it there while this reads (a couple of seconds)."
    ),
    "waist": (
        "With the robot flat on its back, align the chest with the pelvis: no twist (yaw),\n"
        "  no side lean (roll), no forward bend (pitch). Press both the pelvis block and the\n"
        "  chest flat to the table -- the table plane sets roll and pitch, its edge sets yaw."
    ),
    "arms": (
        "Pose BOTH arms hanging straight down, relaxed and symmetric."
    ),
}
POSE_HINT["lower"] = POSE_HINT["legs"] + "\n\n  AND for the waist:\n  " + POSE_HINT["waist"]
POSE_HINT["all"] = POSE_HINT["lower"] + "\n\n  AND for the arms:\n  " + POSE_HINT["arms"]


def read_eeprom_limits(bus, sid):
    """EEPROM angle limits (reg 0x09) -> (lo, hi), or None if the servo did not answer."""
    d = bus.read_reg(sid, REG_ANGLE_LIMIT, 4)
    if d is None or len(d) != 4:
        return None
    return ((d[0] << 8) | d[1], (d[2] << 8) | d[3])


def sample(bus, sid, n):
    """n present-position reads -> (median, spread), or (None, None) if the bus stayed quiet."""
    vals = []
    for _ in range(n):
        p = bus.read_pos(sid)
        if p is not None:
            vals.append(int(p))
        time.sleep(0.01)
    if not vals:
        return None, None
    return int(statistics.median(vals)), max(vals) - min(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(DEFAULT_MAP))
    ap.add_argument("--group", default="lower", choices=sorted(GROUPS),
                    help="which joints to measure (default: lower = legs + waist)")
    ap.add_argument("--idx", default=None,
                    help="explicit comma-separated action indices; overrides --group")
    ap.add_argument("--samples", type=int, default=15, help="reads per joint per round (median)")
    ap.add_argument("--repeat", type=int, default=2,
                    help="independent posing rounds; you disturb and re-pose between them")
    ap.add_argument("--out", default=None, help="where to save the measurement (YAML)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.map))
    upr = float(cfg.get("units_per_rad", 195))
    default_center = int(cfg.get("servo_center", 512))

    if args.idx:
        want = [int(x) for x in args.idx.split(",") if x.strip() != ""]
        label, hint = "idx" + args.idx.replace(",", "-"), POSE_HINT["all"]
    else:
        want, label, hint = GROUPS[args.group], args.group, POSE_HINT[args.group]
    joints = [j for j in sorted(cfg["joints"], key=lambda x: x["idx"]) if j["idx"] in want]
    if not joints:
        print(f"no joints matched {label}")
        return 2
    ids = [int(j["servo_id"]) for j in joints]

    print(f"About to switch torque OFF on servos {ids} ({label}).")
    print("Those joints go LIMP IMMEDIATELY. Support or lay the robot down FIRST.\n")
    print("How to pose it:\n  " + hint + "\n")
    if input("Type 'y' once the robot is supported and you are ready: ").strip().lower() != "y":
        print("aborted -- nothing was changed")
        return 1

    bus = ServoBus().connect()
    try:
        bus.set_torque(ids, False)
        print(f"\nTorque OFF on {ids}. Nothing will be driven or written.\n")

        # EEPROM limits first: they are static, and they decide whether the measured straight
        # pose is even reachable. This is the check that would have caught the three arm
        # joints that could not straighten (8/5) before a single command was ever sent.
        print("reading EEPROM angle limits (reg 0x09)...")
        eeprom = {}
        for j in joints:
            sid = int(j["servo_id"])
            eeprom[sid] = read_eeprom_limits(bus, sid)
            got = eeprom[sid]
            print(f"  {j['dof']:<18} servo {sid:2d}: "
                  f"{f'{got[0]:4d} .. {got[1]:4d}' if got else '  NO REPLY'}")

        rounds = []
        for r in range(args.repeat):
            if r == 0:
                input(f"\n[Enter] when the pose is set (round 1/{args.repeat}): ")
            else:
                print(f"\nNow DISTURB the pose and set it up again from scratch "
                      f"(round {r + 1}/{args.repeat}).")
                print("This is what measures whether the procedure is repeatable at all.")
                input(f"  [Enter] when re-posed (round {r + 1}/{args.repeat}): ")
            print(f"  reading {args.samples} samples per joint...")
            this = {}
            for j in joints:
                sid = int(j["servo_id"])
                med, spread = sample(bus, sid, args.samples)
                this[sid] = med
                if med is None:
                    print(f"    !! {j['dof']:<18} servo {sid:2d}: NO valid reads -- check the bus")
                else:
                    flag = "   <<NOISY" if (spread or 0) > 4 else ""
                    print(f"    {j['dof']:<18} servo {sid:2d}: median {med:4d}  "
                          f"spread {spread}{flag}")
            rounds.append(this)
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

    # ---------------- analysis ----------------
    rows, bad = [], []
    for j in joints:
        sid = int(j["servo_id"])
        vals = [rd[sid] for rd in rounds if rd.get(sid) is not None]
        if not vals:
            bad.append(j)
            continue
        med = int(statistics.median(vals))
        cur = int(j.get("center", default_center))
        off = med - cur
        lo, hi = int(j["servo_limit"][0]), int(j["servo_limit"][1])
        ee = eeprom.get(sid)
        rows.append({
            "j": j, "sid": sid, "med": med, "vals": vals,
            "pose_spread": max(vals) - min(vals),
            "cur_center": cur, "off_units": off,
            "off_deg": off / upr * DEG,
            "off_rad": int(j["sign"]) * off / upr,          # signed, in SIM radians
            "map_lim": (lo, hi), "map_clip": not (lo <= med <= hi),
            "eeprom": ee, "ee_clip": ee is not None and not (ee[0] <= med <= ee[1]),
            "new_lim": (lo + off, hi + off),
        })

    print("\n" + "=" * 100)
    print("MEASURED STRAIGHT POSE")
    print("=" * 100)
    print(f"{'idx':>3} {'dof':<18} {'srv':>3} {'straight':>8} {'in use':>7} "
          f"{'offset':>7} {'deg':>7} {'sim-rad':>8} {'pose+-':>6}  flags")
    for r in rows:
        flags = []
        if r["map_clip"]:
            flags.append("MAP-CLIP")
        if r["ee_clip"]:
            flags.append("EEPROM-CLIP")
        if r["eeprom"] is None:
            flags.append("no-eeprom-read")
        if r["pose_spread"] > 8:
            flags.append("POSE-UNSTABLE")
        print(f"{r['j']['idx']:>3} {r['j']['dof']:<18} {r['sid']:>3} {r['med']:>8} "
              f"{r['cur_center']:>7} {r['off_units']:>+7d} {r['off_deg']:>+7.1f} "
              f"{r['off_rad']:>+8.3f} {r['pose_spread']:>6}  {' '.join(flags)}")
    for j in bad:
        print(f"{j['idx']:>3} {j['dof']:<18}  NO VALID READS -- check the bus, re-run")

    if any(r["map_clip"] or r["ee_clip"] for r in rows):
        print("\n!! CLIP: a joint's measured straight pose lies OUTSIDE its own limits, so it")
        print("   physically cannot be commanded straight -- the command is silently clipped")
        print("   (zero force, zero motion, no error). That is the 8/5 arm fault. Fix the")
        print("   limits (EEPROM first, then the map) BEFORE applying any new centre.")

    # Repeatability -- the scale bar for everything above.
    if args.repeat > 1 and rows:
        worst = max(rows, key=lambda r: r["pose_spread"])
        med_spread = statistics.median([r["pose_spread"] for r in rows])
        print(f"\nPose repeatability over {args.repeat} rounds: median {med_spread:.0f} units "
              f"({med_spread / upr * DEG:.1f} deg), worst {worst['pose_spread']} units "
              f"({worst['pose_spread'] / upr * DEG:.1f} deg, {worst['j']['dof']}).")
        print("Read every offset above against THAT number, not against zero.")

    # Left/right pairing: an independent check on the pose, free of any axis convention.
    pairs = {}
    for r in rows:
        d = r["j"]["dof"]
        if d[:2] in ("R_", "L_"):
            pairs.setdefault(d[2:], {})[d[0]] = r
    both = {k: v for k, v in pairs.items() if len(v) == 2}
    if both:
        print("\nLeft/right check (magnitudes only -- this MJCF does not use one mirror "
              "convention across all axes):")
        for name, v in both.items():
            a, b = abs(v["R"]["off_units"]), abs(v["L"]["off_units"])
            note = "   <<asymmetric: suspect the pose, or a real mount difference" \
                if abs(a - b) > max(10, 0.5 * max(a, b)) else ""
            print(f"  {name:<12} R {v['R']['off_units']:>+5d}   L {v['L']['off_units']:>+5d}"
                  f"   |R|-|L| = {a - b:>+4d}{note}")

    # ---------------- the verdict this run exists to deliver ----------------
    print("\n" + "=" * 100)
    print("VERDICT -- were the standing runs measured on correct zeros?")
    print("=" * 100)
    if rows:
        w = max(rows, key=lambda r: abs(r["off_rad"]))
        print(f"Worst joint: {w['j']['dof']}  {w['off_units']:+d} units = "
              f"{w['off_deg']:+.1f} deg = {w['off_rad']:+.3f} sim-rad")
        print("A wrong centre displaces the residual policy's whole 'straight' baseline AND its")
        print("+/-0.20 rad clamp box by that much, and biases the joint-angle obs by the same")
        print("amount. For scale: 0.20 rad = 39 units = 11.5 deg is the full clamp half-width,")
        print("and one encoder count = 0.0051 rad.")
        m = abs(w["off_rad"])
        if m < 0.02:
            print("\n=> CLEAN (< 0.02 rad everywhere). The zeros were right. The standing")
            print("   negative result is NOT confounded by leg/waist calibration -- which")
            print("   strengthens it, and leaves the 8/5 loop-delay / phase-margin diagnosis")
            print("   holding the blame.")
        elif m < 0.10:
            print("\n=> MINOR. Real, but small next to the 0.20 rad clamp. Worth folding in as")
            print("   measured constants; unlikely on its own to explain a 2-second fall.")
        else:
            print("\n=> CONFOUNDED. This is a large fraction of -- or larger than -- the clamp")
            print("   box the residual policy had to work inside. Runs J-O were commanded to a")
            print("   pose that is not the pose they thought, and the saturation those runs")
            print("   measured has to be re-read in that light. Fix the zeros first, then")
            print("   decide whether standing deserves a re-measurement. Not a residual4.")

    # ---------------- paste-ready output ----------------
    print("\n" + "-" * 100)
    print("PASTE INTO config/joint_servo_map.yaml -- centre plus the SAME shift on servo_limit,")
    print("so the physical travel arc is preserved and merely re-labelled:")
    print("-" * 100)
    for r in rows:
        if abs(r["off_units"]) <= 2:
            print(f"  idx {r['j']['idx']:2d} {r['j']['dof']:<18} center: {r['med']}"
                  f"   # {r['off_units']:+d} units -- within noise, no change needed")
            continue
        print(f"  idx {r['j']['idx']:2d} {r['j']['dof']:<18} "
              f"servo_limit: [{r['new_lim'][0]}, {r['new_lim'][1]}], center: {r['med']}"
              f"   # was [{r['map_lim'][0]}, {r['map_lim'][1]}] @ {r['cur_center']}; "
              f"{r['off_units']:+d} units = {r['off_deg']:+.1f} deg")

    need_ee = [r for r in rows if abs(r["off_units"]) > 2 and r["eeprom"] is not None
               and not (r["eeprom"][0] <= r["new_lim"][0] and r["new_lim"][1] <= r["eeprom"][1])]
    print("\n" + "-" * 100)
    if need_ee:
        print("EEPROM angle limits must be shifted FIRST (before the map centres), or the joint")
        print("clips silently -- copy fix_arm_limits.py, which is the verified writer:")
        print("-" * 100)
        for r in need_ee:
            print(f"  {r['sid']:2d}: ({r['new_lim'][0]}, {r['new_lim'][1]}),"
                  f"   # {r['j']['dof']:<18} straight {r['med']}, "
                  f"currently {r['eeprom'][0]}..{r['eeprom'][1]}")
    else:
        print("EEPROM angle limits: no write needed -- every shifted range still fits inside the")
        print("limits already in the servos. Only the map changes.")

    # ---------------- save (this project has lost hand-measured constants twice) ----------
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # NOT data/ -- that directory is gitignored, so saving there would silently produce an
    # untracked file and reproduce the exact loss this tool exists to prevent. config/ is
    # tracked, and a measured constant is configuration, not output.
    out = (Path(args.out) if args.out
           else REPO / "config" / "calibration" / f"center_calibration_{label}_{stamp}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "measured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "group": label, "samples": args.samples, "repeat": args.repeat,
        "units_per_rad": upr, "servo_center": default_center,
        "joints": [{
            "idx": r["j"]["idx"], "dof": r["j"]["dof"], "servo_id": r["sid"],
            "straight_units": r["med"], "rounds": r["vals"], "pose_spread": r["pose_spread"],
            "center_in_use": r["cur_center"], "offset_units": r["off_units"],
            "offset_deg": round(r["off_deg"], 2), "offset_sim_rad": round(r["off_rad"], 4),
            "map_limit_now": list(r["map_lim"]), "map_limit_shifted": list(r["new_lim"]),
            "eeprom_limit": list(r["eeprom"]) if r["eeprom"] else None,
            "clips_map": r["map_clip"], "clips_eeprom": r["ee_clip"],
        } for r in rows],
    }
    with open(out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print(f"\nSaved: {out}")
    print("scp this back to the dev box and COMMIT it. The 6/24 whole-body re-zero and the")
    print("8/5 arm re-zero were both left Pi-only; the 6/24 one was lost outright.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
