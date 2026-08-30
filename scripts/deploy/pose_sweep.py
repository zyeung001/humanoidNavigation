#!/usr/bin/env python3
# pose_sweep.py  --  RUN ON THE PI
"""Find the pose this robot actually balances in, by measuring instead of asking sim.

WHY THIS EXISTS. Every standing policy so far was trained around a baseline pose that was
GUESSED three times: v1 used sim's settled stance, v2 zeroed it to straight, v3 widened
the box around straight. All three failed on hardware. The policy has since re-derived
v1's stance independently across ~30M steps -- it wants waist_yaw -0.52 and R_hip_pitch
+0.32 and is clipped short of both -- while the one time that stance WAS allowed, the real
robot hung crooked at 13-16 degrees. So sim is confident about a pose hardware rejects,
and nobody has ever measured which pose hardware would accept.

The feet can answer it. Centre of pressure is where the weight actually lands, and it does
not care what any model believes. This walks a trim through a range, holds each pose, and
reads where the pressure sits -- turning "where should it stand" from an argument into a
line with a zero crossing.

WHAT IT DOES NOT DO. It does not balance the robot. Hold it, or use a slack tether: the
poses at the ends of the sweep are deliberately ones it may not stand in, which is the
point. No policy runs here, so nothing can seize.

READING THE RESULT. The SLOPE, in mm of CoP per degree of trim, is trustworthy. The
absolute zero is only as good as the FSR calibration: if the foot plate shares load with
the sensors, every reading is biased toward centre by the same amount. Run the flat-bar
offset test and pass --cop-offset, or treat the crossing as an estimate whose sign and
slope are solid and whose exact value is not.

  python3 scripts/deploy/pose_sweep.py --joints waist_pitch --range -6 6 --step 2
  python3 scripts/deploy/pose_sweep.py --joints R_hip_pitch,L_hip_pitch --range -6 6 --step 3
  python3 scripts/deploy/pose_sweep.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsr_monitor import fsr_resistance  # noqa: E402
from sim_real_map import SimRealMap, DEFAULT_MAP  # noqa: E402

MAX_TRIM_DEG = 12.0        # beyond this it is a contortion, not a trim
SIM_COP_MM = 4.3           # scripts/debug/sim_cop.py, straight pose
COM_HEIGHT_MM = 414.7      # for converting a CoP shift into an equivalent body lean


def cop_mm(volts, rfixed, vcc, x_heel, x_toe):
    """Load-weighted mean of the two sensor positions. None when a sensor is unloaded."""
    g = []
    for v in volts:
        r = fsr_resistance(v, rfixed, vcc, False)
        g.append(0.0 if r == float("inf") else 1.0 / max(r, 1e-6))
    tot = g[0] + g[1]
    if tot <= 1e-9 or min(g) / tot < 0.05:
        return None, tot
    return (g[0] * x_heel + g[1] * x_toe) / tot, tot


def build_plan(m, dofs, names, trims):
    base = m.centers.astype(float)      # straight: the per-joint hand-measured centres
    idxs = [dofs.index(n) for n in names]
    out = []
    for tr in trims:
        u = base.copy()
        for i in idxs:
            u[i] += m.signs[i] * np.deg2rad(tr) * m.units_per_rad
        out.append((float(tr), np.clip(u, m.lim_lo, m.lim_hi).round().astype(int)))
    return out, idxs


def main():
    p = argparse.ArgumentParser(description="Measure the pose the robot balances in")
    p.add_argument("--joints", default="waist_pitch",
                   help="comma-separated joints trimmed together (trim both hips as a pair, "
                        "or the robot twists instead of leaning)")
    p.add_argument("--range", type=float, nargs=2, default=[-6.0, 6.0], metavar=("LO", "HI"))
    p.add_argument("--step", type=float, default=2.0, help="degrees per point")
    p.add_argument("--probe", type=float, default=None, metavar="TRIM",
                   help="hold ONE trim and stream lean + CoP live until Ctrl-C, instead of "
                        "sweeping. This is the mode to use on a robot that will not stand "
                        "unaided: steady it, release, and watch which way it goes. Falls "
                        "backward -> probe a more forward trim; falls forward -> go back. "
                        "Five or six of these bisect the balance point to about a degree, "
                        "and none of them needs the robot to stay up.")
    p.add_argument("--dwell", type=float, default=4.0, help="seconds to settle and sample")
    p.add_argument("--ramp", type=float, default=2.0, help="seconds to move between points")
    p.add_argument("--speed", type=int, default=300)
    p.add_argument("--rfixed", type=float, default=2000.0)
    p.add_argument("--vcc", type=float, default=3.3)
    p.add_argument("--pos-heel", type=float, default=-40.0, help="mm from the foot centre")
    p.add_argument("--pos-toe", type=float, default=36.0)
    p.add_argument("--cop-offset", type=float, default=0.0,
                   help="mm to subtract, from the flat-bar zero test")
    p.add_argument("--fsr-channels", default="0,1")
    p.add_argument("--tilt-cut", type=float, default=0.80,
                   help="abort and ramp back if the pelvis tilts past this (cosine of upright; "
                        "0.80 is about 37 deg). This tool holds a pose like a statue and does "
                        "NOT balance, so without a cut it would keep driving servos while the "
                        "robot topples.")
    p.add_argument("--upright-deg", type=float, default=5.0,
                   help="a probe sample counts as UPRIGHT while |lean| stays under this. Only "
                        "those samples are averaged, because a reading taken while the robot "
                        "is toppling measures the topple and not the pose.")
    p.add_argument("--imu-calib", default=str(ROOT / "config" / "imu_calib.yaml"))
    p.add_argument("--no-imu", action="store_true",
                   help="skip the IMU entirely: no tilt cut and no lean cross-check")
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    m = SimRealMap(args.map)
    dofs = [j.dof for j in m.joints]
    names = [s.strip() for s in args.joints.split(",") if s.strip()]
    for n in names:
        if n not in dofs:
            raise SystemExit(f"unknown joint '{n}'. choices: {dofs}")
    lo, hi = sorted(args.range)
    if max(abs(lo), abs(hi)) > MAX_TRIM_DEG:
        raise SystemExit(f"trim beyond +-{MAX_TRIM_DEG} deg is a contortion, not a trim")
    if args.probe is not None:
        trims = np.array([float(args.probe)])
    else:
        trims = np.arange(lo, hi + 1e-9, abs(args.step))
        # Visit from the CENTRE outward, alternating sides, rather than marching from one
        # end to the other. A robot that cannot stand unaided topples at the extremes, and
        # marching from an end spends the first points there -- losing the middle of the
        # curve, which is the part that carries the crossing.
        mid = 0.5 * (lo + hi)
        trims = trims[np.argsort(np.abs(trims - mid), kind="stable")]
    plans, idxs = build_plan(m, dofs, names, trims)

    print(f"sweeping {', '.join(names)} over {lo:+.0f}..{hi:+.0f} deg in {args.step:g} deg steps")
    print(f"  {len(plans)} poses x ({args.ramp:g}s ramp + {args.dwell:g}s dwell) = "
          f"{len(plans)*(args.ramp+args.dwell)/60:.1f} min")
    print(f"  CoP from sensors at {args.pos_heel:+.0f}/{args.pos_toe:+.0f} mm, "
          f"R_fixed {args.rfixed:.0f} ohm, offset {args.cop_offset:+.1f} mm")
    if args.dry_run:
        for tr, u in plans:
            print(f"  trim {tr:+5.1f} deg -> " + "  ".join(f"{dofs[i]}={u[i]}" for i in idxs))
        print("\n[dry-run] no hardware touched.")
        return 0

    if args.probe is not None:
        print(f"\nPROBE at trim {args.probe:+.1f} deg. Steady it, then RELEASE and watch.")
        print("  falls BACKWARD -> the balance point is at a more forward trim")
        print("  falls FORWARD  -> go back the other way")
        print("Ctrl-C ramps back and drops torque.\n")
    else:
        print("\nHOLD THE ROBOT, or use a slack tether. The ends of this sweep are deliberately")
        print("poses it may not stand in. Ctrl-C ramps back and drops torque.\n")

    from frame_log import FsrSampler  # noqa: PLC0415
    from hardware import ServoBus  # noqa: PLC0415

    bus = ServoBus().connect()
    fsr = FsrSampler(channels=[int(c) for c in args.fsr_channels.split(",")])
    # The IMU earns its place twice here: it aborts before a topple, and it measures the
    # LEAN at each trim. Lean and centre of pressure are independent instruments aimed at
    # the same question, so if they disagree about which way the pose should move, that
    # disagreement is worth more than either number on its own.
    imu = None
    if not args.no_imu:
        import os

        import yaml  # noqa: PLC0415

        from hardware import IMU  # noqa: PLC0415
        remap = None
        if os.path.exists(args.imu_calib):
            remap = np.asarray(yaml.safe_load(open(args.imu_calib))["axis_remap"],
                               dtype=np.float32)
        imu = IMU(axis_remap=remap).connect()
    start = None
    rows = []
    probe_hist = []
    try:
        bus.set_torque(m.servo_ids, True)
        start = bus.read_all(m.servo_ids)
        start = np.where(np.isfinite(start), start, m.centers.astype(float))
        cur = start.astype(float).copy()
        for tr, target in plans:
            steps = max(1, int(args.ramp * 40))
            for k in range(1, steps + 1):
                bus.write_all(m.servo_ids,
                              (cur + (target - cur) * k / steps).round().astype(int),
                              speed=args.speed)
                time.sleep(1 / 40)
            cur = target.astype(float)
            if args.probe is not None:
                # Hold and stream until Ctrl-C. No tilt cut here on purpose: the whole point
                # is to watch it go over and see WHICH WAY, so cutting torque mid-topple
                # would throw away the measurement being taken.
                print("  holding. release it now.\n")
                upright_since = None
                while True:
                    v, _ = fsr.read()
                    c, tot = cop_mm(v, args.rfixed, args.vcc, args.pos_heel, args.pos_toe)
                    ln = float("nan")
                    if imu is not None:
                        pg = imu.projected_gravity()
                        ln = float(np.degrees(np.arctan2(pg[0], -pg[2])))
                    now = time.time()
                    probe_hist.append(
                        (now, ln, float("nan") if c is None else c - args.cop_offset))
                    up = bool(np.isfinite(ln) and abs(ln) <= args.upright_deg)
                    upright_since = (upright_since or now) if up else None
                    held = 0.0 if upright_since is None else now - upright_since
                    where = ("FORWARD" if ln > args.upright_deg else
                             "BACKWARD" if ln < -args.upright_deg else "UPRIGHT")
                    ctxt = "  --  " if c is None else f"{c - args.cop_offset:+6.1f}"
                    print(f"\r  lean {ln:+6.1f} deg {where:8s} CoP {ctxt} mm   "
                          f"load {1000*tot:5.2f} mS   upright {held:4.1f}s   ",
                          end="", flush=True)
                    time.sleep(0.1)
            time.sleep(args.dwell * 0.4)          # settle before sampling, not while moving
            samples, loads = [], []
            t0 = time.time()
            while time.time() - t0 < args.dwell * 0.6:
                v, _age = fsr.read()
                c, tot = cop_mm(v, args.rfixed, args.vcc, args.pos_heel, args.pos_toe)
                if c is not None:
                    samples.append(c - args.cop_offset)
                    loads.append(tot)
                time.sleep(0.05)
            lean = float("nan")
            if imu is not None:
                pg = imu.projected_gravity()
                lean = float(np.degrees(np.arctan2(pg[0], -pg[2])))    # + = leaning forward
                if -float(pg[2]) < args.tilt_cut:
                    print(f"  trim {tr:+5.1f} deg -> TILT CUT at upright {-float(pg[2]):.2f}: "
                          f"it is going over. Aborting the sweep and ramping back.")
                    break
            if samples:
                med, sd = float(np.median(samples)), float(np.std(samples))
                rows.append((tr, med, sd, lean))
                print(f"  trim {tr:+5.1f} deg -> CoP {med:+7.1f} mm   spread {sd:4.1f}   "
                      f"load {1000*np.median(loads):5.2f} mS   lean {lean:+5.1f} deg   "
                      f"n={len(samples)}")
            else:
                rows.append((tr, float("nan"), float("nan"), lean))
                print(f"  trim {tr:+5.1f} deg -> a sensor is UNLOADED (lean {lean:+5.1f} deg): "
                      f"the foot lifted, or the robot is not on the instrumented one")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        print("ramping back to the starting pose and releasing torque.")
        try:
            if start is not None:
                cur = bus.read_all(m.servo_ids)
                cur = np.where(np.isfinite(cur), cur, start)
                for k in range(1, 81):
                    u = np.clip(cur + (start - cur) * k / 80, m.lim_lo, m.lim_hi)
                    bus.write_all(m.servo_ids, u.round().astype(int), speed=args.speed)
                    time.sleep(1 / 40)
        except Exception:
            pass
        fsr.close()
        bus.set_torque(m.servo_ids, False)
        bus.close()
        if imu is not None:
            imu.close()

    if args.probe is not None:
        probe_report(probe_hist, args)
    else:
        report(rows, names, args)
    return 0


def probe_report(hist, args):
    """Summarise only the stretch where the robot was actually upright.

    The first version printed a live line and left whatever happened to be on screen when
    you hit Ctrl-C, which was almost always mid-topple: two probes at the same trim came
    back 30 mm apart for exactly that reason, with lean already past +10 deg in both. A
    centre-of-pressure reading is a statement about a pose, and it is only a statement
    about THAT pose while the robot is still in it.
    """
    print()
    print(f"--- PROBE at trim {args.probe:+.1f} deg ---")
    if not hist:
        print("  no samples captured.")
        return
    t0 = hist[0][0]
    up = [(t - t0, ln, c) for t, ln, c in hist
          if np.isfinite(ln) and abs(ln) <= args.upright_deg]
    print(f"  {len(hist)} samples over {hist[-1][0] - t0:.1f}s; {len(up)} upright "
          f"(|lean| <= {args.upright_deg:.0f} deg)")
    if not up:
        closest = min(hist, key=lambda r: abs(r[1]) if np.isfinite(r[1]) else 1e9)
        print(f"  NEVER upright -- closest approach was {closest[1]:+.1f} deg.")
        print("  This trim did not hold, so it has no centre of pressure to report.")
        return
    # The longest UNBROKEN upright stretch, not every upright sample: two seconds of
    # holding says something about the pose, whereas scattered frames on the way past
    # vertical are just the trajectory of a fall.
    runs, cur = [], [up[0]]
    for prev, nxt in zip(up[:-1], up[1:]):
        if nxt[0] - prev[0] < 0.35:
            cur.append(nxt)
        else:
            runs.append(cur)
            cur = [nxt]
    runs.append(cur)
    best = max(runs, key=len)
    held = best[-1][0] - best[0][0]
    lean = np.array([r[1] for r in best])
    cop = np.array([r[2] for r in best if np.isfinite(r[2])])
    print(f"  longest unbroken upright stretch: {held:.1f}s")
    print(f"  lean over it  {np.median(lean):+6.2f} deg   (spread {np.std(lean):.2f})")
    if not len(cop):
        print("  a sensor was unloaded throughout -- no CoP available")
        return
    print(f"  CoP over it   {np.median(cop):+6.1f} mm    (spread {np.std(cop):.1f}, n={len(cop)})")
    print(f"  sim predicts  {SIM_COP_MM:+6.1f} mm at the straight pose")
    print()
    print(f"  log this as: trim {args.probe:+.1f} -> CoP {np.median(cop):+.1f} mm, "
          f"lean {np.median(lean):+.2f} deg, held {held:.1f}s")


def report(rows, names, args):
    good = [(t, c) for t, c, _, _ in rows if np.isfinite(c)]
    print("\n--- RESULT ---")
    if len(good) < 3:
        print("  not enough loaded points to fit a line. Was the robot standing on the")
        print("  instrumented foot for the whole sweep?")
        return
    x = np.array([g[0] for g in good])
    y = np.array([g[1] for g in good])
    slope, icept = np.polyfit(x, y, 1)
    label = "+".join(names)
    print(f"  CoP moves {slope:+.2f} mm per degree of {label} trim")
    print(f"  measured CoP at trim 0: {icept:+.1f} mm     sim predicts {SIM_COP_MM:+.1f} mm")
    gap = icept - SIM_COP_MM
    print(f"  sim-vs-real gap {gap:+.1f} mm = "
          f"{np.degrees(np.arctan(gap / COM_HEIGHT_MM)):+.2f} deg of body lean")
    if abs(slope) > 1e-6:
        zero = -icept / slope
        inside = min(x) <= zero <= max(x)
        print(f"\n  pressure centres at a trim of {zero:+.1f} deg"
              + ("" if inside else "  -- EXTRAPOLATED beyond the sweep; widen --range to confirm"))
        print(f"  that is the baseline to train around: {label} shifted {zero:+.1f} deg")
    leans = [(t, ln) for t, _, _, ln in rows if np.isfinite(ln)]
    if len(leans) >= 3 and abs(slope) > 1e-6:
        lx = np.array([v[0] for v in leans])
        ly = np.array([v[1] for v in leans])
        lslope, licept = np.polyfit(lx, ly, 1)
        print()
        print(f"  IMU cross-check: lean moves {lslope:+.2f} deg per degree of trim, "
              f"{licept:+.1f} deg at trim 0")
        if abs(lslope) > 1e-6:
            lzero = -licept / lslope
            agree = abs(lzero - (-icept / slope)) < 2.0
            print(f"  lean reaches vertical at trim {lzero:+.1f} deg"
                  + ("   AGREES with the pressure crossing" if agree else
                     "   DISAGREES with the pressure crossing -- trust neither until they do"))

    print("\n  Trust the slope. The absolute zero carries your FSR calibration error"
          + ("." if args.cop_offset else
             " --\n  run the flat-bar test and pass --cop-offset to remove it."))


if __name__ == "__main__":
    sys.exit(main())
