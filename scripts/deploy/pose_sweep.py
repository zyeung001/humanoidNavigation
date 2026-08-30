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
    trims = np.arange(lo, hi + 1e-9, abs(args.step))
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

    print("\nHOLD THE ROBOT, or use a slack tether. The ends of this sweep are deliberately")
    print("poses it may not stand in. Ctrl-C ramps back and drops torque.\n")

    from frame_log import FsrSampler  # noqa: PLC0415
    from hardware import ServoBus  # noqa: PLC0415

    bus = ServoBus().connect()
    fsr = FsrSampler(channels=[int(c) for c in args.fsr_channels.split(",")])
    start = None
    rows = []
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
            if samples:
                med, sd = float(np.median(samples)), float(np.std(samples))
                rows.append((tr, med, sd))
                print(f"  trim {tr:+5.1f} deg -> CoP {med:+7.1f} mm   "
                      f"spread {sd:4.1f}   load {1000*np.median(loads):5.2f} mS   n={len(samples)}")
            else:
                rows.append((tr, float("nan"), float("nan")))
                print(f"  trim {tr:+5.1f} deg -> a sensor is UNLOADED: the foot lifted, or the "
                      f"robot is not standing on the instrumented one")
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

    report(rows, names, args)
    return 0


def report(rows, names, args):
    good = [(t, c) for t, c, _ in rows if np.isfinite(c)]
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
    print("\n  Trust the slope. The absolute zero carries your FSR calibration error"
          + ("." if args.cop_offset else
             " --\n  run the flat-bar test and pass --cop-offset to remove it."))


if __name__ == "__main__":
    sys.exit(main())
