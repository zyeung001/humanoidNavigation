#!/usr/bin/env python3
"""Read a frame log and answer the questions you actually have about a run.

Three of them:

  WHY IS SO MUCH OF THIS NaN?  Every empty column in these logs means "this channel was
  not enabled", not "this reading failed", and the difference matters. The schema is
  shared across every script that writes a frame log, so a single-joint excitation sweep
  still carries the columns a standing run would fill. This prints which channels are
  empty AND the flag that would have filled them, so an empty column is a decision you
  can revisit rather than a mystery.

  HOW WELL DID IT TRACK?  Mean, p95 and worst |commanded - actual| per joint, in degrees.
  This is the quantity every open question on this robot reduces to, and until now it
  existed only as two raw columns that had to be subtracted by hand.

  HOW FAST DID IT GET THERE?  Two different questions depending on the log, chosen
  automatically. When the command moves in discrete steps (excitation sweeps) the answer
  is a rise time: how long from the command changing until the joint arrives within
  tolerance. When the command moves continuously (a standing run) there are no steps to
  time, so the honest analogue is the LAG that best aligns the two signals -- the delay
  at which commanded and actual correlate most strongly.

  python scripts/debug/log_summary.py --log logs/20260814_160219_excite_waist_pitch_step.csv
  python scripts/debug/log_summary.py --log logs/demo/sim_standing.csv --tol 2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Why a channel is empty, and what would have filled it.
WHY_EMPTY = {
    "pg": ("pelvis attitude", "deploy_standing fills this; excite_joint needs --imu"),
    "av": ("pelvis angular rate", "deploy_standing fills this; excite_joint needs --imu"),
    "upright_cos": ("tilt against vertical",
                    "only meaningful when balancing; excite_joint never computes it"),
    "fsr": ("foot force, raw divider volts", "pass --log-fsr (deploy_standing only)"),
    "jvel_f": ("filtered joint velocity",
               "first row only: nothing to finite-difference against yet"),
}


def why(col):
    for key, (what, fix) in WHY_EMPTY.items():
        if col.startswith(key):
            return what, fix
    return "unknown", "not a channel this tool knows about"


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} is empty")
    meta_p = Path(path).with_suffix(".meta.json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return rows, meta


def col(rows, name):
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        v = r.get(name, "")
        if v not in ("", None):
            try:
                out[i] = float(v)
            except ValueError:
                pass
    return out


def rise_times(t, cmd, act, tol_units):
    """For each discrete command step: seconds until the joint is within tol of the target.

    Reported as the time from the command CHANGING, so it includes the servo's dead time,
    its lag, and anything else in the way. That is the number you care about when asking
    how fast the robot gets where it was told; it is not the same as the actuator's tau,
    which fit_actuator.py separates out properly.
    """
    edges = np.flatnonzero(np.abs(np.diff(cmd)) > tol_units) + 1
    out = []
    for i, k in enumerate(edges):
        stop = edges[i + 1] if i + 1 < len(edges) else len(t)
        target = cmd[k]
        hit = np.flatnonzero(np.abs(act[k:stop] - target) <= tol_units)
        if len(hit):
            out.append((t[k + hit[0]] - t[k], abs(target - act[k - 1]), True))
        else:
            out.append((t[stop - 1] - t[k], abs(target - act[k - 1]), False))
    return out


def best_lag(cmd, act, dt, max_ms=400):
    """Delay at which a continuously-moving command best aligns with the response."""
    c = cmd - cmd.mean()
    a = act - act.mean()
    if np.std(c) < 1e-9 or np.std(a) < 1e-9:
        return None, 0.0
    best, bl = -2.0, 0
    for lag in range(0, int(max_ms / 1000.0 / dt) + 1):
        if lag >= len(c):
            break
        x, y = c[:len(c) - lag], a[lag:]
        r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 8 else -2.0
        if r > best:
            best, bl = r, lag
    return bl * dt * 1000.0, best


def main():
    p = argparse.ArgumentParser(description="Summarize a frame log")
    p.add_argument("--log", required=True)
    p.add_argument("--tol", type=float, default=3.0,
                   help="arrival tolerance in ENCODER UNITS (1 unit = 0.29 deg). Default 3 "
                        "sits just above the 1-unit quantum; the arms' own deadband is 16, "
                        "so a tolerance below that can never be met on an arm joint.")
    p.add_argument("--top", type=int, default=8, help="joints to list")
    args = p.parse_args()

    rows, meta = load(args.log)
    cols = list(rows[0].keys())
    dofs = meta.get("dofs") or sorted({c.split(".", 1)[1] for c in cols if c.startswith("u_meas.")})
    t = col(rows, "t_rel")
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.025
    upr = 195.0 / (180.0 / np.pi)     # encoder units per degree
    deg = lambda u: u / (195.0 * np.pi / 180.0)   # noqa: E731

    print(f"{args.log}")
    print(f"  {len(rows)} rows over {t[-1]-t[0]:.2f}s = {len(rows)/max(t[-1]-t[0],1e-9):.1f} Hz"
          f"   {len(dofs)} joint(s), {len(cols)} columns")
    if meta.get("script"):
        print(f"  written by {meta['script']}"
              + (f"   [{meta['source']}]" if meta.get("source") else ""))

    # ---- 1. what is empty, and why ----
    print("\n--- EMPTY CHANNELS (not failures: channels that were never switched on) ---")
    empty = []
    for c in cols:
        v = col(rows, c)
        n = int(np.sum(~np.isfinite(v)))
        if n:
            empty.append((c, n))
    if not empty:
        print("  none -- every column has data")
    for c, n in empty:
        what, fix = why(c.split(".", 1)[0])
        pct = 100 * n / len(rows)
        print(f"  {c:22s} {pct:5.1f}% empty   {what}")
        if pct > 50:
            print(f"  {'':22s}               -> {fix}")

    # ---- 2. tracking error ----
    print("\n--- TRACKING ERROR  (commanded minus actual) ---")
    errs = {}
    for j in dofs:
        c_, a_ = col(rows, f"u_cmd.{j}"), col(rows, f"u_meas.{j}")
        m = np.isfinite(c_) & np.isfinite(a_)
        if m.sum() < 5:
            continue
        e = np.abs(c_[m] - a_[m])
        errs[j] = (deg(e.mean()), deg(np.percentile(e, 95)), deg(e.max()), np.ptp(c_[m]))
    if errs:
        allmean = np.mean([v[0] for v in errs.values()])
        print(f"  AVERAGE over {len(errs)} joints: {allmean:.2f} deg"
              f"   ({allmean*upr:.1f} encoder units)")
        print(f"\n  {'joint':18s} {'mean':>7} {'p95':>7} {'worst':>7}   {'cmd travel':>10}")
        for j in sorted(errs, key=lambda k: -errs[k][0])[:args.top]:
            mn, p95, mx, trav = errs[j]
            note = "  (never commanded to move)" if trav < 2 else ""
            print(f"  {j:18s} {mn:6.2f}d {p95:6.2f}d {mx:6.2f}d   {deg(trav):8.1f}d{note}")

    # ---- 3. how fast it gets there ----
    print("\n--- SPEED TO REACH THE COMMANDED POINT ---")
    for j in (list(errs)[:1] if len(dofs) == 1 else
              sorted(errs, key=lambda k: -errs[k][3])[:3]):
        c_, a_ = col(rows, f"u_cmd.{j}"), col(rows, f"u_meas.{j}")
        m = np.isfinite(c_) & np.isfinite(a_)
        c_, a_, tt = c_[m], a_[m], t[m]
        if np.ptp(c_) < 2:
            print(f"  {j}: command never moves -- nothing to time")
            continue
        # Is this a staircase or a continuously moving command? Decide by how often the
        # command HOLDS STILL, not by counting changes. A standing policy issues a new
        # target every frame, so counting "moves" there finds hundreds of them and then
        # reports that none arrived -- which says nothing, because the target had already
        # moved on before the joint could reach it.
        hold_frac = float(np.mean(np.abs(np.diff(c_)) < 0.5))
        if hold_frac > 0.5:
            steps = [s for s in rise_times(tt, c_, a_, args.tol) if s[1] > 4 * args.tol]
            arrived = [s[0] for s in steps if s[2]]
            miss = len(steps) - len(arrived)
            print(f"  {j}: step-like command, {len(steps)} moves "
                  f"(command holds still {100*hold_frac:.0f}% of frames)")
            if arrived:
                print(f"     median time to arrive within {args.tol:.0f} units "
                      f"({deg(args.tol):.1f} deg): {1000*np.median(arrived):.0f} ms"
                      f"   [{1000*min(arrived):.0f}-{1000*max(arrived):.0f}]")
            if miss:
                print(f"     {miss} of {len(steps)} never arrived -- the joint stopped short "
                      f"of what it was told by more than {deg(args.tol):.1f} deg")
        else:
            lag, r = best_lag(c_, a_, dt)
            print(f"  {j}: command moves continuously ({100*(1-hold_frac):.0f}% of frames), "
                  f"so there is no step to time.")
            if lag is not None:
                print(f"     the response trails the command by {lag:.0f} ms "
                      f"(best correlation {r:.3f})")
                print(f"     mean gap while tracking: {errs[j][0]:.2f} deg")

    # ---- 4. loop health ----
    lp = col(rows, "loop_ms")
    lp = lp[np.isfinite(lp)]
    if len(lp):
        rej = col(rows, "rej_total")
        rej = rej[np.isfinite(rej)]
        print(f"\n--- LOOP ---\n  period median {np.median(lp):.1f} ms, p95 {np.percentile(lp,95):.1f} ms,"
              f" worst {lp.max():.1f} ms")
        if len(rej):
            print(f"  rejected servo reads: {int(rej[-1])} cumulative")
        if meta.get("rows_dropped"):
            print(f"  !! {meta['rows_dropped']} rows DROPPED by the writer -- log is incomplete")


if __name__ == "__main__":
    main()
