"""Analyze hardware turn-replay logs (hack vs fix) -> WTR numbers + paper figure.

Takes the two CSV logs written by scripts/deploy/replay_turn.py on the Pi and produces
the Contribution-2 hardware result:

  - HW WTR   = std(waist_yaw servo readback) over the turn phases. The waist_yaw joint
    IS the torso-pelvis twist DOF, so this is directly comparable to the sim WTR
    (std(torso_yaw - pelvis_yaw) from measure_waist_twist_real.py).
  - pelvis yaw = integrated pelvis gyro z (did the robot actually rotate?)
  - a 2x2 figure: rows = waist twist / pelvis yaw, columns = hack / fix, with the sim
    ground truth from the recorded .npz overlaid and the command phases shaded.

Expected: HACK sweeps waist_yaw with little pelvis rotation (twist exploit survives on
hardware); FIX shows no waist sweep.

Usage (defaults match the replay/record file names):
    python scripts/analyze_turn_replay.py --hack turn_traj_hack_log.csv \
        --fix turn_traj_fix_log.csv --out figures/hw_turn_replay.png
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).parent.parent

BLUE = "#2a78d6"      # hardware signal
GRAY_REF = "#898781"  # sim reference (dashed)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BAND_CCW = "#f0efec"  # phase shading (neutral, no hue)
BAND_CW = "#e6e5e0"


def load_log(path):
    """Read a replay_turn.py CSV into a dict of float arrays (NaN-safe)."""
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise SystemExit(f"{path}: empty log")

    def col(name):
        return np.array([float(r[name]) if r.get(name) not in (None, "", "nan") else np.nan
                         for r in rows], dtype=np.float64)

    d = {k: col(k) for k in ("t", "yaw_cmd", "waist_yaw_target_rad", "waist_yaw_rad",
                             "pelvis_gyro_z", "pelvis_yaw_int")}
    d["n"] = len(rows)
    return d


def arm_metrics(log, label):
    """Per-arm hardware numbers over the turn phases (yaw_cmd != 0)."""
    turn = np.abs(log["yaw_cmd"]) > 1e-6
    ccw = log["yaw_cmd"] > 1e-6
    waist = log["waist_yaw_rad"][turn]
    waist = waist[np.isfinite(waist)]
    yaw = log["pelvis_yaw_int"]
    yaw_ok = np.isfinite(yaw)

    m = {"label": label, "frames": log["n"]}
    m["hw_wtr"] = float(np.std(waist)) if waist.size else float("nan")
    m["waist_range"] = float(np.ptp(waist)) if waist.size else float("nan")
    if yaw_ok.any():
        m["pelvis_yaw_range"] = float(np.ptp(yaw[yaw_ok & turn])) if (yaw_ok & turn).any() else float("nan")
        # yaw tracking over the CCW phase: net rotation / commanded rotation
        idx = np.flatnonzero(ccw & yaw_ok)
        if idx.size > 1:
            net = yaw[idx[-1]] - yaw[idx[0]]
            commanded = 0.5 * (log["t"][idx[-1]] - log["t"][idx[0]])
            m["ccw_net_yaw"] = float(net)
            m["ccw_tracking"] = float(net / commanded) if commanded > 0 else float("nan")
        else:
            m["ccw_net_yaw"] = m["ccw_tracking"] = float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            m["twist_to_turn"] = m["hw_wtr"] / max(float(np.std(yaw[yaw_ok & turn])), 1e-6)
    else:
        m["pelvis_yaw_range"] = m["ccw_net_yaw"] = m["ccw_tracking"] = float("nan")
        m["twist_to_turn"] = float("nan")
    return m


def shade_phases(ax, t, yaw_cmd):
    """Gray bands over the turn phases; labels go on the top row only (caller decides)."""
    for band, mask in ((BAND_CCW, yaw_cmd > 1e-6), (BAND_CW, yaw_cmd < -1e-6)):
        if mask.any():
            edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
            for s, e in zip(edges[::2], edges[1::2]):
                ax.axvspan(t[s], t[min(e, len(t) - 1)], color=band, zorder=0, lw=0)


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hack", default="turn_traj_hack_log.csv", help="replay CSV, hack arm")
    ap.add_argument("--fix", default="turn_traj_fix_log.csv", help="replay CSV, fix arm")
    ap.add_argument("--traj-hack", default="data/turn_traj_hack.npz",
                    help="recorded sim trajectory for overlay (optional)")
    ap.add_argument("--traj-fix", default="data/turn_traj_fix.npz")
    ap.add_argument("--out", default="figures/hw_turn_replay.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = []
    for label, log_path, traj_path in (("hack", args.hack, args.traj_hack),
                                       ("fix", args.fix, args.traj_fix)):
        if not Path(log_path).exists():
            print(f"[skip] {label}: no log at {log_path}")
            continue
        log = load_log(log_path)
        sim = None
        tp = PROJ / traj_path if not Path(traj_path).is_absolute() else Path(traj_path)
        if tp.exists():
            npz = np.load(tp, allow_pickle=True)
            sim = {"dt": float(npz["dt"]),
                   "waist": np.asarray(npz["sim_waist_yaw"], dtype=np.float64),
                   "pelvis": np.unwrap(np.asarray(npz["sim_pelvis_yaw"], dtype=np.float64))}
            sim["pelvis"] -= sim["pelvis"][0]
        arms.append((label, log, sim, arm_metrics(log, label)))

    if not arms:
        sys.exit("no logs found -- run replay_turn.py on the Pi first")

    # ---- console report ----
    print(f"{'arm':<6} {'frames':>6} {'HW WTR':>8} {'waist rng':>10} {'pelvis rng':>11} "
          f"{'ccw net yaw':>12} {'ccw track':>10} {'twist/turn':>11}")
    for label, log, sim, m in arms:
        print(f"{label:<6} {m['frames']:>6d} {m['hw_wtr']:>8.4f} {m['waist_range']:>9.3f}r "
              f"{m['pelvis_yaw_range']:>10.3f}r {m['ccw_net_yaw']:>11.3f}r "
              f"{m['ccw_tracking']:>9.1%} {m['twist_to_turn']:>11.2f}")
    if len(arms) == 2:
        wtr = {label: m["hw_wtr"] for label, _, _, m in arms}
        if np.isfinite(wtr["hack"]) and wtr["fix"] > 1e-6:
            ratio = wtr["hack"] / wtr["fix"]
            print(f"\nHW WTR ratio hack/fix = {ratio:.1f}x "
                  f"({'EXPECTED direction (hack twists more)' if ratio > 1 else 'INVERTED -- investigate'})")

    # ---- figure: rows = waist twist / pelvis yaw, cols = arms ----
    ncols = len(arms)
    fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 5.2), sharex="col",
                             sharey="row", squeeze=False)
    fig.patch.set_facecolor("white")

    for c, (label, log, sim, m) in enumerate(arms):
        t = log["t"]
        ax_w, ax_p = axes[0][c], axes[1][c]

        for ax in (ax_w, ax_p):
            shade_phases(ax, t, log["yaw_cmd"])
            style_axis(ax)

        if sim is not None:
            ts = np.arange(len(sim["waist"])) * sim["dt"]
            ax_w.plot(ts, sim["waist"], color=GRAY_REF, lw=1.2, ls="--", label="sim")
            ax_p.plot(ts, sim["pelvis"], color=GRAY_REF, lw=1.2, ls="--", label="sim")
        ax_w.plot(t, log["waist_yaw_rad"], color=BLUE, lw=1.6, label="hardware")
        ax_p.plot(t, log["pelvis_yaw_int"], color=BLUE, lw=1.6, label="hardware")

        ax_w.set_title(f"{label.upper()}  (heading = {'torso' if label == 'hack' else 'pelvis'})",
                       fontsize=10, color=INK, pad=18)
        ax_w.text(0.02, 0.95, f"HW WTR = {m['hw_wtr']:.3f}", transform=ax_w.transAxes,
                  fontsize=9, color=INK, va="top")
        # phase labels once per column, on the top row
        ccw_idx = np.flatnonzero(log["yaw_cmd"] > 1e-6)
        cw_idx = np.flatnonzero(log["yaw_cmd"] < -1e-6)
        for idx, txt in ((ccw_idx, "+0.5 rad/s"), (cw_idx, "−0.5 rad/s")):
            if idx.size:
                ax_w.text(t[idx].mean(), 1.02, txt, transform=ax_w.get_xaxis_transform(),
                          ha="center", fontsize=8, color=MUTED)
        ax_p.set_xlabel("time (s)", fontsize=9, color=MUTED)

    axes[0][0].set_ylabel("waist twist (rad)", fontsize=9, color=INK)
    axes[1][0].set_ylabel("pelvis yaw, gyro-integrated (rad)", fontsize=9, color=INK)
    axes[0][-1].legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle("Open-loop turn replay on hardware: torso-heading policy twists the waist, "
                 "pelvis-heading policy does not", fontsize=10, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = PROJ / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
