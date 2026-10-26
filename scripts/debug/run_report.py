#!/usr/bin/env python3
"""One page per hardware run: what the robot did, in numbers and in plots.

WHY. After every robot run the same questions get asked, and each used to need its own
script and its own reading of the CSV: did it fall and when, how far did it lean, how much of
that lean did the encoders see, how close did the COM get to the edge of the feet, did the
knees do what they were told, did the loop keep 40 Hz. This answers all of them at once from
the frame log and its sidecars, and writes them twice:

  <stem>.report.md    numbers and tables only -- the version Claude reads; no images needed
  <stem>.report.html  the same numbers plus plots (matplotlib PNGs embedded as base64, so the
                      file is self-contained and opens anywhere)

    python scripts/debug/run_report.py logs/pi/20261025_031319_standing.csv
    python scripts/debug/run_report.py <log.csv> --out-dir /tmp/x --replay

The body numbers come from log_pose.py (pose, COM, support margins, hidden flex) -- read its
docstring for the assumptions -- and its pose CSV/summary are written alongside. The event
timeline is diagnose_run.py's, verbatim. Optional inputs are used when they exist and skipped
when they do not, never required:
  <stem>.meta.json      provenance, deploy settings, FSR layout, servo-telemetry decoding
  <stem>.events.jsonl   operator key markers -> vertical lines on every plot + a table
  <stem>.console.txt    console output -> warnings/errors pulled out with their times
  v2 columns            acc_*, load_raw/temp_raw/volt_raw, q_prior, posture_*, fsr_v2..

--replay starts the sim at the log's first measured pose and drives it OPEN LOOP with the
logged commands, then overlays sim and real pelvis pitch. The commands were computed from the
REAL robot's sensors, so this is not what the controller would have done in sim; it is what
the sim plant does when handed the robot's own command stream. The gap between the two lines
is the sim-to-real gap for this run. It is skipped with a note if anything about it fails.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import html
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "deploy"))
sys.path.insert(0, str(ROOT / "scripts" / "debug"))

import log_pose  # noqa: E402
from diagnose_run import STALL_MULT, fsr_conductance  # noqa: E402
from sim_real_map import DEFAULT_MAP, SimRealMap  # noqa: E402

# diagnose_run's defaults for the FSR divider, used only when the meta does not record them.
RFIXED_DEFAULT, VCC_DEFAULT = 2000.0, 3.3
FSR_OPEN_V = 0.1        # divider volts this low = the sensor is open (no load on that pad)
# Console lines worth surfacing. Deliberately broad: a false positive costs one line.
WARN_RE = re.compile(r"(?i)(warn|error|fail|safety|stall|slow|!!|traceback|exception|reject|"
                     r"abort|cut|torque off|tipped|nan\b|timeout|retry|clip)")
MAX_CONSOLE_LINES = 60
REPLAY_MODEL_16CM = ROOT / "models" / "humanoid_real_v2_soleplate_flex.xml"
REPLAY_MODEL_12CM = ROOT / "models" / "humanoid_real_v2_compliant.xml"

# Reference palette (dataviz skill): fixed slot order, recessive ink for axes and grid.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------------------------------
# sidecars
# ---------------------------------------------------------------------------------------

def sidecar(log_path, kind):
    p = Path(log_path)
    return p.with_name(f"{p.stem}.{kind}")


def load_events(log_path, meta):
    """(events, n_unreadable). Each event gets a t_rel even when only t_wall was written."""
    p = sidecar(log_path, "events.jsonl")
    if not p.exists():
        return [], 0
    t0 = meta.get("started_unix")
    out, bad = [], 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if not isinstance(e, dict):
            bad += 1
            continue
        t = e.get("t_rel")
        if t is None and e.get("t_wall") is not None and t0 is not None:
            t = float(e["t_wall"]) - float(t0)
        try:
            e["t_rel"] = None if t is None else float(t)
        except (TypeError, ValueError):
            e["t_rel"] = None
        out.append(e)
    return out, bad


def load_console(log_path, meta):
    """(warning lines as (t_rel or None, text), total line count). Lines may carry the
    '<unix time> | ' prefix the v2 console tee writes; older captures do not."""
    p = sidecar(log_path, "console.txt")
    if not p.exists():
        return None, 0
    t0 = meta.get("started_unix")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    out, seen = [], set()
    for ln in lines:
        t, text = None, ln
        m = re.match(r"^\s*(\d{9,}\.\d+)\s*\|\s?(.*)$", ln)
        if m:
            text = m.group(2)
            if t0 is not None:
                t = float(m.group(1)) - float(t0)
        if WARN_RE.search(text) and text not in seen:
            seen.add(text)
            out.append((t, text.strip()))
    return out[:MAX_CONSOLE_LINES], len(lines)


def fsr_layout(meta, n):
    """Per-channel (name, foot, x_mm, y_mm) and divider params; v1 metas name no positions."""
    f = meta.get("fsr") if isinstance(meta.get("fsr"), dict) else {}
    chans = f.get("channels")
    names = []
    if isinstance(chans, list) and chans and isinstance(chans[0], dict):
        for k in range(n):
            c = chans[k] if k < len(chans) else {}
            names.append((c.get("name") or f"fsr_v{k}", c.get("foot"), c.get("x_mm"), c.get("y_mm")))
    else:
        # schema 1: "0,1" on 0x48 = left heel, left toe (frame_log v2's legacy naming,
        # and diagnose_run's TOE = channel 1)
        legacy = [("L_heel (assumed)", "left", None, None), ("L_toe (assumed)", "left", None, None)]
        names = [legacy[k] if k < 2 else (f"fsr_v{k}", None, None, None) for k in range(n)]
    rf = f.get("rfixed_ohm") or f.get("rfixed")
    vcc = f.get("vcc")
    assumed = rf is None or vcc is None
    return names, float(rf or RFIXED_DEFAULT), float(vcc or VCC_DEFAULT), assumed


def decode_load(raw):
    """load_raw -> signed 0.1 % of max output (sign-magnitude, bit 10 = direction), the rule
    frame_log v2 copies into meta servo_telemetry.decoding from the SDK."""
    raw = np.asarray(raw, dtype=float)
    out = np.full(raw.shape, np.nan)
    ok = np.isfinite(raw)
    r = raw[ok].astype(int)
    v = (r & 0x3FF).astype(float)
    out[ok] = np.where(r & 0x400, -v, v)
    return out


# ---------------------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------------------

def replay(xml, smap, q0, q_cmd, t, settle_s=2.0):
    """Open-loop sim driven by the logged commands. Returns (t, pitch, roll, note)."""
    import mujoco
    from sim_cop import pelvis_attitude, settle

    m = mujoco.MjModel.from_xml_path(str(xml))
    dofs = {j.mjcf: k for k, j in enumerate(smap.joints)}
    perm = np.array([dofs[m.joint(int(m.actuator_trnid[a, 0])).name] for a in range(m.nu)])
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    data = settle(m, seconds=settle_s, target=np.clip(q0[perm], lo, hi))
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    pitch = np.full(len(t), np.nan)
    roll = np.full(len(t), np.nan)
    note = "sim stayed up for the whole log"
    dt_default = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 0.025
    for i in range(len(t)):
        data.ctrl[:] = np.clip(q_cmd[i][perm], lo, hi)
        dt = (t[i + 1] - t[i]) if i + 1 < len(t) and np.isfinite(t[i + 1] - t[i]) else dt_default
        for _ in range(max(1, int(round(max(dt, 0.0) / m.opt.timestep)))):
            mujoco.mj_step(m, data)
        pitch[i], roll[i] = pelvis_attitude(m, data)
        if data.xmat[base].reshape(3, 3)[2, 2] < log_pose.TIP_COS:
            note = f"sim tipped over at t={t[i]:.2f} s; nothing after that is plotted"
            break
    return pitch, roll, note


# ---------------------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
        "legend.frameon": False, "lines.linewidth": 1.4})
    return plt


def figure(plt, series, title, ylabel, events, fall_t, hline=None, height=2.6):
    """One time-series chart, one y-axis. series = [(t, y, label, color, style)]."""
    fig, ax = plt.subplots(figsize=(10, height))
    for t, y, label, color, style in series:
        ax.plot(t, y, style, color=color, label=label, linewidth=1.3 if style == "-" else 1.0)
    if hline is not None:
        ax.axhline(hline, color=INK2, linewidth=0.8)
    _markers(ax, events, fall_t)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("t_rel (s)")
    if len(series) > 1:
        # above the plot area, right-aligned against the left-aligned title: dense traces leave
        # no empty corner inside the axes
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=min(len(series), 4), fontsize=8)
    fig.tight_layout()
    return _b64(plt, fig)


def _markers(ax, events, fall_t):
    for e in events:
        if e.get("t_rel") is not None:
            ax.axvline(e["t_rel"], color=C4, linewidth=0.8, linestyle="--", alpha=0.7)
            ax.text(e["t_rel"], 1.0, f" {e.get('label') or e.get('key') or ''}", color=C4,
                    fontsize=7, rotation=90, va="top", ha="left", transform=ax.get_xaxis_transform())
    if fall_t is not None:
        ax.axvline(fall_t, color=INK, linewidth=1.2)
        ax.text(fall_t, 1.0, " tip", color=INK, fontsize=8, va="top", transform=ax.get_xaxis_transform())


def joint_grid(plt, t, cmd, meas, names, events, fall_t):
    """Small multiples: one panel per joint, commanded vs measured, shared time axis."""
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.7 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k, nm in zip(axes, range(n), names):
        ax.plot(t, cmd[:, k], "--", color=C2, linewidth=1.0, label="commanded")
        ax.plot(t, meas[:, k], "-", color=C1, linewidth=1.3, label="measured")
        _markers(ax, events, fall_t)
        ax.set_ylabel(f"{nm}\n(deg)", fontsize=8)
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8)
    axes[0].set_title("Leg joints: commanded (u_cmd) vs measured (u_meas)", loc="left")
    axes[-1].set_xlabel("t_rel (s)")
    fig.tight_layout()
    return _b64(plt, fig)


def _b64(plt, fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------------------

def _f(v, nd=2, unit=""):
    if v is None:
        return "-"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "nan" if not np.isfinite(v) else f"{v:.{nd}f}{unit}"


def _stat_row(name, s, unit):
    if not s:
        return [name, "-", "-", "-", "-", "-", "-"]
    return [name, _f(s["median"]), _f(s["min"]), _f(s["max"]), _f(s["std"]), _f(s["p05"]), _f(s["p95"])]


def build(log_path, model=None, out_dir=None, do_replay=False, replay_model=None, map_path=DEFAULT_MAP):
    log_path = Path(log_path)
    cols, table, S, A = log_pose.analyze(log_path, model, "u_cmd", map_path)
    pose_csv, pose_json = log_pose.write_outputs(log_path, cols, table, S, out_dir)
    rows, meta = log_pose.load_log(log_path)
    smap = SimRealMap(map_path)
    dofs = [j.dof for j in smap.joints]
    col = lambda name: log_pose.column(rows, name)  # noqa: E731
    t = col("t_rel")
    end = int(np.argmax(A["fallen"] == 1)) if S["fell"] else len(t)
    fall_t = S["fall"]["t_rel"] if S["fell"] else None
    events, bad_events = load_events(log_path, meta)
    console, n_console = load_console(log_path, meta)
    sections = []        # (title, [blocks]); block = ("kv", pairs) | ("table", head, rows) | ("pre", text)
    figs = {}            # section title -> [base64 png]
    plt = _plt()

    # ---- run -----------------------------------------------------------------------
    dur = S["duration_s"]
    lm = col("loop_ms")
    hz = len(rows) / dur if dur else None
    sections.append(("Run", [("kv", [
        ("log", str(log_path)),
        ("script / argv", " ".join(str(a) for a in (meta.get("argv") or [meta.get("script", "?")]))),
        ("started", meta.get("started_iso", "-")),
        ("schema_version", S["schema_version"]),
        ("rows", f"{S['rows']} written, {meta.get('rows_dropped', '-')} dropped by the logger"),
        ("duration", f"{_f(dur, 1)} s at {_f(hz, 1)} Hz"),
        ("model", f"{S['model']['xml']} -- {S['model']['why']} (feet {S['model']['foot_len_mm']:.0f} x "
                  f"{S['model']['foot_width_mm']:.0f} mm, md5 {S['model']['md5']})"),
        ("stance / policy", f"{meta.get('residual_config', '-')} / "
                            f"{next((k for k in (meta.get('file_md5') or {}) if k.endswith('.npz')), '-')}"),
        ("sidecars", f"events.jsonl {'yes (%d)' % len(events) if events or bad_events else 'no'}, "
                     f"console.txt {'yes (%d lines)' % n_console if console is not None else 'no'}"),
    ])]))

    # ---- verdict -------------------------------------------------------------------
    ip, ir = A["imu_pitch_deg"], A["imu_roll_deg"]
    w = slice(0, end)

    def at(arr, fn):
        a = arr[w]
        if not np.isfinite(a).any():
            return "-"
        k = int(fn(a))
        return f"{a[k]:+.1f} deg at t={t[k]:.1f} s"

    verdict = (f"FELL {S['fall']['direction']} at t={S['fall']['t_rel']:.2f} s "
               f"(upright_cos {S['fall']['upright_cos']:.2f} < {log_pose.TIP_COS}); stats below use the "
               f"{S['standing_window_s']:.1f} s before it" if S["fell"]
               else f"did not tip (upright_cos stayed >= {log_pose.TIP_COS} for all {_f(dur, 1)} s)")
    sections.append(("Verdict", [("kv", [
        ("outcome", verdict),
        ("max pitch (forward)", at(ip, np.nanargmax)), ("min pitch (back)", at(ip, np.nanargmin)),
        ("max roll (left)", at(ir, np.nanargmax)), ("min roll (right)", at(ir, np.nanargmin)),
        ("COM outside the support polygon", f"{100 * (S['share_outside_support'] or 0):.1f}% of standing frames "
                                            "(on hardware that means held, rocking onto an edge, or falling)"),
    ])]))

    # ---- attitude & flex -------------------------------------------------------------
    head = ["quantity (deg)", "median", "min", "max", "std", "p05", "p95"]
    sections.append(("Pelvis attitude and hidden flex", [
        ("table", head, [_stat_row("IMU pitch", S["imu_pitch_deg"], "deg"),
                         _stat_row("encoder-implied pitch", S["enc_pitch_deg"], "deg"),
                         _stat_row("flex pitch (IMU - enc)", S["flex_pitch_deg"], "deg"),
                         _stat_row("IMU roll", S["imu_roll_deg"], "deg"),
                         _stat_row("encoder-implied roll", S["enc_roll_deg"], "deg"),
                         _stat_row("flex roll (IMU - enc)", S["flex_roll_deg"], "deg")]),
        ("kv", [("share of IMU pitch excursion the encoders miss", _f(S["pitch_missed_by_encoders"], 3)),
                ("share of IMU roll excursion the encoders miss", _f(S["roll_missed_by_encoders"], 3)),
                ("IMU pitch std / encoder pitch std", _f(S["pitch_excursion_ratio_imu_over_enc"])),
                ("note", "encoder-implied = the pelvis attitude that puts both soles flat; the static part of "
                         "flex also holds IMU mounting error and the model's ~0.75 deg sole roll. "
                         "share > 1 = the joints moved AGAINST the lean (a feedback law correcting it).")])]))
    series = [(t, ip, "IMU (pg)", C1, "-"), (t, A["enc_pitch_deg"], "encoder-implied", C2, "-")]
    if "acc_pitch_deg" in A:
        series.append((t, A["acc_pitch_deg"], "raw accelerometer", C3, "-"))
    figs["Pelvis attitude and hidden flex"] = [
        figure(plt, series, "Pelvis pitch (+ forward)", "deg", events, fall_t, hline=0),
        figure(plt, [(t, ir, "IMU (pg)", C1, "-"), (t, A["enc_roll_deg"], "encoder-implied", C2, "-")]
               + ([(t, A["acc_roll_deg"], "raw accelerometer", C3, "-")] if "acc_roll_deg" in A else []),
               "Pelvis roll (+ robot's left)", "deg", events, fall_t, hline=0),
        figure(plt, [(t, A["flex_pitch_deg"], "pitch", C1, "-"), (t, A["flex_roll_deg"], "roll", C2, "-")],
               "Hidden flex: IMU minus encoder-implied attitude", "deg", events, fall_t, hline=0)]

    # ---- balance ---------------------------------------------------------------------
    mn = S["support_margin_min"] or {}
    fm = S["foot_edge_min_while_stance"]
    bal = [("support margin (polygon)", f"median {_f((S['support_margin_mm'] or {}).get('median'), 1)} mm, "
                                        f"quiet-standing median {_f(S['support_margin_quiet_median_mm'], 1)} mm"),
           ("minimum margin", f"{_f(mn.get('mm'), 1)} mm at t={_f(mn.get('t_rel'))} s, nearest edge "
                              f"{mn.get('nearest', '-')}, stance {mn.get('stance', '-')}"),
           ("encoder-only margin (flat feet)", f"median {_f((S['support_margin_flat_mm'] or {}).get('median'), 1)} mm, "
                                               f"min {_f((S['support_margin_flat_mm'] or {}).get('min'), 1)} mm"),
           ("COM x (+ forward of mid-feet)", f"median {_f(S['com_x_mm']['median'], 1)} mm, "
                                            f"p05..p95 {_f(S['com_x_mm']['p05'], 1)}..{_f(S['com_x_mm']['p95'], 1)} mm"),
           ("COM y (+ left)", f"median {_f(S['com_y_mm']['median'], 1)} mm"),
           ("stance foot frames", f"R {S['stance_frames']['R']}, L {S['stance_frames']['L']}; both feet "
                                  f"supporting {S['support_frames']['both']}")]
    for s in ("R", "L"):
        if s in fm:
            bal.append((f"closest edge while {s} is stance", f"{fm[s]['edge']} {fm[s]['mm']:.1f} mm"))
    if "com_err_x_mm" in S:
        bal.append(("COM error from tracking (cmd - meas)",
                    f"x median {_f(S['com_err_x_mm']['median'], 1)} mm, y median {_f(S['com_err_y_mm']['median'], 1)} mm, "
                    f"|err| p95 {_f(S['com_err_abs_p95_mm'], 1)} mm"))
    sections.append(("Balance: COM over the feet", [("kv", bal)]))
    figs["Balance: COM over the feet"] = [
        figure(plt, [(t, A["support_margin_mm"], "IMU attitude", C1, "-"),
                     (t, A["support_margin_flat_mm"], "encoders only", C2, "-")],
               "COM margin to the support polygon (+ inside)", "mm", events, fall_t, hline=0),
        figure(plt, [(t, A["com_x_mm"], "measured", C1, "-")]
               + ([(t, A["com_cmd_x_mm"], "commanded", C2, "--")] if "com_cmd_x_mm" in A else [])
               + [(t, A["com_flat_x_mm"], "encoders only", C3, "-")],
               "COM fore/aft (+ forward of the mid-feet point)", "mm", events, fall_t, hline=0)]

    # ---- joints ----------------------------------------------------------------------
    upr = smap.units_per_rad
    # out-of-range reads (garbage that passed the checksum) are NaN here, as in log_pose
    u_cmd = log_pose.units_matrix(rows, smap, "u_cmd")
    u_meas = log_pose.units_matrix(rows, smap, "u_meas")
    err = np.degrees((u_cmd - u_meas) / upr)
    jt = []
    for k, d in enumerate(dofs):
        e = err[:end, k]
        e = e[np.isfinite(e)]
        if e.size:
            jt.append([d, _f(np.sqrt(np.mean(e ** 2))), _f(np.mean(e)), _f(np.max(np.abs(e)))])
    jt.sort(key=lambda r: -float(r[1]))
    sections.append(("Joint tracking (u_cmd - u_meas)", [
        ("table", ["joint", "rms deg", "mean deg", "max |err| deg"], jt)]))
    legs = ["R_hip_pitch", "L_hip_pitch", "R_knee", "L_knee", "waist_pitch"]
    li = [dofs.index(x) for x in legs]
    q_c = np.degrees(np.asarray(smap.units_to_rad(np.nan_to_num(u_cmd)), dtype=float))[:, li]
    q_m = np.degrees(np.asarray(smap.units_to_rad(np.nan_to_num(u_meas)), dtype=float))[:, li]
    q_c[~np.isfinite(u_cmd[:, li])] = np.nan
    q_m[~np.isfinite(u_meas[:, li])] = np.nan
    figs["Joint tracking (u_cmd - u_meas)"] = [joint_grid(plt, t, q_c, q_m, legs, events, fall_t)]

    # ---- feedback law / posture (v2) -------------------------------------------------
    extra = []
    qp = {d: col(f"q_prior.{d}") for d in ("R_knee", "L_knee", "R_hip_pitch", "L_hip_pitch")}
    qp = {d: np.degrees(v) for d, v in qp.items() if np.isfinite(v).any()}
    for d, v in qp.items():
        extra.append((f"q_prior.{d}", f"median {_f(np.nanmedian(v))} deg, range {_f(np.nanmin(v))}..{_f(np.nanmax(v))} deg"))
    post = {c: col(c) for c in ("posture_lean", "posture_zero", "posture_corr")}
    post = {c: v for c, v in post.items() if np.isfinite(v).any()}
    for c, v in post.items():
        extra.append((c, f"median {_f(np.nanmedian(v))} deg, range {_f(np.nanmin(v))}..{_f(np.nanmax(v))} deg"))
    if extra:
        sections.append(("Feedback law and posture controller", [("kv", extra)]))
        ser = [(t, v, f"q_prior.{d}", c, "-") for (d, v), c in zip(qp.items(), (C1, C2, C3, C4))]
        ser += [(t, v, c, col_, "--") for (c, v), col_ in zip(post.items(), (C2, C3, C4))]
        figs["Feedback law and posture controller"] = [
            figure(plt, ser[:4], "Law / posture contributions", "deg", events, fall_t, hline=0)]

    # ---- loop timing -----------------------------------------------------------------
    bm = col("busy_ms")
    if np.isfinite(lm).any():
        med = float(np.nanmedian(lm))
        sections.append(("Loop timing", [("kv", [
            ("loop_ms", f"median {med:.1f}, p95 {np.nanpercentile(lm, 95):.1f}, max {np.nanmax(lm):.1f}"),
            ("busy_ms", f"median {_f(np.nanmedian(bm), 1)}, p95 {_f(np.nanpercentile(bm, 95), 1)}, "
                        f"max {_f(np.nanmax(bm), 1)}" if np.isfinite(bm).any() else "-"),
            ("stalls", f"{int(np.nansum(lm > STALL_MULT * med))} frames over {STALL_MULT} x median"),
            ("rejected servo reads", int(np.nanmax(col('rej_total'))) if np.isfinite(col('rej_total')).any() else "-"),
            ("encoder rows held from the previous frame",
             f"{S['u_meas_filled_rows']} ({S['u_meas_bad_values']} joint values NaN or outside 0..1023)"
             + (f" at t={S['u_meas_filled_at_t']}" if S["u_meas_filled_rows"] else "")),
        ])]))
        figs["Loop timing"] = [figure(plt, [(t, lm, "loop_ms", C1, "-"), (t, bm, "busy_ms", C2, "-")],
                                      "Control loop period and busy time", "ms", events, fall_t,
                                      hline=25.0, height=2.2)]

    # ---- feet (FSR) ------------------------------------------------------------------
    fcols = sorted((c for c in rows[0] if re.fullmatch(r"fsr_v\d+", c)), key=lambda c: int(c[5:]))
    fv = {c: col(c) for c in fcols}
    fv = {c: v for c, v in fv.items() if np.isfinite(v).any()}
    if fv:
        names, rf, vcc, assumed = fsr_layout(meta, len(fcols))
        g = {c: fsr_conductance(v, rf, vcc) for c, v in fv.items()}
        kv = [("divider", f"R_fixed {rf:g} ohm, Vcc {vcc:g} V" + (" (ASSUMED: not in the meta)" if assumed else ""))]
        for c, v in fv.items():
            nm = names[int(c[5:])][0]
            kv.append((f"{c} = {nm}", f"median {np.nanmedian(v):.3f} V, range {np.nanmin(v):.3f}..{np.nanmax(v):.3f} V, "
                                      f"open/unloaded (< {FSR_OPEN_V:g} V) {100 * np.nanmean(v < FSR_OPEN_V):.0f}% of frames"))
        ser = [(t, v, names[int(c[5:])][0], cc, "-") for (c, v), cc in zip(fv.items(), (C1, C2, C3, C4))]
        plots = [figure(plt, ser, "FSR raw divider volts", "V", events, fall_t)]
        # centre of pressure per foot, only where the meta gives every channel's position
        for foot in ("left", "right"):
            ch = [(c, names[int(c[5:])]) for c in fv if names[int(c[5:])][1] == foot]
            if ch and all(n[2] is not None for _, n in ch):
                G = np.column_stack([np.nan_to_num(g[c]) for c, _ in ch])
                x = np.array([float(n[2]) for _, n in ch])
                tot = G.sum(axis=1)
                cop = np.where(tot > 0, (G @ x) / np.maximum(tot, 1e-12), np.nan)
                kv.append((f"{foot} foot CoP x (+ toe)", f"median {_f(np.nanmedian(cop), 1)} mm, "
                                                        f"p05..p95 {_f(np.nanpercentile(cop, 5), 1)}.."
                                                        f"{_f(np.nanpercentile(cop, 95), 1)} mm"))
                plots.append(figure(plt, [(t, cop, f"{foot} CoP", C1, "-")],
                                    f"{foot} foot centre of pressure (from FSR conductance)", "mm", events,
                                    fall_t, hline=0))
        toe = [c for c in fv if "toe" in names[int(c[5:])][0].lower()]
        heel = [c for c in fv if "heel" in names[int(c[5:])][0].lower()]
        if toe and heel:
            gt = sum(np.nan_to_num(g[c]) for c in toe)
            gh = sum(np.nan_to_num(g[c]) for c in heel)
            share = np.where(gt + gh > 0, gt / np.maximum(gt + gh, 1e-12), np.nan)
            kv.append(("toe share of FSR conductance", f"median {_f(np.nanmedian(share))}, "
                                                       f"p05..p95 {_f(np.nanpercentile(share, 5))}.."
                                                       f"{_f(np.nanpercentile(share, 95))} (1 = all on the toe)"))
        sections.append(("Feet (FSR)", [("kv", kv)]))
        figs["Feet (FSR)"] = plots

    # ---- servo telemetry (v2) --------------------------------------------------------
    tel = []
    load = np.column_stack([decode_load(col(f"load_raw.{d}")) for d in dofs])
    temp = np.column_stack([col(f"temp_raw.{d}") for d in dofs])
    volt = np.column_stack([col(f"volt_raw.{d}") for d in dofs]) * 0.1
    if np.isfinite(load).any() or np.isfinite(temp).any() or np.isfinite(volt).any():
        for k, d in enumerate(dofs):
            if not (np.isfinite(load[:, k]).any() or np.isfinite(temp[:, k]).any()):
                continue
            tel.append([d, _f(np.nanmedian(np.abs(load[:, k])) / 10, 1), _f(np.nanmax(np.abs(load[:, k])) / 10, 1),
                        _f(np.nanmax(temp[:, k]), 0), _f(np.nanmin(volt[:, k]), 1)])
        tel.sort(key=lambda r: -float(r[2]) if r[2] not in ("-", "nan") else 0)
        dec = (meta.get("servo_telemetry") or {}).get("decoding") if isinstance(meta.get("servo_telemetry"), dict) else None
        blocks = [("table", ["joint", "|load| median %", "|load| max %", "temp max C", "volt min V"], tel),
                  ("kv", [("decoding", "load = sign-magnitude, bit 10 = direction, 0.1 % of max output (a PWM "
                                       "torque proxy); volt = raw x 0.1 V; temp = raw C"
                                       + ("; meta records: " + json.dumps(dec) if dec else
                                          "; NOT recorded in this meta"))])]
        sections.append(("Servo telemetry", blocks))
        ser = [(t, load[:, dofs.index(d)] / 10, d, cc, "-")
               for d, cc in zip(("R_knee", "L_knee", "R_hip_pitch", "L_hip_pitch"), (C1, C2, C3, C4))]
        figs["Servo telemetry"] = [figure(plt, ser, "Leg servo load (signed, % of max output)", "%",
                                          events, fall_t, hline=0)]
        if np.isfinite(temp).any():
            figs["Servo telemetry"].append(figure(
                plt, [(t, np.nanmax(temp, axis=1), "hottest servo", C2, "-")], "Hottest servo", "C",
                events, fall_t, height=2.0))

    # ---- events / console / timeline ------------------------------------------------
    if events or bad_events:
        rows_e = [[_f(e.get("t_rel")), str(e.get("step", "-")), str(e.get("key", "")), str(e.get("label", "")),
                   str(e.get("note", ""))] for e in sorted(events, key=lambda e: (e.get("t_rel") is None, e.get("t_rel") or 0))]
        blocks = [("table", ["t_rel s", "step", "key", "label", "note"], rows_e)]
        if bad_events:
            blocks.append(("kv", [("unreadable lines", bad_events)]))
        sections.append(("Events (events.jsonl)", blocks))
    if console is not None:
        sections.append(("Console warnings (console.txt)", [
            ("table", ["t_rel s", "line"], [[_f(tt, 1), txt] for tt, txt in console]) if console
            else ("kv", [("result", f"no warning-like lines in {n_console} lines")])]))
    sections.append(("Timeline (diagnose_run.py)", [("pre", diagnose_text(log_path))]))

    # ---- replay ----------------------------------------------------------------------
    if do_replay:
        sections.append(("Sim replay (open loop)", replay_section(log_path, meta, S, rows, smap, t, end,
                                                                  replay_model, plt, figs, events, fall_t)))

    sections.append(("Files", [("kv", [("pose CSV", str(pose_csv)), ("pose summary", str(pose_json))])]))
    return sections, figs, S, meta


def replay_section(log_path, meta, S, rows, smap, t, end, replay_model, plt, figs, events, fall_t):
    title = "Sim replay (open loop)"
    if replay_model:
        xml = Path(replay_model)
    else:
        xml = (REPLAY_MODEL_12CM if str(meta.get("started_iso", ""))[:10] < log_pose.SOLE_PLATES_FROM
               and meta.get("started_iso") else REPLAY_MODEL_16CM)
    try:
        q_meas = np.array([log_pose.frame_log.pose_from_row(r, smap)["q_meas"] for r in rows], dtype=float)
        q_meas, _ = log_pose.forward_fill(q_meas)
        q_cmd, _ = log_pose.forward_fill(log_pose.joint_matrix(rows, smap, "u_cmd"))
        sp, sr, note = replay(xml, smap, q_meas[0], q_cmd, t)
    except Exception as exc:          # a replay must never cost the report
        return [("kv", [("skipped", f"replay failed: {type(exc).__name__}: {exc}")])]
    real = log_pose.pitch_roll_deg(np.column_stack([log_pose.column(rows, f"pg_{a}") for a in "xyz"]))[0]
    ok = np.isfinite(sp) & np.isfinite(real)
    ok[end:] = False
    gap = sp - real
    kv = [("what this is", "sim started at the first measured pose (settled 2 s), then driven OPEN LOOP by the "
                           "logged u_cmd; the commands came from the REAL robot's sensors. Raw MJCF actuators, "
                           "no servo lag model."),
          ("model", str(xml.relative_to(ROOT)) if xml.resolve().is_relative_to(ROOT) else str(xml)),
          ("result", note)]
    if ok.sum() > 10:
        kv += [("pitch gap sim - real", f"median {np.median(gap[ok]):+.2f} deg, rms {np.sqrt(np.mean(gap[ok] ** 2)):.2f} deg"),
               ("sim pitch std vs real", f"{np.std(sp[ok]):.2f} vs {np.std(real[ok]):.2f} deg"),
               ("correlation of the two pitch traces",
                f"{np.corrcoef(sp[ok], real[ok])[0, 1]:+.2f}" if min(np.std(sp[ok]), np.std(real[ok])) > 1e-9
                else "- (one trace is constant)")]
    figs[title] = [figure(plt, [(t, real, "real (IMU pg)", C1, "-"), (t, sp, "sim, open loop", C2, "-")],
                          "SIM REPLAY: pelvis pitch, sim vs real (open loop on the logged commands)", "deg",
                          events, fall_t, hline=0)]
    return [("kv", kv)]


def diagnose_text(log_path):
    """diagnose_run.py's printed timeline, captured verbatim."""
    import diagnose_run
    buf = io.StringIO()
    argv = sys.argv
    try:
        sys.argv = ["diagnose_run.py", str(log_path)]
        with contextlib.redirect_stdout(buf):
            diagnose_run.main()
    except (Exception, SystemExit) as exc:
        buf.write(f"\n[diagnose_run failed: {type(exc).__name__}: {exc}]")
    finally:
        sys.argv = argv
    return buf.getvalue().strip("\n")


# ---------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------

def _md_cell(v):
    return str(v).replace("|", "\\|").replace("\n", " ")


def render_md(stem, sections):
    out = [f"# Run report: {stem}", "",
           "Numbers only; the .report.html next to this has the same numbers with plots.", ""]
    for title, blocks in sections:
        out += [f"## {title}", ""]
        for b in blocks:
            if b[0] == "kv":
                out += [f"- **{k}**: {_md_cell(v)}" for k, v in b[1]]
            elif b[0] == "table":
                out.append("| " + " | ".join(_md_cell(h) for h in b[1]) + " |")
                out.append("|" + "---|" * len(b[1]))
                out += ["| " + " | ".join(_md_cell(c) for c in r) + " |" for r in b[2]]
            elif b[0] == "pre":
                out += ["```", b[1], "```"]
            out.append("")
    return "\n".join(out)


CSS = """
:root{--bg:#f4f4f2;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121211;--surface:#1a1a19;
 --ink:#f0efec;--ink2:#c3c2b7;--line:#33322f;--accent:#3987e5}}
:root[data-theme="dark"]{--bg:#121211;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#33322f;--accent:#3987e5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;padding:16px}
main{max-width:1100px;margin:0 auto}
h1{font-size:20px;margin:4px 0 2px} h2{font-size:15px;margin:0 0 10px;color:var(--ink)}
.sub{color:var(--ink2);margin:0 0 16px}
section{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:0 0 14px;overflow-x:auto}
dl{display:grid;grid-template-columns:minmax(140px,280px) 1fr;gap:4px 14px;margin:0}
dt{color:var(--ink2)} dd{margin:0;font-variant-numeric:tabular-nums;word-break:break-word}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px;margin:4px 0 8px}
th,td{border-bottom:1px solid var(--line);padding:3px 10px 3px 0;text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600}
pre{white-space:pre-wrap;font:12px/1.45 ui-monospace,Consolas,monospace;margin:0;color:var(--ink)}
img{display:block;width:100%;height:auto;border-radius:4px;margin:8px 0 0;background:#fcfcfb}
@media (max-width:640px){dl{grid-template-columns:1fr}}
"""


def render_html(stem, sections, figs):
    e = html.escape
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
             f"<meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>Run report {e(stem)}</title><style>{CSS}</style></head><body><main>",
             f"<h1>Run report: {e(stem)}</h1><p class=sub>Hardware frame log, posed with MuJoCo by "
             f"log_pose.py. The .report.md next to this file has the same numbers without plots.</p>"]
    for title, blocks in sections:
        parts.append(f"<section><h2>{e(title)}</h2>")
        for b in blocks:
            if b[0] == "kv":
                parts.append("<dl>" + "".join(f"<dt>{e(str(k))}</dt><dd>{e(str(v))}</dd>" for k, v in b[1]) + "</dl>")
            elif b[0] == "table":
                parts.append("<table><thead><tr>" + "".join(f"<th>{e(h)}</th>" for h in b[1]) + "</tr></thead><tbody>"
                             + "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>" for r in b[2])
                             + "</tbody></table>")
            elif b[0] == "pre":
                parts.append(f"<pre>{e(b[1])}</pre>")
        for png in figs.get(title, []):
            parts.append(f"<img alt='{e(title)} plot' src='data:image/png;base64,{png}'>")
        parts.append("</section>")
    parts.append("</main></body></html>")
    return "".join(parts)


def main():
    p = argparse.ArgumentParser(description="Frame log -> one-page run report (.report.md + .report.html)")
    p.add_argument("log")
    p.add_argument("--model", default=None, help="MJCF for posing (default: chosen by log_pose.py from the log's date)")
    p.add_argument("--out-dir", default=None, help="default: next to the log")
    p.add_argument("--replay", action="store_true",
                   help="also drive the sim open loop with the logged commands and overlay its pelvis pitch")
    p.add_argument("--replay-model", default=None,
                   help=f"MJCF for --replay (default: the flex plant {REPLAY_MODEL_16CM.name} for 16 cm-foot logs)")
    p.add_argument("--map", default=str(DEFAULT_MAP))
    args = p.parse_args()

    log = Path(args.log)
    sections, figs, S, _ = build(log, args.model, args.out_dir, args.replay, args.replay_model, args.map)
    d = Path(args.out_dir) if args.out_dir else log.parent
    d.mkdir(parents=True, exist_ok=True)
    md, page = d / f"{log.stem}.report.md", d / f"{log.stem}.report.html"
    md.write_text(render_md(log.stem, sections), encoding="utf-8")
    page.write_text(render_html(log.stem, sections, figs), encoding="utf-8")
    print(f"{log}: {'FELL ' + S['fall']['direction'] + ' at t=' + str(S['fall']['t_rel']) if S['fell'] else 'no fall'}; "
          f"{sum(len(v) for v in figs.values())} plots")
    print(f"  -> {md}\n  -> {page} ({page.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
