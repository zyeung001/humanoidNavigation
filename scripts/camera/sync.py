#!/usr/bin/env python3
# sync.py  --  dev box only. Puts the camera and the frame log on one clock.
"""Camera (tags.csv or the raw video) + frame log -> one CSV keyed on the log's t_wall.

WHY A TAP. The phone and the Pi keep separate clocks, and a phone's video timestamps start
at zero, so nothing in the files says which video frame goes with which log row. A firm
tap on the pelvis is an event both see at the same instant: a jump in tag motion in the
video, a spike in |acceleration| (schema v2 acc_x/y/z) or angular rate (av_*, the only
choice in a v1 log) in the log. A v2 log is tried both ways and the signal whose start/end
events match best is used: a finger tap's acceleration spike lasts a few ms, and the log
keeps one IMU sample per 25 ms loop, so it can fall between samples; the body's rocking
(gyro) lasts far longer. Tapping again at the END measures clock drift between the
two -- phone video clocks and nominal frame rates are off by up to ~0.1%, i.e. 60 ms over
a minute, which is more than a frame.

    python scripts/camera/sync.py run.tags.csv logs/pi/20261026_101500_standing.csv
    python scripts/camera/sync.py run.mp4      logs/pi/...csv     # frame difference instead of tags

HOW. (1) Coarse: cross-correlate the two motion-event signals over every lag, so the
match is found wherever the recordings overlap and from every shared event (taps, pushes,
a fall), not just one. The score is how many events line up, not how big they are --
lifting the robot into view moves every tag far more than a tap does, and the log never
sees it (coarse_search). The protocol (film before the log starts, stop after it ends) is
used too: only lags that put the log inside the video are searched, and a stronger match
outside that is reported, not taken.
(2) Fine: take the largest event near the start (and the end) of the overlap in the video,
the matching spike in the log within +-0.3 s of where the coarse offset puts it, and
compare their ONSETS (30% of peak, interpolated) -- onsets, because a 40 Hz IMU and a
60 fps camera smear a peak differently but agree on when it began. With two events, t_wall = a + b * t_video (b - 1 = drift); with one, b = 1.

VERIFIED means two events (start and end) gave the same clock within DRIFT_MAX_PPM. One
event, or two that disagree, still writes the files, but sync.json says verified: false,
the printout says SYNC NOT VERIFIED and the exit status is 2: a wrong offset looks exactly
like a right one in aligned.csv, so it must never pass quietly.

PRECISION: about +-half a log sample (12 ms at 40 Hz) per event; the onset of a tap is
sharp in both signals but the log only samples it every 25 ms. Check the reported events
are the taps: a fall or a push also syncs correctly (it is just as simultaneous), but a
tap is what the start/end windows are meant to find.

WRITES next to the video/tags.csv (or --out):
  <stem>.aligned.csv  one row per LOG row: t_wall, t_rel, step, t_video, in_video, the IMU's
                      pelvis roll/pitch, every links.csv column interpolated to that instant as
                      cam.<link>.<field> / cam.<pair>.<angle> (blank across a detection gap),
                      cam.world_ref, and enc.<pair>.<angle>: the same joint angle from the
                      encoders (tags.yaml `pairs`), so camera and encoder sit side by side.
                        imu.pg.roll/pitch   from the logged pg, which deploy_standing EMA-filters
                                            (projgrav_alpha; at 0.35 that is ~2 samples, ~50 ms
                                            of lag): a lag in a push is the filter, not flex.
                        imu.acc.roll/pitch  (v2 logs) from the raw accelerometer, unfiltered --
                                            exact when quasi-static, but a push's linear
                                            acceleration tilts it too.
                      cam.<link>.* are blank where world_ref = 0 (floor tag never seen): those
                      angles would be in the CAMERA frame, not comparable with the IMU. Pair
                      angles never use the world and are always kept.
  <stem>.sync.json    verified, offset, drift, the events used, the log's filter settings
                      (from <log>.meta.json) and camera-vs-encoder/IMU offsets.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camlib  # noqa: E402

GRID_DT = 0.01          # s, common grid for the coarse cross-correlation
HP_WINDOW_S = 0.5       # moving-median window removed from each signal: keeps the sharp events only
MIN_EVENT_Z = 6.0       # an event must stand this many robust sigmas above the noise
EVENT_CLIP_Z = 3 * MIN_EVENT_Z  # cross-correlation input cap: one huge event must not decide the lag
EVENT_SEP_S = 1.0       # peaks closer than this are one event (a push rings for a while)
EVENT_HALF_WIDTH_S = 0.15  # an event's triangle in the coarse search: covers onset smear + ~0.1% drift
MIN_LOG_COVER = 0.9     # the protocol films the whole log; a lag putting less of it in view is suspect
ONSET_FRAC = 0.3        # onset = where the spike first reaches this fraction of its peak
MATCH_S = 0.3           # the log spike must lie this close to where the coarse offset predicts
MIN_OVERLAP_S = 3.0
DRIFT_MAX_PPM = 5000.0  # more than 0.5% means the two "end" events are not the same event
MAX_GAP_S = 0.1         # camera samples farther than this from a log row are not interpolated across
_ANGLE_FIELDS = ("roll", "pitch", "yaw")


# ---------------------------------------------------------------- reading

def read_columns(path):
    """CSV -> (header, {column: float array}); blanks and text become NaN."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    cols = {}
    for j, name in enumerate(header):
        out = np.full(len(rows), np.nan)
        for i, row in enumerate(rows):
            if j < len(row) and row[j] not in ("", None):
                try:
                    out[i] = float(row[j])
                except ValueError:
                    pass
        cols[name] = out
    return header, cols


def moving_median(x, n):
    """Centred moving median along axis 0 (edges reflected); n forced odd."""
    x = np.asarray(x, dtype=float)
    n = max(1, int(n) | 1)
    if n == 1 or len(x) < 2:
        return x.copy()
    h = n // 2
    pad = [(h, h)] + [(0, 0)] * (x.ndim - 1)
    xp = np.pad(x, pad, mode="reflect" if len(x) > h else "edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, n, axis=0)
    return np.nanmedian(win, axis=-1)


def robust_z(s):
    """(s - median) / (1.4826 * MAD): noise ~ N(0, 1), events stand out as large positives."""
    s = np.asarray(s, dtype=float)
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med)) * 1.4826
    if not np.isfinite(mad) or mad <= 1e-12:
        mad = np.nanstd(s) or 1.0
    return np.nan_to_num((s - med) / mad)


def _rate(t):
    d = np.diff(t)
    d = d[d > 0]
    return 1.0 / float(np.median(d)) if len(d) else 1.0


# ---------------------------------------------------------------- signals

_LOG_SIGNALS = (("acc", ("acc_x", "acc_y", "acc_z")), ("gyro", ("av_x", "av_y", "av_z")))


def log_motion_signals(cols):
    """[(t_wall, event signal, kind)] for every IMU signal the log has: acc (schema v2), gyro.

    Both, because neither is always best. |acc| has the sharpest onset, but a finger tap's
    acceleration spike lasts a few ms and the log keeps one sample per 25 ms loop (the
    ICM-20948's low-pass is not narrowed by hardware.py), so it is caught at a random point
    in its decay or missed. The gyro sees the body rock for ~100 ms: a blunter onset that
    is always there. synchronise_best() keeps whichever matches better.
    """
    t = cols["t_wall"]
    out = []
    for kind, names in _LOG_SIGNALS:
        if not all(c in cols for c in names):
            continue
        X = np.column_stack([cols[c] for c in names])
        ok = np.isfinite(t) & np.isfinite(X).all(axis=1)
        if ok.mean() <= 0.5:
            continue
        tk, X = t[ok], X[ok]
        hp = X - moving_median(X, round(HP_WINDOW_S * _rate(tk)))
        out.append((tk, robust_z(np.linalg.norm(hp, axis=1)), kind))
    if not out:
        raise SystemExit("log has neither acc_x/y/z (schema v2) nor av_x/y/z")
    return out


def log_motion_signal(cols, kind=None):
    """(t_wall, event signal, kind): the named kind, else the first available (acc, then gyro)."""
    for s in log_motion_signals(cols):
        if kind is None or s[2] == kind:
            return s
    raise SystemExit(f"log has no {kind} signal")


def tag_motion_signal(tag_cols, exclude_ids=()):
    """(t_video, event signal) from tags.csv: mean image speed (px/s) of the body tags."""
    frame, t = tag_cols["frame"], tag_cols["t_video"]
    tid, missing = tag_cols["tag_id"], tag_cols["missing"]
    cx, cy = tag_cols["px_cx"], tag_cols["px_cy"]
    frames, first = np.unique(frame, return_index=True)
    tf = t[first]
    pos = {f: i for i, f in enumerate(frames)}
    dt_nom = 1.0 / _rate(tf)
    total, count = np.zeros(len(frames)), np.zeros(len(frames))
    for k in np.unique(tid):
        if k in exclude_ids:          # the floor tag never moves; it only adds noise
            continue
        m = (tid == k) & (missing == 0) & np.isfinite(cx)
        f, x, y, tt = frame[m], cx[m], cy[m], t[m]
        o = np.argsort(f)
        f, x, y, tt = f[o], x[o], y[o], tt[o]
        dt = np.diff(tt)
        ok = (dt > 0) & (dt < 2.5 * dt_nom)
        sp = np.hypot(np.diff(x), np.diff(y))[ok] / dt[ok]
        for ff, v in zip(f[1:][ok], sp):
            i = pos[ff]
            total[i] += v
            count[i] += 1
    s = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return tf, robust_z(s - moving_median(s, round(HP_WINDOW_S * _rate(tf))))


def frame_diff_signal(video, fps=None):
    """(t_video, event signal) from the raw video: mean absolute frame-to-frame change.
    Cruder than tags -- the tapping hand moving in shows up too -- but needs nothing."""
    import video_pose
    cv2 = camlib.cv2_aruco()
    ts, vals, prev = [], [], None
    for _, t, img in video_pose.iter_frames(video, fps=fps):
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (320, max(1, round(320 * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32)
        vals.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))
        ts.append(t)
        prev = g
    t, s = np.array(ts), np.array(vals)
    return t, robust_z(s - moving_median(s, round(HP_WINDOW_S * _rate(t))))


# ---------------------------------------------------------------- matching

def event_onsets(t, z, min_z=MIN_EVENT_Z, sep_s=EVENT_SEP_S):
    """Onset times of the distinct events in a signal: peaks >= min_z, strongest first, each
    at least sep_s from a stronger one (one push's ringing is one event, not five)."""
    cand = np.flatnonzero(z >= min_z)
    taken = []
    for i in cand[np.argsort(-z[cand], kind="stable")]:
        if all(abs(t[i] - t[j]) >= sep_s for j in taken):
            taken.append(int(i))
    return np.array(sorted(onset_time(t, z, i) for i in taken))


def _event_train(grid, onsets, half_width=EVENT_HALF_WIDTH_S):
    """One unit triangle per event: the correlation of two trains counts the events that line up."""
    x = np.zeros(len(grid))
    for te in onsets:
        x += np.clip(1.0 - np.abs(grid - te) / half_width, 0.0, None)
    return x


def _xcorr_all_lags(v, lg):
    """(lags, c): c[k] = sum_i v[i] * lg[i + k] for every lag at which the two overlap."""
    nv, nl = len(v), len(lg)
    nfft = 1 << int(math.ceil(math.log2(nv + nl)))
    c = np.fft.irfft(np.conj(np.fft.rfft(v, nfft)) * np.fft.rfft(lg, nfft), nfft)
    lags = np.concatenate([np.arange(0, nl), np.arange(-(nv - 1), 0)])
    return lags, (np.concatenate([c[:nl], c[nfft - (nv - 1):]]) if nv > 1 else c[:nl])


def coarse_search(tv, zv, tl, zl, dt=GRID_DT, min_overlap_s=MIN_OVERLAP_S, clip_z=EVENT_CLIP_Z):
    """Cross-correlate the two recordings' events over every lag.

    -> {offset, log_cover, free_offset, free_log_cover, free_ratio}. offset (t_log - t_video)
    is the best lag that keeps at least MIN_LOG_COVER of the best achievable share of the
    log inside the video (the protocol); free_* is the best lag with no such constraint, and
    free_ratio its score over the constrained one (> 1 = a lag outside the protocol fits better).

    THE SCORE counts events, not signal energy. A correlation of raw z is decided by the
    largest product, so one big disturbance only one stream holds -- lifting the robot into
    view, the camera bumped -- outvoted both taps and put the log a minute off (review, on
    the test signals). Clipping alone is not enough either: handling is a long event, and
    its clipped area still beats two short taps. So each recording becomes a train of unit
    triangles at its event onsets, and the lag where the most events coincide wins; the
    clipped signals' correlation only breaks ties (and decides when neither stream has a
    clear event).
    """
    gv = np.arange(tv[0], tv[-1], dt)
    gl = np.arange(tl[0], tl[-1], dt)
    v = np.clip(np.interp(gv, tv, zv), 0, clip_z)
    lg = np.clip(np.interp(gl, tl, zl), 0, clip_z)
    nv, nl = len(v), len(lg)
    lags, c_sig = _xcorr_all_lags(v, lg)
    _, c_ev = _xcorr_all_lags(_event_train(gv, event_onsets(tv, zv)), _event_train(gl, event_onsets(tl, zl)))
    # One coinciding event scores ~(2/3) * half_width / dt; the tie-break is worth less than that.
    one_event = (2.0 / 3.0) * EVENT_HALF_WIDTH_S / dt
    top = np.max(np.abs(c_sig)) if len(c_sig) else 0.0
    cc = c_ev + (0.5 * one_event * c_sig / top if top > 0 else 0.0)
    overlap = np.minimum(nv, nl - lags) - np.maximum(0, -lags)
    need = min(round(min_overlap_s / dt), max(1, min(nv, nl) // 2))
    free = np.where(overlap >= need, cc, -np.inf)
    cover = overlap / max(nl, 1)                    # share of the log inside the video
    best_cover = min(nv, nl) / max(nl, 1)           # < 1 only if the video is shorter than the log
    held = np.where(cover >= MIN_LOG_COVER * best_cover - 1e-9, free, -np.inf)
    k, kf = int(np.argmax(held)), int(np.argmax(free))
    if not np.isfinite(held[k]):
        k = kf
    ratio = float(free[kf] / held[k]) if held[k] > 0 else (math.inf if free[kf] > 0 else 1.0)

    def offset(i):
        return float(gl[0] - gv[0] + int(lags[i]) * dt)
    return {"offset": offset(k), "log_cover": float(cover[k]), "free_offset": offset(kf),
            "free_log_cover": float(cover[kf]), "free_ratio": ratio}


def coarse_offset(tv, zv, tl, zl, dt=GRID_DT, min_overlap_s=MIN_OVERLAP_S):
    """offset (t_log - t_video) maximising the cross-correlation of the two event signals."""
    return coarse_search(tv, zv, tl, zl, dt, min_overlap_s)["offset"]


def onset_time(t, z, ipk, frac=ONSET_FRAC):
    """Where the spike peaking at index ipk first rose through frac * peak (interpolated)."""
    level = frac * z[ipk]
    k = ipk
    while k > 0 and z[k - 1] >= level:
        k -= 1
    if k == 0:
        return float(t[0])
    z0, z1 = z[k - 1], z[k]
    f = (level - z0) / (z1 - z0) if z1 != z0 else 0.0
    return float(t[k - 1] + f * (t[k] - t[k - 1]))


def find_events(tv, zv, tl, zl, offset, window_s=20.0, min_z=MIN_EVENT_Z):
    ov0, ov1 = max(tv[0], tl[0] - offset), min(tv[-1], tl[-1] - offset)
    if ov1 <= ov0:
        return []
    W = min(window_s, (ov1 - ov0) / 2)
    events = []
    for name, (a, b) in (("start", (ov0, ov0 + W)), ("end", (ov1 - W, ov1))):
        iv = np.flatnonzero((tv >= a) & (tv <= b))
        if not len(iv):
            continue
        i = int(iv[np.argmax(zv[iv])])
        if zv[i] < min_z:
            continue
        il = np.flatnonzero(np.abs(tl - (tv[i] + offset)) <= MATCH_S)
        if not len(il):
            continue
        j = int(il[np.argmax(zl[il])])
        if zl[j] < min_z:
            continue
        on_v, on_l = onset_time(tv, zv, i), onset_time(tl, zl, j)
        if events and abs(on_v - events[0]["t_video"]) < 1.0:
            continue                       # short overlap: start and end found the same event
        events.append({"name": name, "t_video": on_v, "t_wall": on_l, "offset": on_l - on_v,
                       "z_video": round(float(zv[i]), 1), "z_log": round(float(zl[j]), 1)})
    return events


def synchronise(tv, zv, tl, zl, window_s=20.0, offset_guess=None):
    """-> dict(a, b, drift_ppm, verified, events, coarse_offset, log_cover, warnings);
    t_wall = a + b * t_video. verified = the start and end events agree on one clock."""
    warn = []
    if offset_guess is None:
        cs = coarse_search(tv, zv, tl, zl)
        coarse, cover = cs["offset"], cs["log_cover"]
        if cs["free_ratio"] > 1.05 and abs(cs["free_offset"] - coarse) > 1.0:
            warn.append(f"a stronger motion match at offset {cs['free_offset']:.2f} s puts only "
                        f"{100 * cs['free_log_cover']:.0f}% of the log inside the video, against the "
                        f"protocol (film before the log starts, stop after it ends); used {coarse:.2f} s. "
                        f"If filming started late, rerun with --offset-guess {cs['free_offset']:.2f}")
    else:
        coarse = float(offset_guess)
        lo, hi = max(tv[0], tl[0] - coarse), min(tv[-1], tl[-1] - coarse)
        cover = max(0.0, hi - lo) / max(tl[-1] - tl[0], 1e-9)
    if cover < MIN_LOG_COVER:
        warn.append(f"only {100 * cover:.0f}% of the log lies inside the video: start filming "
                    "before the run and stop after it.")
    events = find_events(tv, zv, tl, zl, coarse, window_s)
    a, b, drift, verified = coarse, 1.0, None, False
    if not events:
        warn.append("SYNC NOT VERIFIED: no clear tap in both recordings; using the coarse "
                    "cross-correlation only (+-1 grid step at best, and unchecked). Tap harder, "
                    "in view, with the robot still.")
    else:
        a = events[0]["offset"]
        if len(events) == 2 and events[1]["t_video"] - events[0]["t_video"] > 5.0:
            e0, e1 = events
            b_try = (e1["t_wall"] - e0["t_wall"]) / (e1["t_video"] - e0["t_video"])
            drift = (b_try - 1.0) * 1e6
            if abs(drift) <= DRIFT_MAX_PPM:
                b = b_try
                a = e0["t_wall"] - b * e0["t_video"]
                verified = True
            else:
                warn.append(f"SYNC NOT VERIFIED: the start and end events disagree by {drift:.0f} ppm "
                            "of drift, so at least one is not the same event in both recordings. "
                            "Using the start event alone; check its time against when you tapped.")
        else:
            warn.append("SYNC NOT VERIFIED: only one event found, so nothing confirms the offset "
                        "(and no drift check). Check the event time against when you tapped; tap "
                        "at the end too.")
    return {"a": a, "b": b, "drift_ppm": drift, "verified": verified, "coarse_offset": coarse,
            "log_cover": cover, "events": events, "warnings": warn}


def synchronise_best(tv, zv, log_signals, window_s=20.0, offset_guess=None):
    """synchronise() against each log signal (acc, gyro) and keep the one that matched best:
    verified first, then more events, then the stronger of the weakest matched log events --
    counted only up to EVENT_CLIP_Z, so when both are clearly strong the first listed (acc,
    the sharper onset) is kept. -> (sync dict, kind, {kind: summary of each try})."""
    order = [kind for _, _, kind in log_signals]
    tries = {kind: synchronise(tv, zv, tl, zl, window_s, offset_guess) for tl, zl, kind in log_signals}

    def weakest(k):
        return min((e["z_log"] for e in tries[k]["events"]), default=0.0)

    def key(k):
        s = tries[k]
        return (s["verified"], len(s["events"]), min(weakest(k), EVENT_CLIP_Z), -order.index(k))
    kind = max(tries, key=key)
    summary = {k: {"verified": s["verified"], "events": len(s["events"]),
                   "min_z_log": weakest(k), "t_wall_at_video_0": s["a"]} for k, s in tries.items()}
    return tries[kind], kind, summary


# ---------------------------------------------------------------- aligned output

def _interp_masked(tq, t, y, max_gap, angle=False):
    """Linear interpolation of y(t) at tq, blank outside the samples and wherever the two
    bracketing samples are more than max_gap apart (a detection dropout is not data)."""
    ok = np.isfinite(t) & np.isfinite(y)
    out = np.full(len(tq), np.nan)
    if ok.sum() < 2:
        return out
    ts, ys = t[ok], y[ok]
    if angle:
        ys = np.degrees(np.unwrap(np.radians(ys)))      # yaw near +-180 must not interpolate through 0
    out = np.interp(tq, ts, ys, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(ts, tq), 1, len(ts) - 1)
    out[(ts[j] - ts[j - 1]) > max_gap] = np.nan
    if angle:
        out = (out + 180.0) % 360.0 - 180.0
    return out


def encoder_angles(log_cols, cfg, servo_map_path=None):
    """{f"{pair}.{angle}": degrees per log row} from u_meas through the deploy servo map."""
    try:
        sys.path.insert(0, str(camlib.ROOT / "scripts" / "deploy"))
        from sim_real_map import SimRealMap
        smap = SimRealMap(servo_map_path) if servo_map_path else SimRealMap()
    except Exception as e:                  # the log is still worth aligning without them
        print(f"(no encoder columns: servo map unavailable: {e})")
        return {}
    idx = {j.dof: k for k, j in enumerate(smap.joints)}
    q = {}
    for dof, k in idx.items():
        u = log_cols.get(f"u_meas.{dof}")
        if u is not None:
            q[dof] = np.degrees(smap.signs[k] * (u - smap.centers[k]) / smap.units_per_rad)
    out = {}
    names = {"x": "roll", "y": "pitch", "z": "yaw"}
    for pr in cfg.pairs:
        for ax, terms in pr.encoder.items():
            if all(d in q for d, _ in terms):
                out[f"{pr.name}.{names.get(ax, ax)}"] = sum(s * q[d] for d, s in terms)
    return out


def _tilt_from_pg(gx, gy, gz):
    """Pelvis roll/pitch (deg) from projected gravity pg = R^T (0, 0, -1), Z-Y-X convention."""
    return {"pitch": np.degrees(np.arctan2(gx, np.hypot(gy, gz))),
            "roll": np.degrees(np.arctan2(-gy, -gz))}


def imu_angles(log_cols):
    """{"pg.roll", "pg.pitch"[, "acc.roll", "acc.pitch"]} in degrees.

    pg is what the policy saw: EMA-filtered on the Pi (projgrav_alpha), so it lags a push by
    a few samples. acc (schema v2) is the raw accelerometer in g, which at rest reads +1 g UP
    = -pg; normalised and negated it is the same tilt with no filter -- but under a push its
    linear acceleration tilts it too, so the two together separate lag from flex.
    """
    out = {}
    if all(c in log_cols for c in ("pg_x", "pg_y", "pg_z")):
        out.update({f"pg.{k}": v for k, v in
                    _tilt_from_pg(log_cols["pg_x"], log_cols["pg_y"], log_cols["pg_z"]).items()})
    if all(c in log_cols for c in ("acc_x", "acc_y", "acc_z")):
        a = np.column_stack([log_cols["acc_x"], log_cols["acc_y"], log_cols["acc_z"]])
        n = np.linalg.norm(a, axis=1, keepdims=True)
        g = -a / np.where(n > 1e-9, n, np.nan)
        out.update({f"acc.{k}": v for k, v in _tilt_from_pg(g[:, 0], g[:, 1], g[:, 2]).items()})
    return out


def log_filters(log_path):
    """The Pi's IMU filter settings from <log>.meta.json (None if the sidecar is missing)."""
    meta = Path(log_path).with_suffix(".meta.json")
    try:
        m = json.loads(meta.read_text())
    except (OSError, ValueError):
        return None
    return {k: m.get(k) for k in ("projgrav_alpha", "angvel_alpha", "hz", "schema_version")}


def _offset_stats(cam, ref):
    m = np.isfinite(cam) & np.isfinite(ref)
    if m.sum() < 5:
        return None
    d = cam[m] - ref[m]
    med = float(np.median(d))
    st = {"n": int(m.sum()), "median_offset_deg": round(med, 3),
          "std_after_offset_deg": round(float(np.std(d - med)), 3)}
    if np.std(cam[m]) > 0.05 and np.std(ref[m]) > 0.05:
        st["corr"] = round(float(np.corrcoef(cam[m], ref[m])[0, 1]), 3)
    return st


def align(log_cols, link_header, link_cols, sync, cfg, servo_map_path=None):
    """(columns of the aligned table, one row per log row; camera-vs-reference stats; warnings)."""
    tw = log_cols["t_wall"]
    tv = (tw - sync["a"]) / sync["b"]
    out = {"t_wall": tw, "t_rel": log_cols.get("t_rel"), "step": log_cols.get("step"), "t_video": tv}
    stats, warn = {}, []
    if link_cols is not None:
        t_cam = link_cols["t_video"]
        out["in_video"] = ((tv >= np.nanmin(t_cam)) & (tv <= np.nanmax(t_cam))).astype(float)
    imu = imu_angles(log_cols)
    for k, v in imu.items():
        out[f"imu.{k}"] = v
    if link_cols is not None:
        gap = max(MAX_GAP_S, 2.5 / _rate(t_cam))
        # world_ref = 0: the floor tag was not seen, so a link's pose is in the CAMERA frame.
        # Next to the gravity-referenced IMU it would read as a tilt the robot never had, so
        # those link columns are blanked; pair angles (child in parent) never use the world.
        world_ref = link_cols.get("world_ref")
        no_world = (world_ref == 0) if world_ref is not None else np.zeros(len(t_cam), bool)
        if world_ref is not None:
            out["cam.world_ref"] = _interp_masked(tv, t_cam, world_ref.astype(float), gap)
        if no_world.any():
            warn.append(f"floor tag missing in {100 * no_world.mean():.0f}% of video frames: link "
                        "poses there are in the camera frame, so cam.<link>.* are blank and no "
                        "camera-vs-IMU stats use them (pair angles are kept)")
        links = set(cfg.links)
        for c in link_header:
            if c in ("frame", "t_video", "world_ref") or c.endswith((".tag", ".reproj_px")):
                continue
            y = link_cols[c]
            if c.rsplit(".", 1)[0] in links and no_world.any():
                y = np.where(no_world, np.nan, y)
            out[f"cam.{c}"] = _interp_masked(tv, t_cam, y, gap,
                                             angle=c.rsplit(".", 1)[-1] in _ANGLE_FIELDS)
    enc = encoder_angles(log_cols, cfg, servo_map_path)
    for k, v in enc.items():
        out[f"enc.{k}"] = v
        if f"cam.{k}" in out:
            stats[k] = _offset_stats(out[f"cam.{k}"], v)
    for k, v in imu.items():               # k = "pg.pitch" etc.
        src, ang = k.split(".")
        st = _offset_stats(out[f"cam.pelvis.{ang}"], v) if f"cam.pelvis.{ang}" in out else None
        if st:                             # none at all without a floor tag (blanked above)
            stats[f"pelvis.{ang} (cam - imu.{src})"] = st
    return {k: v for k, v in out.items() if v is not None}, stats, warn


def write_columns(path, cols):
    names = list(cols)
    arrs = [cols[n] for n in names]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in range(len(arrs[0])):
            row = []
            for n, a in zip(names, arrs):
                v = a[i]
                if not np.isfinite(v):
                    row.append("")
                elif n == "t_wall":
                    row.append(f"{v:.6f}")       # epoch seconds: keep the microseconds
                elif n in ("t_video", "t_rel"):
                    row.append(f"{v:.6f}")
                elif n in ("step", "in_video", "cam.world_ref"):
                    row.append(int(v))
                else:
                    row.append(f"{v:.6g}")
            w.writerow(row)


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("camera", help="<stem>.tags.csv from video_pose.py, or the video itself")
    ap.add_argument("log", help="frame log CSV (scripts/deploy/frame_log.py)")
    ap.add_argument("--tags", default=str(camlib.DEFAULT_TAGS))
    ap.add_argument("--servo-map", default=None, help="default: config/joint_servo_map.yaml")
    ap.add_argument("--window", type=float, default=20.0, help="s at each end of the overlap searched for a tap")
    ap.add_argument("--offset-guess", type=float, default=None,
                    help="skip the cross-correlation: t_wall - t_video, seconds")
    ap.add_argument("--fps", type=float, default=None, help="video input only: override its timestamps")
    ap.add_argument("--out", default=None, help="aligned CSV path; default <stem>.aligned.csv next to the camera file")
    a = ap.parse_args(argv)

    cfg = camlib.load_tag_config(a.tags)
    cam = Path(a.camera)
    _, log_cols = read_columns(a.log)
    log_signals = log_motion_signals(log_cols)

    if cam.name.endswith(".tags.csv"):
        stem = cam.name[: -len(".tags.csv")]
        _, tag_cols = read_columns(cam)
        tv, zv = tag_motion_signal(tag_cols, [t for t, tg in cfg.tags.items() if tg.link == "world"])
        vid_kind = "tag motion"
    else:
        stem = cam.stem if cam.is_file() else cam.name
        tv, zv = frame_diff_signal(cam, a.fps)
        vid_kind = "frame difference"
    links_path = cam.parent / f"{stem}.links.csv"

    s, log_kind, tried = synchronise_best(tv, zv, log_signals, a.window, a.offset_guess)
    link_header = link_cols = None
    if links_path.exists():
        link_header, link_cols = read_columns(links_path)
    else:
        s["warnings"].append(f"{links_path.name} not found: run video_pose.py first for camera columns")
    cols, stats, align_warn = align(log_cols, link_header, link_cols, s, cfg, a.servo_map)
    s["warnings"] += align_warn
    filters = log_filters(a.log)

    out = Path(a.out) if a.out else cam.parent / f"{stem}.aligned.csv"
    write_columns(out, cols)
    t_rel0 = float(np.nanmin(log_cols["t_rel"])) if "t_rel" in log_cols else 0.0
    t_wall0 = float(np.nanmin(log_cols["t_wall"]))
    report = {"camera": str(cam), "log": str(a.log), "verified": s["verified"],
              "video_signal": vid_kind, "log_signal": log_kind, "log_signals_tried": tried,
              "t_wall_at_video_0": s["a"], "scale_b": s["b"], "drift_ppm": s["drift_ppm"],
              "log_t_rel_at_video_0": s["a"] - t_wall0 + t_rel0, "log_in_video_frac": s["log_cover"],
              "coarse_offset": s["coarse_offset"], "events": s["events"], "warnings": s["warnings"],
              "log_filters": filters, "camera_vs_reference": stats, "aligned_csv": str(out)}
    json_path = out.with_name(f"{stem}.sync.json")
    json_path.write_text(json.dumps(report, indent=2, default=float))

    print(f"video signal: {vid_kind}; log signal: {'|acc|' if log_kind == 'acc' else 'gyro'}"
          f" (tried {', '.join(tried)})")
    for e in s["events"]:
        print(f"  {e['name']:5s} event: video t={e['t_video']:8.3f} s (z {e['z_video']}) <-> log t_rel="
              f"{e['t_wall'] - t_wall0 + t_rel0:8.3f} s (z {e['z_log']})  offset {e['offset']:.4f} s")
    print(f"video t=0 is log t_rel = {report['log_t_rel_at_video_0']:.3f} s  (t_wall {s['a']:.3f})")
    if s["drift_ppm"] is not None:
        e0, e1 = s["events"][:2]
        print(f"drift check: {s['drift_ppm']:+.0f} ppm = {1000 * (e1['offset'] - e0['offset']):+.1f} ms "
              f"over {e1['t_video'] - e0['t_video']:.1f} s" + ("  (applied)" if s["b"] != 1.0 else ""))
    for w in s["warnings"]:
        print(f"WARNING: {w}")
    if not s["verified"]:
        print("\n*** SYNC NOT VERIFIED: do not trust the camera-vs-log timing until the event "
              "times above match when you tapped (see the warnings). ***")
    if stats:
        print(f"\n{'camera vs reference':<31} {'n':>6} {'offset deg':>11} {'std deg':>8} {'corr':>6}")
        for k, st in stats.items():
            if st:
                print(f"{k:<31} {st['n']:>6} {st['median_offset_deg']:>11.2f} {st['std_after_offset_deg']:>8.2f} "
                      f"{st.get('corr', float('nan')):>6.2f}")
    print(f"\nwrote {out}\n      {json_path}")
    return report


if __name__ == "__main__":
    # Exit 2 when not verified, so a wrapper or a shell `&&` stops instead of using the files.
    sys.exit(0 if main()["verified"] else 2)
