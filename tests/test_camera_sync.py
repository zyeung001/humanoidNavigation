"""Camera <-> frame-log sync (scripts/camera/sync.py) on synthetic recordings with a KNOWN offset.

The phone and the Pi share no clock; the only link is a tap both see. These build a log
(40 Hz, schema v2 with acc_x/y/z, or v1 with only the gyro) and a tags.csv (60 fps) of
the same taps, offset and optionally drifting by known amounts, and check sync recovers
the mapping to within half a log sample.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path("scripts/camera").resolve()))
import camlib  # noqa: E402
import sync  # noqa: E402

T_WALL0 = 1792907000.0          # log start (unix s)
VIDEO_START = T_WALL0 - 2.0     # the phone started filming 2 s before the log
TAPS = (T_WALL0 + 5.31, T_WALL0 + 63.77)
LOG_S, VIDEO_S = 70.0, 75.0
CFG = camlib.load_tag_config()


def tap_shape(dt, tau):
    return np.where(dt >= 0, np.exp(-np.clip(dt, 0, None) / tau), 0.0)


def make_log(v2=True, seed=0, knee_deg=10.0):
    rng = np.random.default_rng(seed)
    t = T_WALL0 + np.arange(0, LOG_S, 1 / 40) + rng.normal(0, 0.0005, int(LOG_S * 40))
    n = len(t)
    sway = 0.03 * np.sin(2 * np.pi * 0.4 * (t - T_WALL0))
    av = np.column_stack([rng.normal(0, 0.004, n), sway + rng.normal(0, 0.004, n), rng.normal(0, 0.004, n)])
    acc = np.column_stack([rng.normal(0, 0.006, n), rng.normal(0, 0.006, n), 1 + rng.normal(0, 0.006, n)])
    for tap in TAPS:
        av[:, 1] += 0.6 * tap_shape(t - tap, 0.06)
        acc[:, 0] += 0.9 * tap_shape(t - tap, 0.02)
    cols = {"t_wall": t, "t_rel": t - T_WALL0 + 6.0, "step": np.arange(n),
            "pg_x": np.full(n, math.sin(math.radians(3))), "pg_y": np.zeros(n),
            "pg_z": np.full(n, -math.cos(math.radians(3))),
            "av_x": av[:, 0], "av_y": av[:, 1], "av_z": av[:, 2]}
    if v2:
        cols.update(acc_x=acc[:, 0], acc_y=acc[:, 1], acc_z=acc[:, 2])
    # R_knee: servo 7, sign -1, centre 510, 195 units/rad (config/joint_servo_map.yaml)
    cols["u_meas.R_knee"] = np.full(n, round(510 - math.radians(knee_deg) * 195))
    return cols


def video_clock(t_wall, drift_ppm=0.0):
    """t_video for a wall time, for a phone whose clock runs drift_ppm fast."""
    return (t_wall - VIDEO_START) * (1 + drift_ppm * 1e-6)


def make_tags(drift_ppm=0.0, seed=1, gap=None):
    rng = np.random.default_rng(seed)
    tv = np.arange(0, VIDEO_S, 1 / 60)
    tw = VIDEO_START + tv / (1 + drift_ppm * 1e-6)
    rows, links = [], []
    for f, (t_v, t_w) in enumerate(zip(tv, tw)):
        # A tap knocks the body 6 px over ~40 ms, then it creeps back over ~1 s.
        push = sum(6.0 * (1 - tap_shape(t_w - tap, 0.04)) * tap_shape(t_w - tap, 1.0) for tap in TAPS)
        for tid, x0 in ((1, 800.0), (7, 820.0), (0, 900.0)):
            missing = gap is not None and gap[0] <= t_v <= gap[1] and tid != 0
            moves = 0.0 if tid == 0 else push
            rows.append({"frame": f, "t_video": t_v, "tag_id": tid, "missing": int(missing),
                         "px_cx": "" if missing else x0 + moves + rng.normal(0, 0.15),
                         "px_cy": "" if missing else 400 + rng.normal(0, 0.15)})
        in_gap = gap is not None and gap[0] <= t_v <= gap[1]
        links.append({"frame": f, "t_video": t_v, "world_ref": 1,
                      "pelvis.pitch": "" if in_gap else 5 * math.sin(2 * math.pi * 0.5 * (t_w - T_WALL0)),
                      "pelvis.roll": "" if in_gap else 0.5,
                      "R_knee.pitch": "" if in_gap else 12.0})
    return rows, links


def _cols(rows):
    keys = list(rows[0])
    return {k: np.array([np.nan if r[k] == "" else float(r[k]) for r in rows]) for k in keys}


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _signals(log, tags):
    tl, zl, kind = sync.log_motion_signal(log)
    tv, zv = sync.tag_motion_signal(_cols(tags), exclude_ids=[0])
    return tv, zv, tl, zl, kind


@pytest.mark.parametrize("v2,kind", [(True, "acc"), (False, "gyro")])
def test_known_offset_is_recovered_from_the_taps(v2, kind):
    tags, _ = make_tags()
    tv, zv, tl, zl, got_kind = _signals(make_log(v2=v2), tags)
    assert got_kind == kind
    s = sync.synchronise(tv, zv, tl, zl)
    assert [e["name"] for e in s["events"]] == ["start", "end"]
    assert s["a"] == pytest.approx(VIDEO_START, abs=0.015)
    assert abs(s["drift_ppm"]) < 600            # no drift injected; +-12 ms over 58 s
    for e, tap in zip(s["events"], TAPS):        # the events found ARE the taps
        assert e["t_wall"] == pytest.approx(tap, abs=0.03)


def test_coarse_cross_correlation_alone_lands_within_a_few_grid_steps():
    tags, _ = make_tags()
    tv, zv, tl, zl, _ = _signals(make_log(), tags)
    assert sync.coarse_offset(tv, zv, tl, zl) == pytest.approx(VIDEO_START, abs=0.05)


def test_clock_drift_is_measured_and_applied():
    drift = 1500.0                               # phone clock 0.15% fast: 87 ms over the run
    tags, _ = make_tags(drift_ppm=drift)
    tv, zv, tl, zl, _ = _signals(make_log(), tags)
    s = sync.synchronise(tv, zv, tl, zl)
    assert s["drift_ppm"] == pytest.approx(-drift, abs=600)
    for tap in TAPS:                             # both taps map back to their wall time
        assert s["a"] + s["b"] * video_clock(tap, drift) == pytest.approx(tap, abs=0.02)


def test_cli_writes_an_aligned_csv_keyed_on_log_t_wall(tmp_path):
    log = make_log(v2=True)
    log_rows = [{k: v[i] for k, v in log.items()} for i in range(len(log["t_wall"]))]
    with open(tmp_path / "run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log))
        w.writeheader()
        for r in log_rows:
            w.writerow({k: (f"{v:.6f}" if k == "t_wall" else v) for k, v in r.items()})
    tags, links = make_tags(gap=(30.0, 31.0))
    _write(tmp_path / "clip.tags.csv", tags)
    _write(tmp_path / "clip.links.csv", links)

    rep = sync.main([str(tmp_path / "clip.tags.csv"), str(tmp_path / "run.csv")])
    assert rep["log_signal"] == "acc" and rep["t_wall_at_video_0"] == pytest.approx(VIDEO_START, abs=0.015)
    assert rep["log_t_rel_at_video_0"] == pytest.approx(4.0, abs=0.015)
    out = list(csv.DictReader(open(tmp_path / "clip.aligned.csv")))
    assert len(out) == len(log_rows)
    t_wall = np.array([float(r["t_wall"]) for r in out])
    assert t_wall == pytest.approx(log["t_wall"], abs=2e-6)
    pitch = np.array([float(r["cam.pelvis.pitch"]) if r["cam.pelvis.pitch"] else np.nan for r in out])
    want = 5 * np.sin(2 * np.pi * 0.5 * (t_wall - T_WALL0))
    tv = t_wall - VIDEO_START
    gap = (tv >= 29.99) & (tv <= 31.01)
    assert np.isnan(pitch[gap]).all()            # a detection dropout is blank, not interpolated
    assert np.nanmax(np.abs(pitch[~gap] - want[~gap])) < 0.5
    knee_enc = np.array([float(r["enc.R_knee.pitch"]) for r in out])
    assert knee_enc == pytest.approx(10.0, abs=0.3)
    assert np.array([float(r["imu.pitch"]) for r in out]) == pytest.approx(3.0, abs=1e-3)
    js = json.loads((tmp_path / "clip.sync.json").read_text())
    knee = js["camera_vs_reference"]["R_knee.pitch"]
    assert knee["median_offset_deg"] == pytest.approx(2.0, abs=0.3)   # cam 12 vs encoder 10
    assert js["camera_vs_reference"]["pelvis.pitch (cam - imu)"]["n"] > 1000


def test_v1_log_cli_falls_back_to_the_gyro(tmp_path):
    log = make_log(v2=False)
    with open(tmp_path / "v1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(log))
        for i in range(len(log["t_wall"])):
            w.writerow([f"{log[k][i]:.6f}" for k in log])
    tags, _ = make_tags()
    _write(tmp_path / "v.tags.csv", tags)
    rep = sync.main([str(tmp_path / "v.tags.csv"), str(tmp_path / "v1.csv")])
    assert rep["log_signal"] == "gyro"
    assert rep["t_wall_at_video_0"] == pytest.approx(VIDEO_START, abs=0.02)
    assert any("links.csv not found" in w for w in rep["warnings"])
