"""log_pose.py / run_report.py on SYNTHETIC frame logs built here (nothing under logs/ is read).

The physical conventions they check come from outside the tools: + pitch = forward (diagnose_run),
and + knee = weight toward the heel with the pelvis pitching back (config/stances/knee2.yaml,
posture.yaml's self_tilt). A rigid robot whose IMU agrees with its encoders must show no flex.
"""

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "debug" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


log_pose = _load("log_pose")
run_report = _load("run_report")
sys.path.insert(0, str(ROOT / "scripts" / "deploy"))
from sim_real_map import SimRealMap  # noqa: E402

SMAP = SimRealMap()
DOFS = [j.dof for j in SMAP.joints]
DT = 0.025

# Schema 1 exactly as the 10/25 robot logs have it, hard-coded rather than taken from
# frame_log.build_columns(): once frame_log is schema 2 that would make every "v1" log here
# v2-shaped and the v1 path (no acc columns, FSR columns mid-row) would lose its coverage.
V1_SCALARS = ["t_wall", "t_rel", "step", "loop_ms", "busy_ms", "pg_x", "pg_y", "pg_z", "av_x", "av_y", "av_z",
              "upright_cos", "rej_total", "fsr_v0", "fsr_v1", "fsr_age_ms"]
V1_COLUMNS = V1_SCALARS + [f"{b}.{d}" for b in ("u_meas", "u_cmd", "q_cmd", "jvel_f") for d in DOFS]


def _pg(pitch_deg=0.0, roll_deg=0.0):
    """Projected gravity for a pelvis pitched/rolled by these angles (+roll = robot's left)."""
    p, r = np.radians(pitch_deg), np.radians(roll_deg)
    v = np.array([np.tan(p), np.tan(r), -1.0])
    return v / np.linalg.norm(v)


def write_log(tmp_path, n=80, q_meas=None, q_cmd=None, pg=None, upright=None, extra=None,
              meta=True, name="20261025_120000_standing"):
    """A schema-v1 frame log (plus any `extra` columns) the way deploy_standing writes one.
    pg is one vector for every row, or a function of the row index."""
    q_meas = np.zeros(17) if q_meas is None else np.asarray(q_meas, float)
    q_cmd = q_meas if q_cmd is None else np.asarray(q_cmd, float)
    pg_at = pg if callable(pg) else (lambda i, v=(_pg() if pg is None else np.asarray(pg, float)): v)
    extra = extra or {}
    assert not set(extra) & set(V1_COLUMNS), "extra columns must be new ones"
    cols = V1_COLUMNS + list(extra)
    path = tmp_path / f"{name}.csv"
    u_meas = SMAP.rad_to_units(q_meas)
    u_cmd = SMAP.rad_to_units(q_cmd)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(n):
            pg = pg_at(i)
            row = {"t_wall": 1.8e9 + i * DT, "t_rel": i * DT, "step": i, "loop_ms": 25.0, "busy_ms": 21.0,
                   "pg_x": pg[0], "pg_y": pg[1], "pg_z": pg[2], "av_x": 0.0, "av_y": 0.0, "av_z": 0.0,
                   "upright_cos": -pg[2] if upright is None else upright(i), "rej_total": 0,
                   "fsr_v0": "", "fsr_v1": "", "fsr_age_ms": ""}
            for k, d in enumerate(DOFS):
                row[f"u_meas.{d}"] = int(u_meas[k])
                row[f"u_cmd.{d}"] = int(u_cmd[k])
                row[f"q_cmd.{d}"] = float(q_cmd[k])
                row[f"jvel_f.{d}"] = 0.0
            for c, fn in extra.items():
                row[c] = fn(i)
            w.writerow({c: row.get(c, "") for c in cols})
    if meta:
        path.with_suffix(".meta.json").write_text(json.dumps(
            {"schema_version": 1, "dofs": DOFS, "started_iso": "2026-10-25T12:00:00",
             "started_unix": 1.8e9, "projgrav_alpha": 0.35, "rows_dropped": 0}))
    return path


def _q(**deg):
    q = np.zeros(17)
    for k, v in deg.items():
        q[DOFS.index(k)] = np.radians(v)
    return q


def test_a_still_upright_robot_has_no_flex_and_its_com_inside_both_feet(tmp_path):
    cols, table, s, a = log_pose.analyze(write_log(tmp_path))
    assert not s["fell"]
    assert "acc_pitch_deg" not in cols and s["optional_columns_used"] == []   # a v1 log has no accelerometer
    assert abs(s["flex_pitch_deg"]["median"]) < 0.2
    assert abs(s["flex_roll_deg"]["median"]) < 1.0          # the MJCF soles sit ~0.75 deg rolled
    assert (a["support_margin_mm"] > 0).all() and (a["stance"] == "both").all()
    assert set(a["anchor"]) <= {"R", "L"} and (a["quiet"] == 1).all()
    assert s["model"]["foot_len_mm"] == pytest.approx(160.0)  # a 10/25 log -> the 16 cm sole plates
    for side in ("R", "L"):
        assert all(a[f"{side}_{e}_mm"].min() > 0 for e in ("toe", "heel")), side
        # the COM sits between the feet: past each foot's INNER edge, well inside its outer one
        assert a[f"{side}_inner_mm"].max() < 0 < a[f"{side}_outer_mm"].min(), side
    # both feet down: each foot's closest edge is a real one (toe/heel/outer), never the inner
    # edge the COM is always past when it sits between the feet
    fm = s["foot_edge_min_while_supporting"]
    assert set(fm) == {"R", "L"} and all(v["edge"] != "inner" and v["mm"] > 0 for v in fm.values())


def test_bending_the_knees_pitches_the_pelvis_back_and_moves_the_com_toward_the_heels(tmp_path):
    (tmp_path / "straight").mkdir()
    straight = log_pose.analyze(write_log(tmp_path / "straight"))[2]
    bent = log_pose.analyze(write_log(tmp_path, q_meas=_q(R_knee=5, L_knee=5), pg=_pg(-5.0)))[2]
    assert bent["enc_pitch_deg"]["median"] == pytest.approx(-5.0, abs=0.5)
    # a rigid robot: the IMU reads exactly what the encoders imply -> no hidden flex
    assert abs(bent["flex_pitch_deg"]["median"]) < 0.5
    assert bent["com_x_mm"]["median"] < straight["com_x_mm"]["median"] - 10.0
    assert bent["com_flat_x_mm"]["median"] < straight["com_flat_x_mm"]["median"] - 10.0


def test_tracking_error_shows_up_as_com_error_with_the_right_sign(tmp_path):
    # told to stand straight, the knees sagged 5 deg: the commanded COM is AHEAD of the real one
    _, _, s, a = log_pose.analyze(write_log(tmp_path, q_meas=_q(R_knee=5, L_knee=5), q_cmd=_q()))
    assert s["com_err_x_mm"]["median"] > 10.0
    assert abs(s["com_err_y_mm"]["median"]) < 3.0


def test_imu_pitch_the_encoders_cannot_see_is_flex_and_moves_the_com_toward_the_toes(tmp_path):
    _, _, s, a = log_pose.analyze(write_log(tmp_path, pg=_pg(8.0)))
    assert s["flex_pitch_deg"]["median"] == pytest.approx(8.0, abs=0.3)
    assert s["pitch_missed_by_encoders"] is None or s["pitch_missed_by_encoders"] >= 0
    assert s["com_x_mm"]["median"] > 40.0                      # ~0.4 m COM height x sin 8 deg
    assert (a["support_nearest"] == "forward").all()
    assert a["R_toe_mm"].max() < a["R_heel_mm"].min()


def test_a_tip_ends_the_standing_window_and_is_reported_with_its_direction(tmp_path):
    path = write_log(tmp_path, n=80, pg=_pg(70.0), upright=lambda i: 0.9 if i < 50 else 0.3)
    _, _, s, a = log_pose.analyze(path)
    assert s["fell"] and s["fall"]["direction"] == "forward"
    assert s["fall"]["t_rel"] == pytest.approx(50 * DT)
    assert a["fallen"][49] == 0 and a["fallen"][50] == 1


def test_v2_columns_are_used_when_present_and_unknown_columns_are_ignored(tmp_path):
    extra = {"acc_x": lambda i: 0.0, "acc_y": lambda i: 0.0, "acc_z": lambda i: 1.0,   # specific force: +1 g up at rest
             "posture_lean": lambda i: 0.5, "posture_zero": lambda i: 0.1, "posture_corr": lambda i: 1.0,
             "fsr_v2": lambda i: 1.2, "something_new": lambda i: 7}
    for d in DOFS:
        extra[f"load_raw.{d}"] = lambda i: 0x400 | 120         # sign bit set: -12.0 %
        extra[f"temp_raw.{d}"] = lambda i: 41
        extra[f"volt_raw.{d}"] = lambda i: 74
        extra[f"spd_raw.{d}"] = lambda i: 0
        extra[f"q_prior.{d}"] = lambda i: 0.01
    path = write_log(tmp_path, extra=extra)
    cols, _, s, a = log_pose.analyze(path)
    assert "acc_pitch_deg" in cols and abs(a["acc_pitch_deg"]).max() < 1e-6
    assert s["optional_columns_used"] == ["acc_x", "acc_y", "acc_z"]
    assert s["ignored_columns"] == ["something_new"]
    sections, figs, _, _ = run_report.build(path, out_dir=tmp_path / "out")
    titles = [t for t, _ in sections]
    assert "Servo telemetry" in titles and "Feedback law and posture controller" in titles
    tel = dict(sections)["Servo telemetry"][0]
    assert tel[2][0][2] == "12.0"                              # |load| max, decoded from sign-magnitude


def test_impossible_encoder_values_are_held_not_posed(tmp_path):
    path = write_log(tmp_path)
    rows = list(csv.DictReader(open(path)))
    rows[10]["u_meas.R_knee"] = "14338"                       # what the bus returned on 10/25 at t=42.3
    rows[11]["u_meas.L_knee"] = "nan"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    _, _, s, a = log_pose.analyze(path)
    assert s["u_meas_filled_rows"] == 2 and s["u_meas_bad_values"] == 2
    assert np.ptp(a["com_x_mm"]) < 1.0                         # the robot did not jump


def test_report_without_any_sidecar(tmp_path, monkeypatch):
    path = write_log(tmp_path, meta=False)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run_report.py", str(path), "--out-dir", str(out)])
    assert run_report.main() == 0
    md = (out / f"{path.stem}.report.md").read_text(encoding="utf-8")
    page = (out / f"{path.stem}.report.html").read_text(encoding="utf-8")
    assert "did not tip" in md and "events.jsonl no" in md and "console.txt no" in md
    assert "data:image/png;base64," in page and "<img" not in md
    assert (out / f"{path.stem}.pose.csv").exists() and (out / f"{path.stem}.pose_summary.json").exists()


def test_report_reads_events_and_console_warnings(tmp_path):
    path = write_log(tmp_path)
    stem = path.with_suffix("")
    Path(f"{stem}.events.jsonl").write_text(
        json.dumps({"t_wall": 1.8e9 + 1.0, "step": 40, "key": "p", "label": "push fwd", "note": ""})
        + "\nnot json\n", encoding="utf-8")
    Path(f"{stem}.console.txt").write_text(
        "1800000000.500 | POSTURE: knees += 1 x lean\n1800000001.250 | [SAFETY] upright_cos=0.40: cutting torque.\n",
        encoding="utf-8")
    sections, _, _, _ = run_report.build(path, out_dir=tmp_path / "out")
    md = run_report.render_md(path.stem, sections)
    assert "push fwd" in md and "| 1.00 | 40 | p |" in md         # t_rel derived from t_wall
    assert "unreadable lines**: 1" in md
    assert "cutting torque" in md and "POSTURE: knees" not in md
    assert "| 1.2 |" in md                                        # console time made relative


def _rewrite(path, edit):
    rows = list(csv.DictReader(open(path)))
    edit(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_a_still_straight_log_replays_upright_and_close_to_the_real_pitch(tmp_path):
    path = write_log(tmp_path, n=40)
    sections, figs, _, _ = run_report.build(path, out_dir=tmp_path / "out", do_replay=True)
    replay = dict(dict(sections)["Sim replay (open loop)"][0][1])
    assert "skipped" not in replay, replay.get("skipped")
    assert replay["result"] == "sim stayed up for the whole log"
    gap = float(re.match(r"median ([+-][\d.]+) deg", replay["pitch gap sim - real"]).group(1))
    # the flex plant (the 16 cm default) leans ~1.9 deg forward at the straight pose on its passive
    # foot hinges; it must hold that lean still, not drift
    assert 0.0 < gap < 2.5
    sim_std = float(re.match(r"([\d.]+) vs", replay["sim pitch std vs real"]).group(1))
    assert sim_std < 0.3
    assert "Sim replay (open loop)" in figs


def test_replay_that_cannot_run_is_skipped_with_the_reason(tmp_path):
    path = write_log(tmp_path, n=20)
    sections, figs, _, _ = run_report.build(path, out_dir=tmp_path / "out", do_replay=True,
                                            replay_model=tmp_path / "no_such_model.xml")
    replay = dict(dict(sections)["Sim replay (open loop)"][0][1])
    assert "replay failed" in replay["skipped"] and "Sim replay (open loop)" not in figs


def test_replay_starts_from_a_masked_pose_not_a_garbage_first_read(tmp_path, monkeypatch):
    path = write_log(tmp_path, n=20, q_meas=_q(R_knee=5, L_knee=5))
    _rewrite(path, lambda rows: rows[0].update({"u_meas.R_knee": "14338"}))
    seen = {}

    def fake_replay(xml, smap, q0, q_cmd, t, settle_s=2.0):
        seen["q0"] = q0
        return np.zeros(len(t)), np.zeros(len(t)), "faked"
    monkeypatch.setattr(run_report, "replay", fake_replay)
    run_report.build(path, out_dir=tmp_path / "out", do_replay=True)
    assert np.degrees(seen["q0"][DOFS.index("R_knee")]) == pytest.approx(5.0, abs=0.5)   # next good read


def test_joint_tracking_table_is_in_sim_degrees_so_mirrored_joints_agree(tmp_path):
    # both knees 5 deg MORE bent than told: the same physical error on two mirrored servos
    # (R_knee and L_knee have opposite servo signs, so raw units disagree in sign)
    path = write_log(tmp_path, q_meas=_q(R_knee=5, L_knee=5), q_cmd=_q())
    sections, _, _, _ = run_report.build(path, out_dir=tmp_path / "out")
    blocks = dict(sections)["Joint tracking (u_cmd - u_meas)"]
    table = next(b for b in blocks if b[0] == "table")
    mean = {r[0]: float(r[2]) for r in table[2]}
    assert mean["R_knee"] == pytest.approx(-5.0, abs=0.4)
    assert mean["L_knee"] == pytest.approx(-5.0, abs=0.4)


def test_rolling_left_moves_the_com_left_onto_the_left_foot(tmp_path):
    (tmp_path / "straight").mkdir()
    _, _, s0, a0 = log_pose.analyze(write_log(tmp_path / "straight"))
    _, _, s, a = log_pose.analyze(write_log(tmp_path, pg=_pg(roll_deg=6.0)))
    assert s["imu_roll_deg"]["median"] == pytest.approx(6.0, abs=0.1)
    # joints unchanged, so all of the new roll is flex
    assert s["flex_roll_deg"]["median"] - s0["flex_roll_deg"]["median"] == pytest.approx(6.0, abs=0.2)
    assert s["com_y_mm"]["median"] > s0["com_y_mm"]["median"] + 15.0      # + y = the robot's left
    assert s["com_y_mm"]["median"] > 0
    assert (a["anchor"] == "L").all() and (a["stance"] == "L").all()      # the right foot lifted
    # the COM moved toward the left foot: further past the right foot's inner edge, away from
    # its outer edge, and closer to the left foot's outer edge
    assert a["R_inner_mm"].max() < a0["R_inner_mm"].min()
    assert a["R_outer_mm"].min() > a0["R_outer_mm"].max()
    assert a["L_outer_mm"].max() < a0["L_outer_mm"].min()


def test_rolling_right_is_the_mirror_image(tmp_path):
    _, _, s, a = log_pose.analyze(write_log(tmp_path, pg=_pg(roll_deg=-6.0)))
    assert s["imu_roll_deg"]["median"] == pytest.approx(-6.0, abs=0.1)
    assert s["com_y_mm"]["median"] < 0
    assert (a["anchor"] == "R").all() and (a["stance"] == "R").all()


@pytest.mark.parametrize("roll, side", [(25.0, "left"), (-25.0, "right")])
def test_a_com_past_the_outer_edge_of_the_one_supporting_foot_is_named_by_side(tmp_path, roll, side):
    _, _, _, a = log_pose.analyze(write_log(tmp_path, pg=_pg(roll_deg=roll)))
    foot = side[0].upper()
    assert (a["stance"] == foot).all()
    assert (a["support_margin_mm"] < 0).all() and (a["support_nearest"] == side).all()
    assert (a[f"{foot}_outer_mm"] < 0).all()


def test_a_tip_to_the_left_is_reported_as_left(tmp_path):
    path = write_log(tmp_path, n=60, pg=_pg(roll_deg=70.0), upright=lambda i: 0.9 if i < 30 else 0.3)
    _, _, s, _ = log_pose.analyze(path)
    assert s["fell"] and s["fall"]["direction"] == "left"


def test_two_flat_feet_a_fraction_of_a_mm_apart_do_not_swap_the_anchor(tmp_path):
    # +-0.1 deg of roll noise about the attitude that has both soles level (the model's soles sit
    # ~0.75 deg rolled at the straight pose): the soles differ by ~0.3 mm, lower side alternating
    (tmp_path / "straight").mkdir()
    level = log_pose.analyze(write_log(tmp_path / "straight", n=10))[2]["enc_roll_deg"]["median"]
    _, _, s, a = log_pose.analyze(write_log(tmp_path, pg=lambda i: _pg(roll_deg=level + (0.1 if i % 2 else -0.1))))
    assert s["anchor_switches"] == 0
    assert (a["stance"] == "both").all() and s["stance_frames"]["both"] == len(a["stance"])


def test_hip_roll_puts_the_com_where_the_settled_sim_does(tmp_path):
    """Lateral geometry against MuJoCo itself: settle the sim at a one-sided hip-roll pose, log
    what its encoders and IMU would read, and the posed COM must sit where the sim's COM sits
    relative to each sole. (One-sided, so the two soles are NOT parallel and the IMU, not the
    encoders, has to decide which foot is flat.)"""
    import mujoco
    from sim_cop import settle

    m = mujoco.MjModel.from_xml_path(str(log_pose.MODEL_16CM))
    idx = {j.mjcf: k for k, j in enumerate(SMAP.joints)}
    perm = np.array([idx[m.joint(int(m.actuator_trnid[a, 0])).name] for a in range(m.nu)])
    jadr = m.jnt_qposadr[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.mjcf) for j in SMAP.joints]]
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    kin = log_pose.Kinematics(log_pose.MODEL_16CM, SMAP)
    sim = {}
    for dof in ("straight", "R_hip_roll", "L_hip_roll"):
        target = _q() if dof == "straight" else _q(**{dof: 4.0})
        d = settle(m, target=target[perm], seconds=3.0)
        q = np.array(d.qpos[jadr], dtype=float)
        pg = d.xmat[base].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        sim[dof] = {side: 1000 * (d.subtree_com[0][1] - d.geom_xpos[kin.foot[side]][1]) for side in ("R", "L")}
        sub = tmp_path / dof
        sub.mkdir()
        _, _, _, a = log_pose.analyze(write_log(sub, n=10, q_meas=q, pg=pg))
        for side in ("R", "L"):
            posed = float(np.median(a["com_y_mm"] - a[f"foot_{side}_y_mm"]))
            assert posed == pytest.approx(sim[dof][side], abs=3.0), (dof, side)
    # and the direction itself, in the sim: + hip roll pulls that leg's foot in under the COM
    assert sim["R_hip_roll"]["R"] < sim["straight"]["R"] - 15.0
    assert sim["L_hip_roll"]["L"] > sim["straight"]["L"] + 15.0
