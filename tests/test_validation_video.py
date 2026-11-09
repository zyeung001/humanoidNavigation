"""The end-of-training validation video: timeline, fall handling, drawing, encoding.

Everything but the last test runs without GL, so CI covers the driver and the overlays; the
render smoke test runs where an offscreen context exists (this laptop) and skips elsewhere.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.robustness import showcase, suite, video
from src.robustness.policies import HoldPosePolicy
from src.robustness.showcase import Beat

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open("config/real_humanoid_stand.yaml"))["standing"]
DT = 0.025

# A 3 s copy of the timeline: one of each kind of beat, small enough that holding the pose
# survives them, so the schedule is not moved by falls.
SHORT = [
    Beat(0.0, "start", "deploy start"),
    Beat(0.5, "push", "1 N behind", 1.0, "torso", (1.0, 0.0), duration_s=0.2),
    Beat(1.25, "kick", "kick", toward=(0.0, 1.0), dv=0.02, yaw_rate=0.1),
    Beat(1.75, "fault_on", "gyro", gyro=(0.0, 0.12, 0.0)),
    Beat(2.0, "push", "1 N pelvis", 1.0, "pelvis", (0.0, -1.0), duration_s=0.1),
    Beat(2.5, "fault_off", "off"),
]


@pytest.fixture(scope="module")
def wenv():
    return suite.make_eval_env(CFG, seconds=60.0)


@pytest.fixture(scope="module")
def short_run(wenv):
    frames = []
    res = showcase.run_showcase(HoldPosePolicy(CFG["residual_baseline"]), CFG, timeline=SHORT,
                                seconds=3.0, wenv=wenv, on_frame=lambda w, f: frames.append(f))
    return res, frames


# ---------------------------------------------------------------- 1. timeline and falls

def test_the_beats_fire_on_schedule_and_the_force_is_what_the_physics_got(short_run):
    res, frames = short_run
    assert res.falls == [] and res.seconds == pytest.approx(3.0)
    fired = {b["name"]: b["step"] for b in res.beats}
    assert fired == {"deploy start": 0, "1 N behind": 20, "kick": 50, "gyro": 70,
                     "1 N pelvis": 80, "off": 100}
    assert [(e["kind"], e["step"], e["source"]) for e in res.events] == \
        [("push", 20, "inject"), ("kick", 50, "inject"), ("push", 80, "inject")]
    force, body = res.trace["force"], res.trace["force_body"]
    for e in (res.events[0], res.events[2]):
        s, n = e["step"], e["duration_steps"]
        # trace[k] = state after k steps: the push acts during steps s+1 .. s+n, then is cleared
        np.testing.assert_allclose(force[s + 1:s + n + 1], np.tile(e["vec"], (n, 1)))
        assert set(body[s + 1:s + n + 1]) == {e["body_id"]}
        assert not np.any(force[s + n + 1]) and not np.any(force[s])
        assert np.linalg.norm(e["vec"]) == pytest.approx(1.0)
    assert res.events[0]["duration_steps"] == 8 and res.events[2]["duration_steps"] == 4
    assert res.events[0]["side"] == "behind" and res.events[2]["side"] == "left"
    # the fault is on from its beat (step 70, acting from step 71) until the fault-off beat
    # (step 100), and shown while it is; frames come every 2nd step
    on = [f["t"] for f in frames if f["fault"]]
    assert min(on) == pytest.approx(1.8) and max(on) == pytest.approx(2.5)
    assert "0.12 rad/s pitch" in frames[[f["t"] for f in frames].index(2.0)]["fault"]


def test_every_graded_beat_gets_an_outcome_and_the_result_round_trips(short_run, tmp_path):
    res, frames = short_run
    graded = res.graded
    assert [b["name"] for b in graded] == ["deploy start", "1 N behind", "kick", "1 N pelvis"]
    assert all(b["outcome"] == "held" for b in graded) and res.beats_held == 4
    assert all(b["max_lean_deg"] >= 0 for b in graded)
    back = showcase.ShowcaseResult.load(res.save(tmp_path / "v.json"))
    assert back == res and back.beats_held == 4
    assert frames[-1]["end_card"] and frames[-1]["summary"] == "beats held 4/4, falls 0"
    assert len(frames) == 61       # step 0, then every 2nd of 120 steps


def test_a_fall_is_shown_then_reset_and_the_rest_of_the_timeline_moves_later(wenv):
    tl = [Beat(0.0, "start", "deploy start"),
          Beat(0.5, "push", "6 N held 1 s", 6.0, "torso", (1.0, 0.0), duration_s=1.0),
          Beat(2.5, "push", "later", 0.5, "torso", (0.0, 1.0))]
    frames = []
    res = showcase.run_showcase(HoldPosePolicy(CFG["residual_baseline"]), CFG, timeline=tl, seconds=3.0,
                                wenv=wenv, max_resets=6, on_frame=lambda w, f: frames.append(f))
    assert len(res.falls) >= 1
    f0 = res.falls[0]
    assert f0["after"] == "6.0 N push from behind" and 0.0 < f0["after_s"] < 1.5
    phase = list(res.trace["phase"])
    k = f0["step"]
    assert phase[k + 1:k + 41] == ["falling"] * 40       # 1 s of pure physics, the fall itself
    assert phase[k + 41:k + 81] == ["settling"] * 40     # then reset, and 1 s to settle
    later = res.beats[-1]
    assert later["planned_t"] == 2.5 and later["t"] == pytest.approx(2.5 + 2.0 * len(res.falls))
    assert res.seconds == pytest.approx(3.0 + 2.0 * len(res.falls))
    assert res.beats[1]["outcome"] == "fell"
    assert any(f["banner"] and f["banner"].startswith("FELL") for f in frames)


def test_the_video_ends_after_max_resets_falls(wenv):
    tl = [Beat(0.0, "start", "s"), Beat(0.3, "push", "big", 20.0, "torso", (1.0, 0.0), duration_s=0.5),
          Beat(2.5, "push", "big again", 20.0, "torso", (1.0, 0.0), duration_s=0.5)]
    res = showcase.run_showcase(HoldPosePolicy(CFG["residual_baseline"]), CFG, timeline=tl, seconds=4.0,
                                wenv=wenv, max_resets=0)
    assert len(res.falls) == 1 and res.beats[-1]["outcome"] == "not reached"
    assert res.trace["phase"][-1] == "falling"


def test_the_end_card_is_held_when_the_video_ends_on_a_fall(wenv):
    """The last fall after max_resets ends the video mid-fall, where the end card is never on
    screen: the driver then hands over one final frame with the card, to hold for 2 s."""
    tl = [Beat(0.0, "start", "s"), Beat(0.3, "push", "big", 20.0, "torso", (1.0, 0.0), duration_s=0.5)]
    frames = []
    showcase.run_showcase(HoldPosePolicy(CFG["residual_baseline"]), CFG, timeline=tl, seconds=4.0,
                          wenv=wenv, max_resets=0, on_frame=lambda w, f: frames.append(f))
    assert frames[-1]["end_card"] and frames[-1]["hold_s"] == 2.0
    assert "FELL" in frames[-1]["end_card"][1] and frames[-2]["hud"][0] == "fallen"
    assert not any(f["hold_s"] for f in frames[:-1])


def test_the_showcase_is_reproducible_and_leaves_the_global_rng_alone(wenv):
    np.random.seed(99)
    before = np.random.get_state()[1].copy()
    a = showcase.run_showcase(HoldPosePolicy(CFG["residual_baseline"]), CFG, timeline=SHORT[:2], seconds=1.0, wenv=wenv)
    np.testing.assert_array_equal(np.random.get_state()[1], before)
    b = showcase.run_showcase(HoldPosePolicy(CFG["residual_baseline"]), CFG, timeline=SHORT[:2], seconds=1.0, wenv=wenv)
    np.testing.assert_array_equal(a.trace["lean"], b.trace["lean"])


def test_the_real_timeline_matches_the_suite_severities():
    by = {b.name: b for b in showcase.SHOWCASE}
    assert [b.newtons for b in showcase.SHOWCASE if b.kind == "push"] == [3, 4, 5, 6, 5, 5]
    kick = suite.SUITE[[s.name for s in suite.SUITE].index("kick_2x")].disturb
    assert by["kick 2x"].dv == pytest.approx(kick.magnitude[0])
    assert by["kick 2x"].yaw_rate == pytest.approx(kick.yaw_rate[0])
    gyro = suite.SUITE[[s.name for s in suite.SUITE].index("push_5N_gyro")].sensors.gyro_bias[0]
    assert np.linalg.norm(by["gyro fault"].gyro) == pytest.approx(gyro)
    assert showcase.SHOWCASE_SECONDS == 30.0 and sum(b.graded for b in showcase.SHOWCASE) == 8


# ---------------------------------------------------------------- 2. directions

def test_direction_labels_are_relative_to_the_robot():
    fwd = np.array([1.0, 0.0, 0.0])
    label = showcase.direction_label
    assert label([1, 0, 0], fwd) == "behind" and label([-1, 0, 0], fwd) == "front"
    assert label([0, 1, 0], fwd) == "right" and label([0, -1, 0], fwd) == "left"
    assert label([1, 0, 0], fwd, toward=True) == "front" and label([1, 1, 0], fwd, toward=True) == "front-left"
    turned = np.array([0.0, 1.0, 0.0])        # robot facing +y: its left is -x
    assert label([0, 1, 0], turned) == "behind" and label([-1, 0, 0], turned, toward=True) == "left"


def test_robot_forward_is_the_axis_checked_on_hardware():
    import mujoco
    spec = importlib.util.spec_from_file_location("sim_cop", ROOT / "scripts" / "debug" / "sim_cop.py")
    sim_cop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim_cop)
    m = mujoco.MjModel.from_xml_path(CFG["xml_file"])
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[3:7] = [np.cos(0.3), 0, 0, np.sin(0.3)]       # yawed 0.6 rad, so it is not just +x
    mujoco.mj_forward(m, d)
    np.testing.assert_allclose(showcase.robot_forward(m, d), sim_cop.forward_axis(m, d), atol=1e-12)
    assert sim_cop.pelvis_attitude(m, d) == pytest.approx(showcase.pelvis_attitude(m, d))


# ---------------------------------------------------------------- 3. arrows

def test_the_push_arrow_ends_short_of_the_point_with_its_tail_upwind():
    p = np.array([0.1, 0.2, 0.6])
    tail, head = video.arrow_segment(p, [0.0, 6.0, 0.0])
    assert np.linalg.norm(head - tail) == pytest.approx(0.05 * 6.0)
    np.testing.assert_allclose(head, p - [0.0, 0.03, 0.0])
    assert tail[1] < head[1]                         # the force points +y: the tail is at -y
    assert video.arrow_segment(p, [0.0, 0.0, 0.0]) is None


def test_project_puts_points_where_a_pinhole_camera_would():
    from types import SimpleNamespace
    cam = SimpleNamespace(pos=[0, 0, 0], forward=[1, 0, 0], up=[0, 0, 1], frustum_near=0.1,
                          frustum_top=0.05, frustum_bottom=-0.05, frustum_center=0.0, frustum_width=0.0)
    p = video.project(cam, [[1, 0, 0], [1, -0.25, 0], [1, 0, 0.25], [-1, 0, 0]], 400, 200)
    np.testing.assert_allclose(p[0], [200, 100])
    np.testing.assert_allclose(p[1], [250, 100])      # -y is to the right looking along +x
    np.testing.assert_allclose(p[2], [200, 50])       # up is up the image
    assert np.isnan(p[3]).all()                        # behind the camera


def test_aim_follows_the_com_and_puts_the_robot_right_of_centre_in_the_main_view(wenv):
    import mujoco
    d = wenv.base.env.unwrapped.data
    cams = video.make_cameras(showcase.robot_forward(wenv.base.env.unwrapped.model, d))
    video.aim(cams, d)
    com = np.asarray(d.subtree_com[1])
    np.testing.assert_allclose(cams["side"].lookat, com)
    np.testing.assert_allclose(np.linalg.norm(cams["main"].lookat - com), video.MAIN_SHIFT_M)
    assert cams["main"].type == mujoco.mjtCamera.mjCAMERA_FREE


def test_force_colour_ramps_yellow_to_red_and_clips():
    g = [video.force_colour(n)[1] for n in np.linspace(0, 12, 25)]
    assert video.force_colour(0.0) == (1.0, 1.0, 0.0) and video.force_colour(8.0) == (1.0, 0.0, 0.0)
    assert all(a >= b for a, b in zip(g, g[1:])) and video.force_colour(20.0) == video.force_colour(8.0)


# ---------------------------------------------------------------- 5. overlays and encoding

def test_compose_frame_draws_onto_the_views_at_full_size(short_run, wenv):
    res, frames = short_run
    fr = dict(frames[12], label="PUSH 1.0 N from behind . torso . 0.20 s . 0.2 N*s",
              push={"t0": frames[12]["t"], "vec": [1.0, 0.0, 0.0], "body_id": 15, "duration_s": 0.2},
              banner="FELL 1.00 s (tilt), 0.50 s after 1.0 N push from behind", fault="FAULT gyro")
    sup = video.support_geometry(wenv)
    assert len(sup["feet"]) == 2 and sup["left_index"] in (0, 1)
    main = np.zeros((472, 640, 3), np.uint8)
    side = np.zeros((236, 320, 3), np.uint8)
    img = video.compose_frame(main, side, fr, sup, "run . best@1.00M (score 50.0)")
    assert img.shape == (540, 960, 3) and img.dtype == np.uint8
    for region in (img[:28], img[28:500, :640], img[264:500, 640:], img[500:]):
        assert len(np.unique(region.reshape(-1, 3), axis=0)) > 5     # something was drawn there
    card = video.compose_frame(main, side, frames[-1], sup, "x")
    assert not np.array_equal(card[28:500, :640], img[28:500, :640])


def test_a_writer_opens_and_the_file_reads_back(tmp_path):
    import cv2
    wr, path = video.open_writer(tmp_path / "clip", 20, 160, 120)
    if wr is None:
        pytest.skip("no video encoder in this opencv build")
    for i in range(10):
        wr.write(np.full((120, 160, 3), 20 * i, np.uint8))
    wr.release()
    assert path.exists() and path.stat().st_size > 0 and not list(tmp_path.glob("*.probe*"))
    ok, frame = cv2.VideoCapture(str(path)).read()
    assert ok and frame.shape == (120, 160, 3)


# ---------------------------------------------------------------- 6. rendering (GL only)

@pytest.fixture(scope="module")
def gl():
    ok, why = video.gl_available()
    if not ok:
        pytest.skip(f"no GL context ({why})")


def test_mujoco_draws_the_arrow_at_the_length_asked_for(gl):
    """_add_arrow doubles size[2] because MuJoCo 3.8 draws an arrow geom over half of it. If a
    MuJoCo upgrade changes that, every arrow in the video is 2x off; this catches it."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><visual><global offwidth="400" offheight="300"/></visual><worldbody><light pos="0 0 3"/>'
        '<geom type="box" size="0.4 0.4 0.01" pos="0 0.5 0"/></worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 300, 400)
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (0, 0, 0.5), 1.6, 90, 0
    r.update_scene(d, camera=cam)
    bg = r.render().astype(int)
    width = []
    for draw in ("arrow", "capsule"):
        r.update_scene(d, camera=cam)
        if draw == "arrow":
            video._add_arrow(r.scene, [-0.4, 0, 0.5], [0.0, 0, 0.5], (1, 0, 0, 1), 0.012)
        else:
            g = r.scene.geoms[r.scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                np.array([1, 0, 0, 1], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, np.array([-0.4, 0, 0.5]),
                                 np.array([0.0, 0, 0.5]))
            r.scene.ngeom += 1
        cols = np.nonzero((np.abs(r.render().astype(int) - bg).sum(axis=2) > 30).any(axis=0))[0]
        width.append(cols.max() - cols.min())
        if draw == "arrow":
            drawn = (cols.min(), cols.max())
            gl_cam = mujoco.mjv_averageCamera(r.scene.camera[0], r.scene.camera[1])
            projected = video.project(gl_cam, [[-0.4, 0, 0.5], [0.0, 0, 0.5]], 400, 300)[:, 0]
    r.close()
    assert abs(width[0] - width[1]) <= 0.1 * width[1], width     # the capsule adds its round caps
    # The flat x-ray copy of the arrow lands on the shaded one.
    np.testing.assert_allclose(sorted(projected), drawn, atol=3)


def test_a_short_validation_video_shows_the_robot_and_the_push(gl, tmp_path):
    import cv2
    tl = [Beat(0.0, "start", "s"), Beat(0.3, "push", "5 N right", 5.0, "torso", (0.0, 1.0), duration_s=0.3)]
    res = video.make_validation_video(HoldPosePolicy(CFG["residual_baseline"]), CFG, tmp_path / "v", "test",
                                      width=160, height=120, timeline=tl, seconds=1.0, progress=None)
    assert res.path.exists() and res.json_path.exists() and res.frames == 21
    cap = cv2.VideoCapture(str(res.path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    assert len(frames) >= 1 and all(len(np.unique(f.reshape(-1, 3), axis=0)) > 10 for f in frames)

    # The arrow itself: render one view with and without this frame's arrows.
    import mujoco
    w = suite.make_eval_env(CFG, seconds=60.0)
    w.base.env.unwrapped.model.vis.global_.offwidth = 640
    r = mujoco.Renderer(w.base.env.unwrapped.model, 120, 160)
    seen = {}

    def grab(wenv, fr):
        if fr["step"] == 16:          # 0.1 s into the push
            d = wenv.base.env.unwrapped.data
            cams = video.make_cameras(fr["fwd"])
            video.aim(cams, d)
            cam = cams["main"]
            r.update_scene(d, camera=cam)
            plain = r.render().copy()
            r.update_scene(d, camera=cam)
            for a in video.scene_arrows(d, fr):
                video._add_arrow(r.scene, *a)
            seen["changed"] = int((np.abs(r.render().astype(int) - plain).sum(axis=2) > 30).sum())
    showcase.run_showcase(HoldPosePolicy(CFG["residual_baseline"]), CFG, timeline=tl, seconds=0.5, wenv=w,
                          on_frame=grab)
    r.close()
    assert seen["changed"] > 20


# ---------------------------------------------------------------- the end-of-training hook

def _hook(tmp_path, monkeypatch, gl=(True, ""), make=None):
    import json

    from src.robustness import policies
    from src.training.experiment import RunDir
    from src.training.run_logger import MetricsSink
    spec = importlib.util.spec_from_file_location("train_standing", ROOT / "scripts" / "train_standing.py")
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    run = RunDir.create("vv", base=tmp_path)
    (run.root / "config.yaml").write_text("{}")
    run.final.mkdir()
    (run.final / "model.zip").write_bytes(b"")
    monkeypatch.setattr(video, "gl_available", lambda: gl)
    monkeypatch.setattr(policies.SB3Policy, "load",
                        classmethod(lambda cls, *a, label=None: HoldPosePolicy(CFG["residual_baseline"], label=label)))
    if make is not None:
        monkeypatch.setattr(video, "make_validation_video", make)
    sink = MetricsSink(run.metrics)
    out = ts._validation_video(run, CFG, sink, 1_000_000, {})
    sink.close()
    return out, run, [json.loads(line) for line in run.metrics.read_text().splitlines()]


def test_a_failure_while_filming_only_warns(tmp_path, monkeypatch, capsys):
    def broken(*a, **k):
        raise RuntimeError("renderer lost its context")
    out, run, rows = _hook(tmp_path, monkeypatch, make=broken)
    assert out is None and rows[-1]["validation/video_failed"] == 1.0
    said = capsys.readouterr().out
    assert "WARNING: validation video failed (RuntimeError: renderer lost its context)" in said
    assert f"scripts/validation_video.py --run {run.root}" in said


def test_ctrl_c_while_filming_still_lets_the_run_close_its_logs(tmp_path, monkeypatch, capsys):
    def interrupted(*a, **k):
        raise KeyboardInterrupt
    out, run, rows = _hook(tmp_path, monkeypatch, make=interrupted)
    assert out is None and rows[-1]["validation/video_failed"] == 1.0
    assert "validation video interrupted" in capsys.readouterr().out


def test_a_video_w_and_b_refuses_does_not_stop_the_run_being_finished(tmp_path):
    from src.training.run_logger import MetricsSink

    class Run:
        finished = False

        def log(self, data, step):
            raise ValueError("unsupported video")

        def finish(self):
            Run.finished = True
    sink = MetricsSink(tmp_path / "m.jsonl", wandb_run=Run())
    sink.log_video("validation/video", tmp_path / "v.mp4", 10)
    assert sink.wandb is not None
    sink.close()
    assert Run.finished


def test_no_gl_skips_the_video_and_says_why(tmp_path, monkeypatch, capsys):
    out, _, rows = _hook(tmp_path, monkeypatch, gl=(False, "EGL not found"))
    assert out is None and rows[-1]["validation/video_skipped"] == 1.0
    assert "validation video skipped: no GL context (EGL not found)" in capsys.readouterr().out


def test_the_hook_films_final_when_there_is_no_best_and_logs_the_file(tmp_path, monkeypatch):
    got = {}

    def fake(policy, env_cfg, out_stem, label, **kw):
        got.update(stem=Path(out_stem), label=label, kw=kw)
        p = Path(out_stem).with_suffix(".mp4")
        p.write_bytes(b"x")
        return video.VideoResult(p, p.with_suffix(".json"), 600, 30.0, 2, 6, 8, 35.0)
    out, run, rows = _hook(tmp_path, monkeypatch, make=fake)
    assert got["stem"] == run.root / "validation" and got["label"] == "vv . final (unscored)"
    assert got["kw"] == {"width": 960, "height": 540, "fps": 20.0}
    assert rows[0]["validation/video"] == str(run.root / "validation.mp4")
    assert rows[1]["validation/falls"] == 2 and rows[1]["validation/beats_held"] == 6


def test_checkpoint_labels_and_choice(tmp_path):
    import json
    (tmp_path / "final").mkdir()
    assert video.pick_checkpoint(tmp_path) == tmp_path / "final"
    (tmp_path / "best").mkdir()
    (tmp_path / "best" / "model.zip").write_bytes(b"")
    (tmp_path / "best" / "eval.json").write_text(json.dumps({"step": 45_000_000, "score": 74.81}))
    assert video.pick_checkpoint(tmp_path) == tmp_path / "best"
    assert video.pick_checkpoint(tmp_path, "final") == tmp_path / "final"
    assert video.checkpoint_label(tmp_path / "best") == "best@45.00M (score 74.8)"
