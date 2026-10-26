"""Camera ground truth (scripts/camera/): tags rendered at a KNOWN pose must come back out.

Every image here is synthetic -- a tag texture warped through a known pinhole camera --
so the true pose is exact and the test runs headless. If these pass, a wrong number from a
real video is the print, the mount, the calibration or the filming, not the maths.
"""
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path("scripts/camera").resolve()))
import calibrate_camera  # noqa: E402
import camlib  # noqa: E402
import make_targets  # noqa: E402
import video_pose  # noqa: E402

W, H = 1920, 1080
K = np.array([[1100.0, 0, 960.0], [0, 1100.0, 540.0], [0, 0, 1]])
DIST = np.zeros(5)
CFG = camlib.load_tag_config()


# ---------------------------------------------------------------- rendering

def tag_texture(tag_id, cell_px=24):
    """Tag bitmap with its one-cell white quiet zone: 10 cells across, the black square 8."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, CFG.dictionary))
    m = cv2.aruco.generateImageMarker(d, tag_id, 8 * cell_px, borderBits=1)
    return cv2.copyMakeBorder(m, cell_px, cell_px, cell_px, cell_px, cv2.BORDER_CONSTANT, value=255)


def render(placements, K=K, size=(W, H), bg=150, ss=2, noise=1.0, seed=0):
    """placements: (texture, metres per texture pixel, T_cam_plane). Plane frame: x right,
    y up, z toward the camera -- the tag frame. Rendered at ss x and area-downsampled so
    edges are anti-aliased like a real sensor's."""
    w, h = size
    Ks = K.copy()
    Ks[:2] *= ss
    Ks[0, 2] = ss * K[0, 2] + 0.5 * (ss - 1)
    Ks[1, 2] = ss * K[1, 2] + 0.5 * (ss - 1)
    img = np.full((h * ss, w * ss), bg, np.uint8)
    for tex, m, T in placements:
        n_y, n_x = tex.shape
        A = np.array([[m, 0, (0.5 - n_x / 2) * m], [0, -m, (n_y / 2 - 0.5) * m], [0, 0, 1]])
        Hm = Ks @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ A
        cv2.warpPerspective(tex, Hm, (w * ss, h * ss), dst=img, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def tag_placement(tag_id, T_cam_tag, cell_px=24):
    tex = tag_texture(tag_id, cell_px)
    return tex, CFG.tags[tag_id].size / (8 * cell_px), T_cam_tag


def look_at(eye, target, up=(0, 0, 1)):
    """T_world_cam for a camera at eye looking at target (OpenCV: x right, y down, z forward)."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return camlib.make_T(np.column_stack([x, y, z]), eye)


# Camera on the robot's right (-y), ~0.75 m out, a bit above hip height, looking at the
# legs; the floor tag lies between camera and robot.
T_WORLD_CAM = look_at(eye=(0.0, -0.75, 0.45), target=(0.0, 0.0, 0.10))
T_CAM_WORLD = camlib.inv_T(T_WORLD_CAM)


def link_in_world(xyz, rpy_deg):
    return camlib.make_T(camlib.rpy_to_matrix(*np.radians(rpy_deg)), xyz)


def scene(links, floor_at=(0.0, -0.25, 0.0), floor_yaw_deg=0.0):
    """Placements for the floor tag + each {tag_id: T_world_link}, through tags.yaml mounts."""
    T_world_floor = camlib.make_T(camlib.rot_z(math.radians(floor_yaw_deg)), floor_at)
    out = [tag_placement(0, T_CAM_WORLD @ T_world_floor @ CFG.tags[0].T_link_tag, cell_px=12)]
    for tid, T_wl in links.items():
        out.append(tag_placement(tid, T_CAM_WORLD @ T_wl @ CFG.tags[tid].T_link_tag))
    return out, T_world_floor


def frames_of(images, fps=60.0):
    return ((i, i / fps, img) for i, img in enumerate(images))


# ---------------------------------------------------------------- maths

@pytest.mark.parametrize("order", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
def test_euler_decomposition_inverts_its_own_chain(order):
    rng = np.random.default_rng(1)
    for _ in range(20):
        ang = rng.uniform(-1.2, 1.2, 3)
        R = camlib._ROT[order[0]](ang[0]) @ camlib._ROT[order[1]](ang[1]) @ camlib._ROT[order[2]](ang[2])
        e = camlib.euler_from_matrix(R, order)
        assert [e[a] for a in order] == pytest.approx(list(ang), abs=1e-9)


def test_forward_lean_is_positive_pitch_like_pg_x():
    """The frame log's pg = R^T (0,0,-1): a forward lean gives pg_x = +sin(pitch)."""
    R = camlib.rpy_to_matrix(0.0, math.radians(10), 0.0)
    assert (R.T @ np.array([0, 0, -1.0]))[0] == pytest.approx(math.sin(math.radians(10)))
    assert R[0, 2] > 0                          # the link's z (up) axis tips toward +x


def test_mounts_are_proper_rotations_facing_the_named_way():
    normal = {"right": (0, -1, 0), "left": (0, 1, 0), "front": (1, 0, 0), "back": (-1, 0, 0), "top": (0, 0, 1)}
    for face, n in normal.items():
        R = camlib.mount_transform(face)[:3, :3]
        assert np.linalg.det(R) == pytest.approx(1.0)
        assert R[:, 2] == pytest.approx(n)
        if face != "top":
            assert R[:, 1] == pytest.approx((0, 0, 1))      # printed "up" = link up


def test_choose_pose_prefers_continuity_but_never_a_bad_fit():
    a = camlib.PoseCandidate(np.eye(3), np.zeros(3), 0.20)
    b = camlib.PoseCandidate(camlib.rot_x(0.4), np.zeros(3), 0.25)
    assert camlib.choose_pose([a, b], prev_R=camlib.rot_x(0.38)) == (1, True)
    assert camlib.choose_pose([a, b])[0] == 0                      # nothing to go on: lowest error
    bad = camlib.PoseCandidate(camlib.rot_x(0.4), np.zeros(3), 3.0)
    assert camlib.choose_pose([a, bad], prev_R=camlib.rot_x(0.4)) == (0, False)


# ---------------------------------------------------------------- pose recovery

@pytest.mark.parametrize("rpy,xyz", [((0, 0, 0), (0.0, 0.0, 0.6)),
                                     ((15, -20, 5), (0.06, -0.04, 0.55)),
                                     ((-10, 30, -8), (-0.05, 0.03, 0.7))])
def test_single_tag_pose_is_recovered_in_the_camera_frame(rpy, xyz):
    # Tag 3 (50 mm) facing the camera: tag z toward the lens = camera -z, then tilted.
    R_face = np.diag([1.0, -1.0, -1.0])
    T = camlib.make_T(R_face @ camlib.rpy_to_matrix(*np.radians(rpy)), xyz)
    img = render([tag_placement(3, T)])
    det = camlib.make_detector(CFG.dictionary)
    corners, ids, _ = det.detectMarkers(img)
    assert ids is not None and list(np.ravel(ids)) == [3]
    cands = camlib.solve_tag(corners[0].reshape(4, 2), CFG.tags[3].size, K, DIST)
    i, _ = camlib.choose_pose(cands, prev_R=T[:3, :3])     # continuity knows the truth
    c = cands[i]
    assert np.linalg.norm(c.t - T[:3, 3]) * 1000 < 5.0
    assert math.degrees(camlib.rotation_angle(c.R, T[:3, :3])) < 1.5
    assert c.err < 0.5


def test_world_frame_link_poses_and_joint_angle_from_a_side_view():
    T_thigh = link_in_world((0.0, 0.0, 0.30), (2.0, -8.0, 3.0))
    knee = 20.0                                            # R_knee, MJCF axis +y
    T_shin = T_thigh @ camlib.make_T(camlib.rot_y(math.radians(knee)), (0, 0, -0.16))
    T_foot = link_in_world(T_shin[:3, 3] + (0.03, 0.0, -0.12), (4.0, 0.0, 1.0))
    T_foot[2, 3] = 0.03
    truth = {"R_thigh": T_thigh, "R_shin": T_shin, "R_foot": T_foot}
    pl, T_world_floor = scene({5: T_thigh, 7: T_shin, 9: T_foot}, floor_yaw_deg=0.0)
    images = [render(pl, seed=s) for s in range(4)]
    res = video_pose.process(frames_of(images), CFG, K, DIST)

    assert res.world is not None and res.world.rot_p95_deg < 0.5
    row = res.link_rows[-1]
    assert row["world_ref"] == 1
    for link, T in truth.items():
        T_exp = camlib.inv_T(T_world_floor) @ T          # world = the floor tag's frame
        got = np.array([row[f"{link}.{a}"] for a in "xyz"])
        assert np.linalg.norm(got - T_exp[:3, 3]) * 1000 < 12.0, link
        exp_rpy = np.degrees(camlib.matrix_to_rpy(T_exp[:3, :3]))
        got_rpy = [row[f"{link}.{a}"] for a in ("roll", "pitch", "yaw")]
        assert got_rpy == pytest.approx(exp_rpy, abs=2.0), link
    assert row["R_knee.pitch"] == pytest.approx(knee, abs=2.0)
    assert row["R_ankle.pitch"] == pytest.approx(-(knee - 8.0), abs=2.5)


def test_world_yaw_follows_the_floor_tag_not_the_camera():
    """Turning the floor tag 30 deg turns the reported world yaw by -30 deg, pitch unchanged."""
    T_shin = link_in_world((0.0, 0.0, 0.15), (0.0, 6.0, 0.0))
    for yaw in (0.0, 30.0):
        pl, _ = scene({7: T_shin}, floor_yaw_deg=yaw)
        row = video_pose.process(frames_of([render(pl)]), CFG, K, DIST).link_rows[0]
        assert row["R_shin.yaw"] == pytest.approx(-yaw, abs=2.0)
        assert row["R_shin.pitch"] == pytest.approx(6.0, abs=1.5)


def test_missing_and_world_less_frames_are_flagged_not_dropped():
    pl, _ = scene({7: link_in_world((0.0, 0.0, 0.15), (0, 0, 0))})
    blank = np.full((H, W), 150, np.uint8)
    res = video_pose.process(frames_of([render(pl), blank]), CFG, K, DIST)
    rows = [r for r in res.tag_rows if r["tag_id"] == 7]
    assert [r["missing"] for r in rows] == [0, 1]
    assert len(res.tag_rows) == 2 * len(CFG.tags)
    # Without the floor tag, links are reported in the camera frame and say so.
    pl_no_floor = pl[1:]
    res = video_pose.process(frames_of([render(pl_no_floor)]), CFG, K, DIST)
    assert res.world is None and res.link_rows[0]["world_ref"] == 0


def _shin_candidates(*spins_and_errs):
    """IPPE-style candidates for tag 7 on an upright shin: the true pose turned by spin deg
    about the lens axis (spin 0 = the truth), each with its reprojection error."""
    T_cam_tag = T_CAM_WORLD @ link_in_world((0.0, 0.0, 0.15), (0, 0, 0)) @ CFG.tags[7].T_link_tag
    return [camlib.PoseCandidate(camlib.rot_z(math.radians(s)) @ T_cam_tag[:3, :3], T_cam_tag[:3, 3], e)
            for s, e in spins_and_errs]


def test_without_a_floor_tag_image_up_breaks_the_tilt_ambiguity(monkeypatch):
    """No floor tag -> no world up. The lower-error branch is then a coin flip for a near-
    frontal tag (0.02-0.05 px apart in renders), and continuity keeps whichever won. The
    phone is upright on a tripod, so image-up picks the upright link instead; and when a
    track that started wrong jumps, the same prior pulls it back."""
    first = video_pose.Detection(0, 0.0, 7, np.zeros((4, 2)), _shin_candidates((40, 0.20), (0, 0.22)), 1.0)
    monkeypatch.setattr(video_pose, "detect", lambda *a: ([(0, 0.0)], [first], (W, H), {}))
    video_pose.process(None, CFG, K, DIST)
    assert first.choice == 1 and first.ambiguous            # the upright one, not the lower error

    stuck = video_pose.Detection(0, 0.0, 7, np.zeros((4, 2)), _shin_candidates((70, 0.20)), 1.0)
    nxt = video_pose.Detection(1, 1 / 60, 7, np.zeros((4, 2)), _shin_candidates((40, 0.20), (0, 0.22)), 1.0)
    video_pose.resolve([stuck, nxt], CFG, prior_up=video_pose.CAMERA_UP)
    assert nxt.jump and nxt.choice == 1                     # continuity alone would keep spin 40


def test_cli_on_an_image_folder_writes_tags_links_and_summary(tmp_path):
    T_thigh = link_in_world((0.0, 0.0, 0.30), (0.0, -5.0, 0.0))
    T_shin = T_thigh @ camlib.make_T(camlib.rot_y(math.radians(12.0)), (0, 0, -0.16))
    pl, _ = scene({5: T_thigh, 7: T_shin})
    d = tmp_path / "clip"
    d.mkdir()
    for i in range(3):
        cv2.imwrite(str(d / f"f{i:03d}.png"), render(pl, seed=i))
    cam = tmp_path / "cam.yaml"
    cam.write_text(yaml.safe_dump({"image_size": [W, H], "camera_matrix": K.tolist(),
                                   "dist_coeffs": DIST.tolist()}))
    video_pose.main([str(d), "--camera", str(cam), "--fps", "60", "--out-dir", str(tmp_path)])
    links = list(csv.DictReader(open(tmp_path / "clip.links.csv")))
    tags = list(csv.DictReader(open(tmp_path / "clip.tags.csv")))
    assert len(links) == 3 and len(tags) == 3 * len(CFG.tags)
    assert float(links[1]["t_video"]) == pytest.approx(1 / 60, abs=1e-6)
    assert float(links[0]["R_knee.pitch"]) == pytest.approx(12.0, abs=2.0)
    assert (tmp_path / "clip.pose.json").exists()


# ---------------------------------------------------------------- calibration + printing

def test_calibration_recovers_known_intrinsics_from_synthetic_board_views():
    board, _ = calibrate_camera.make_board(CFG.board)
    sq = CFG.board["square_mm"] / 1000.0
    nx, ny = CFG.board["squares_x"], CFG.board["squares_y"]
    px_per_sq = 60
    tex = board.generateImage((nx * px_per_sq, ny * px_per_sq), marginSize=0, borderBits=1)
    tex = cv2.copyMakeBorder(tex, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    Kc = np.array([[900.0, 0, 485.0], [0, 900.0, 355.0], [0, 0, 1]])
    rng = np.random.default_rng(3)
    images = []
    for _ in range(14):
        R = np.diag([1.0, -1.0, -1.0]) @ camlib.rpy_to_matrix(*np.radians(rng.uniform(-35, 35, 3) * (1, 1, 0.5)))
        t = (rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05), rng.uniform(0.45, 0.6))
        images.append(render([(tex, sq / px_per_sq, camlib.make_T(R, t))], K=Kc, size=(960, 720), bg=90))
    cal = calibrate_camera.calibrate(frames_of(images, fps=1.0), CFG.board, max_frames=14)
    assert cal["rms_px"] < 0.5
    Kr = np.array(cal["camera_matrix"])
    assert Kr[0, 0] == pytest.approx(900.0, rel=0.01)
    assert Kr[1, 1] == pytest.approx(900.0, rel=0.01)
    assert Kr[0, 2] == pytest.approx(485.0, abs=5.0)
    assert Kr[1, 2] == pytest.approx(355.0, abs=5.0)


def test_targets_pdf_is_written_for_both_paper_sizes(tmp_path):
    paths = make_targets.main(["--out-dir", str(tmp_path)])
    assert {p.name for p in paths} == {"camera_targets_A4.pdf", "camera_targets_Letter.pdf"}
    for p in paths:
        assert p.read_bytes()[:5] == b"%PDF-" and p.stat().st_size > 5000
