#!/usr/bin/env python3
# camlib.py  --  shared by the camera ground-truth tools (scripts/camera/). Dev box only.
"""Tag config, rotation conventions and single-tag pose for the camera ground truth.

Everything that two camera tools must agree on lives here, so they cannot drift apart:
what a tag id means (config/camera/tags.yaml), which way a tag faces on its link, the
roll/pitch/yaw convention, and how one square tag becomes a 3D pose.

CONVENTIONS (the same as the sim and the frame log, on purpose):
  - Link frames are the sim base frame at q = 0: x forward, y left, z up. Every MJCF body
    of humanoid_real_v2 has identity orientation, so at q = 0 all links share these axes.
  - roll/pitch/yaw = intrinsic Z-Y-X: R = Rz(yaw) Ry(pitch) Rx(roll). Pitch > 0 = the
    link's top tilted forward, the same sign as pg_x in the frame log.
  - Tag frame (OpenCV): x right, y up as printed, z out of the printed face.

THE PLANAR AMBIGUITY. A small square seen nearly face-on has two poses that project to
almost the same four corners -- mirror tilts about the line of sight. IPPE_SQUARE returns
both with their reprojection errors. Picking the lower error flips between them frame to
frame when the errors are close, which shows up as a link snapping back and forth by
10-30 deg. choose_pose() keeps the one nearest the previous frame's pose (a link cannot
turn 20 deg in 1/60 s), and on the first frame the one whose link stands upright in the
world (tags are on links that start upright), falling back to the lower error only when
there is nothing else to go on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAGS = ROOT / "config" / "camera" / "tags.yaml"
CAMERA_DIR = ROOT / "config" / "camera"

# Columns = the tag's x, y, z axes expressed in LINK coordinates, for a tag mounted
# upright (label toward the floor) with its printed face looking the named way.
FACE_AXES = {
    "right": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "left": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "front": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "back": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    "top": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
}


def cv2_aruco():
    """cv2 with the >= 4.7 aruco API (ArucoDetector, CharucoDetector), or a clear error."""
    import cv2
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        raise SystemExit(f"OpenCV {cv2.__version__} has no cv2.aruco.ArucoDetector; "
                         "needs opencv-python >= 4.7 (pip install -r requirements.txt)")
    return cv2


# ---------------------------------------------------------------- rotations

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_ROT = {"x": rot_x, "y": rot_y, "z": rot_z}
_AXIS = {"x": 0, "y": 1, "z": 2}


def rpy_to_matrix(roll, pitch, yaw):
    """Radians, intrinsic Z-Y-X: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def euler_from_matrix(R, order="zyx"):
    """Intrinsic Tait-Bryan angles for R = R_a(t1) R_b(t2) R_c(t3), order = "abc".

    Returned as {"x": .., "y": .., "z": ..} (radians) -- by axis, not by position, so a
    caller asks for "the pitch" without caring where pitch sits in the chain. Using the
    joint chain's own order makes each angle equal one joint exactly (hip = x, z, y).
    """
    i, j, k = (_AXIS[a] for a in order)
    if len({i, j, k}) != 3:
        raise ValueError(f"order {order!r}: need three distinct axes")
    s = 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
    t2 = math.asin(max(-1.0, min(1.0, s * R[i, k])))
    t1 = math.atan2(-s * R[j, k], R[k, k])
    t3 = math.atan2(-s * R[i, j], R[i, i])
    return {order[0]: t1, order[1]: t2, order[2]: t3}


def matrix_to_rpy(R):
    """(roll, pitch, yaw) radians, intrinsic Z-Y-X -- the inverse of rpy_to_matrix."""
    e = euler_from_matrix(R, "zyx")
    return e["x"], e["y"], e["z"]


def rotation_angle(Ra, Rb):
    """Angle (radians) of the rotation taking Ra to Rb."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def mean_rotation(Rs):
    """Chordal L2 mean: the rotation nearest the element-wise average (SVD projection)."""
    M = np.mean(np.asarray(Rs), axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


# ---------------------------------------------------------------- config

@dataclass
class Tag:
    id: int
    link: str
    size: float                 # black square edge, metres
    face: str
    T_link_tag: np.ndarray      # 4x4: tag pose in its link's frame
    where: str = ""


@dataclass
class Pair:
    name: str
    parent: str
    child: str
    order: str = "zyx"
    encoder: dict = field(default_factory=dict)   # axis -> [(dof, sign), ...]


@dataclass
class TagConfig:
    dictionary: str
    world_tag: int | None
    tags: dict
    pairs: list
    board: dict
    path: Path | None = None

    @property
    def links(self):
        """Body links in config order (world excluded)."""
        out = []
        for t in self.tags.values():
            if t.link != "world" and t.link not in out:
                out.append(t.link)
        return out


def mount_transform(face="right", spin_deg=0.0, offset_mm=(0, 0, 0)):
    if face not in FACE_AXES:
        raise ValueError(f"face {face!r}: expected one of {sorted(FACE_AXES)}")
    R = np.array(FACE_AXES[face], dtype=float).T @ rot_z(math.radians(spin_deg))
    return make_T(R, np.asarray(offset_mm, dtype=float) / 1000.0)


def load_tag_config(path=DEFAULT_TAGS):
    path = Path(path)
    d = yaml.safe_load(path.read_text())
    tags = {}
    for k, v in (d.get("tags") or {}).items():
        tid = int(k)
        tags[tid] = Tag(id=tid, link=str(v["link"]), size=float(v["size_mm"]) / 1000.0,
                        face=v.get("face", "right"),
                        T_link_tag=mount_transform(v.get("face", "right"), v.get("spin_deg", 0.0),
                                                   v.get("offset_mm", (0, 0, 0))),
                        where=v.get("where", ""))
    pairs = []
    for name, v in (d.get("pairs") or {}).items():
        enc = {ax: [(str(dof), float(sign)) for dof, sign in terms]
               for ax, terms in (v.get("encoder") or {}).items()}
        pairs.append(Pair(name=name, parent=v["parent"], child=v["child"],
                          order=v.get("order", "zyx"), encoder=enc))
    wt = d.get("world_tag")
    return TagConfig(dictionary=d.get("dictionary", "DICT_APRILTAG_36h11"),
                     world_tag=None if wt is None else int(wt), tags=tags, pairs=pairs,
                     board=d.get("calibration_board") or {}, path=path)


# ---------------------------------------------------------------- camera intrinsics

def load_camera(path, image_size=None):
    """(K, dist, info) from a calibrate_camera.py YAML.

    If the video is a different resolution with the SAME aspect ratio, K is scaled to it
    (a phone usually bins the same sensor area) with a warning -- the mode may still crop,
    so calibrating at the filming resolution is always better. A different aspect ratio is
    a different crop of the sensor and is refused.
    """
    info = yaml.safe_load(Path(path).read_text())
    K = np.array(info["camera_matrix"], dtype=float).reshape(3, 3)
    dist = np.array(info["dist_coeffs"], dtype=float).ravel()
    w0, h0 = info["image_size"]
    if image_size is not None and tuple(image_size) != (w0, h0):
        w, h = image_size
        if abs(w / h - w0 / h0) > 0.01:
            raise SystemExit(f"{path}: calibrated at {w0}x{h0} but the video is {w}x{h} "
                             "(different aspect ratio = different sensor crop). Recalibrate in this mode.")
        s = w / w0
        print(f"WARNING: calibrated at {w0}x{h0}, video is {w}x{h}: scaling K by {s:.3f}. "
              "Recalibrate at the filming resolution if anything looks off.")
        K = K.copy()
        K[0, :] *= s
        K[1, :] *= s
        # pixel-centre convention: x_new + 0.5 = s * (x_old + 0.5)
        K[0, 2] += 0.5 * (s - 1)
        K[1, 2] += 0.5 * (s - 1)
    return K, dist, info


# ---------------------------------------------------------------- detection + pose

def make_detector(dictionary="DICT_APRILTAG_36h11"):
    cv2 = cv2_aruco()
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    p = cv2.aruco.DetectorParameters()
    # Sub-pixel corners: the pose of a 40 px tag is only as good as its corners.
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, p)


def tag_object_points(size):
    """Corner order of cv2.aruco (TL, TR, BR, BL), in the tag frame, as IPPE_SQUARE wants."""
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


@dataclass
class PoseCandidate:
    R: np.ndarray       # tag -> camera rotation
    t: np.ndarray       # tag centre in the camera frame, metres
    err: float          # RMS reprojection error, px


def solve_tag(corners, size, K, dist):
    """Both IPPE_SQUARE solutions for one tag, lowest reprojection error first."""
    cv2 = cv2_aruco()
    img = np.asarray(corners, dtype=np.float64).reshape(4, 1, 2)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(tag_object_points(size), img, K, dist,
                                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
    out = []
    for i in range(int(n)):
        R, _ = cv2.Rodrigues(rvecs[i])
        e = float(np.ravel(errs)[i]) if errs is not None and len(np.ravel(errs)) > i else float("nan")
        out.append(PoseCandidate(R=R, t=np.ravel(tvecs[i]).astype(float), err=e))
    return sorted(out, key=lambda c: c.err)


def choose_pose(cands, prev_R=None, prior_up=None, R_link_tag=None, rel_tol=2.0, abs_tol_px=0.3):
    """Pick one IPPE candidate. Returns (index, ambiguous).

    A candidate is admissible if its error is within rel_tol x the best or abs_tol_px of
    it; a clearly worse fit is never chosen. Among admissible ones: nearest the previous
    frame's rotation (prev_R, camera frame); else the one whose link z axis is nearest
    prior_up (world up in the camera frame); else the lowest error. ambiguous = more than
    one candidate was admissible, i.e. the choice was made by continuity or prior.
    """
    if not cands:
        raise ValueError("no candidates")
    best = cands[0].err
    tol = max(rel_tol * best, best + abs_tol_px)
    adm = [i for i, c in enumerate(cands) if c.err <= tol]
    # Two solutions that agree to within 2 deg are one answer, not an ambiguity.
    adm = [i for i in adm if i == adm[0] or rotation_angle(cands[adm[0]].R, cands[i].R) > math.radians(2.0)]
    if len(adm) == 1:
        return adm[0], False
    if prev_R is not None:
        return min(adm, key=lambda i: rotation_angle(prev_R, cands[i].R)), True
    if prior_up is not None:
        Rlt = np.eye(3) if R_link_tag is None else R_link_tag

        def up_err(i):
            z = (cands[i].R @ Rlt.T)[:, 2]
            return math.acos(max(-1.0, min(1.0, float(z @ prior_up))))
        return min(adm, key=up_err), True
    return adm[0], True


def refine_corners(gray, corners, n_samples=12, half_width_cells=0.6, step_px=0.05):
    """Re-fit the four corners from the tag's OUTER edges; returns (corners, ok).

    WHY. cv2's SUBPIX refinement looks at a window around each corner, and on a 35-45 px
    tag that window also holds the data cells next to the corner, which drag it inward by
    ~0.4 px. 0.4 px on a 35 px tag is a 2% size error = 2% depth (17 mm at 0.9 m) and
    ~1.8 deg of tilt, measured on synthetic renders (tests/test_camera_pose.py). The outer
    edge -- black border inside, white quiet zone outside, one full cell of each -- is a
    clean step along its whole length: find the 50% crossing at a dozen points per side,
    fit a line, intersect neighbours. Same renders: 0.05-0.12 px, ~1 mm, ~0.3 deg.

    Falls back to the input corners (ok = False) when a side has no clean step (white
    border trimmed off, occluded edge, heavy blur) or the fit moved a corner more than
    half a cell -- never worse than not refining.
    """
    cv2 = cv2_aruco()
    g = gray.astype(np.float32)
    c = np.asarray(corners, dtype=float).reshape(4, 2)
    ctr = c.mean(axis=0)
    cell = np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)) / 8.0
    offs = np.arange(-half_width_cells * cell, half_width_cells * cell + 1e-9, step_px)
    q = max(2, len(offs) // 4)
    lines = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([d[1], -d[0]])
        if np.dot(n, (a + b) / 2 - ctr) < 0:
            n = -n                                  # outward: black border -> white border
        pts = []
        for s in np.linspace(0.12, 0.88, n_samples):     # stay clear of the corners themselves
            p = a + s * (b - a)
            xy = (p[None, :] + offs[:, None] * n[None, :]).astype(np.float32)
            prof = cv2.remap(g, xy[:, 0].reshape(1, -1), xy[:, 1].reshape(1, -1), cv2.INTER_LINEAR).ravel()
            lo, hi = prof[:q].mean(), prof[-q:].mean()
            if hi - lo < 20:
                continue
            mid = (lo + hi) / 2
            k = np.flatnonzero((prof[:-1] < mid) & (prof[1:] >= mid))
            if len(k) != 1:
                continue
            k = int(k[0])
            f = (mid - prof[k]) / (prof[k + 1] - prof[k])
            pts.append(p + (offs[k] + f * step_px) * n)
        if len(pts) < 4:
            return c, False
        pts = np.array(pts)
        m = pts.mean(axis=0)
        lines.append((m, np.linalg.svd(pts - m)[2][0]))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[(i - 1) % 4], lines[i]
        A = np.column_stack([d1, -d2])
        if abs(np.linalg.det(A)) < 1e-6:
            return c, False
        s = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s[0] * d1)
    out = np.array(out)
    if np.max(np.linalg.norm(out - c, axis=1)) > 0.5 * cell:
        return c, False
    return out, True


def sharpness(gray, corners):
    """Variance of the Laplacian inside the tag's bounding box -- drops with motion blur."""
    cv2 = cv2_aruco()
    c = np.asarray(corners).reshape(-1, 2)
    h, w = gray.shape[:2]
    x0, y0 = np.floor(c.min(0)).astype(int)
    x1, y1 = np.ceil(c.max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1 + 1, w), min(y1 + 1, h)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return float("nan")
    return float(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var())
