"""The validation video: the showcase timeline (showcase.py) filmed, one file per run.

    +------------------------------ top bar: run . checkpoint . t ------------------------------+
    | main view: 3/4 from front-left, push arrow,       | side view: from the robot's right      |
    | event label, HUD (pitch/roll, height, recovery)  +----------------------------------------+
    |                                                   | support map: feet, COM, push, top-down |
    +------------------------- timeline: beats, faults, falls, playhead -------------------------+

The PUSH ARROW is what the physics applied: read every frame from data.xfrc_applied, drawn
ending 3 cm short of the body's centre of mass (where MuJoCo applies it), 5 cm per newton,
yellow at 0 N to red at 8 N, then a faint afterimage until 0.6 s after onset. It is drawn
twice: shaded in 3D, and flat and translucent over the view, so a push from behind the robot
still shows through its torso. A KICK has no
force to read, so its cyan arrow (0.5 m per m/s, at the pelvis COM) comes from the event log.
The SUPPORT MAP answers "from where and how much" without any camera-angle ambiguity: forward
is up, 1.2 px/mm, feet filled while in contact, COM dot red once it leaves the feet's hull.

Encoding is cv2.VideoWriter, no new dependency: H.264 mp4 through Media Foundation on Windows
(plays in browsers and W&B), else VP9/VP8 webm or mp4v through opencv's bundled ffmpeg.

cv2 and mujoco's renderer are imported lazily, so importing this module costs nothing.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .showcase import SHOWCASE_SECONDS, robot_forward, robot_left, run_showcase
from .suite import make_eval_env

M_PER_N = 0.05          # push arrow: 5 cm per newton (6 N = 30 cm on a 75 cm robot)
ARROW_GAP = 0.03        # head stops this short of the COM, so it is not buried in the body
FULL_RED_N = 8.0        # colour ramp: yellow at 0 N, red at this and above
AFTERIMAGE_S = 0.6      # arrows stay up (faint) this long after onset
KICK_M_PER_MPS = 0.5    # kick arrow: 0.5 m per m/s of velocity change
MAP_PX_PER_MM = 1.2     # support map scale at full size
MAP_PX_PER_N = 12.0     # push arrow on the map (the 3D scale would leave the map at 4 N)
MAP_PX_PER_MPS = 300.0  # kick arrow on the map
ARROW_DRAW_SCALE = 2.0  # see _add_arrow
XRAY_ALPHA = 0.5        # the arrow is also drawn flat over each view, so the body cannot hide it
MAIN_SHIFT_M = 0.20     # main view aims this far left of the robot: it stands clear of the HUD

# RGB, 0-255
_WHITE, _GREY, _DARK = (235, 235, 235), (140, 140, 140), (22, 24, 28)
_AMBER, _GREEN, _RED, _CYAN = (255, 176, 32), (90, 210, 110), (235, 60, 50), (60, 220, 235)


def arrow_segment(point, force, m_per_n: float = M_PER_N, gap: float = ARROW_GAP):
    """(tail, head) of the arrow for `force` applied at `point`: length m_per_n * |F|, head
    `gap` short of the point, tail upwind. None for a zero force."""
    f = np.asarray(force, dtype=np.float64)
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        return None
    u = f / n
    head = np.asarray(point, dtype=np.float64) - gap * u
    return head - m_per_n * n * u, head


def force_colour(newtons: float, full: float = FULL_RED_N):
    """RGB in 0-1: yellow at 0 N, red at `full` N and above."""
    x = float(np.clip(newtons / full, 0.0, 1.0))
    return (1.0, 1.0 - x, 0.0)


def _rgb255(c):
    return tuple(int(round(255 * v)) for v in c)


# ------------------------------------------------------------------------------ layout

@dataclass(frozen=True)
class Layout:
    W: int
    H: int
    top: int
    tl: int
    main_w: int
    main_h: int
    side_w: int
    side_h: int
    map_h: int
    s: float          # text/line scale relative to 960x540

    @classmethod
    def of(cls, W: int, H: int) -> "Layout":
        s = H / 540.0
        top, tl = max(8, round(28 * s)), max(8, round(40 * s))
        main_w = round(W * 2 / 3)
        main_h = H - top - tl
        side_h = main_h // 2
        return cls(W, H, top, tl, main_w, main_h, W - main_w, side_h, main_h - side_h, s)


# ------------------------------------------------------------------------------ 3D scene

def scene_arrows(data, frame) -> List[tuple]:
    """(tail, head, rgba, width) for every arrow this frame: applied forces (solid), the last
    push's afterimage, the last kick."""
    out = []
    t = frame["t"]
    f = data.xfrc_applied[:, :3]
    active = np.nonzero(np.abs(f).sum(axis=1) > 0)[0]
    for bid in active:
        seg = arrow_segment(data.xipos[bid], f[bid])
        if seg is not None:
            out.append((*seg, (*force_colour(np.linalg.norm(f[bid])), 1.0), 0.012))
    p = frame.get("push")
    if not len(active) and p is not None and t - p["t0"] < AFTERIMAGE_S:
        seg = arrow_segment(data.xipos[p["body_id"]], p["vec"])
        if seg is not None:
            out.append((*seg, (*force_colour(np.linalg.norm(p["vec"])), 0.35), 0.012))
    k = frame.get("kick")
    if k is not None and t - k["t0"] < AFTERIMAGE_S:
        tail = np.array(data.xipos[1], dtype=np.float64)
        out.append((tail, tail + KICK_M_PER_MPS * np.asarray(k["vec"]), (0.25, 0.85, 0.95, 1.0), 0.010))
    return out


def _add_arrow(scene, tail, head, rgba, width) -> None:
    import mujoco  # noqa: PLC0415
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, float(width),
                         np.asarray(tail, dtype=np.float64), np.asarray(head, dtype=np.float64))
    # MuJoCo draws an arrow geom over HALF its size[2] (ARROW2 spans +-size[2]/2), so a
    # connector arrow ends halfway: 0.40 m asked, 0.20 m drawn (3.8.1, measured 10/25). The
    # GL test in tests/test_validation_video.py checks the drawn length against a capsule.
    g.size[2] *= ARROW_DRAW_SCALE
    scene.ngeom += 1


def make_cameras(fwd) -> dict:
    """Cameras that follow the pelvis subtree COM (aim() each frame), fixed in azimuth from
    the heading at reset (they do not turn with the robot, so a yaw kick shows). MuJoCo's
    azimuth is the direction the camera LOOKS: main sits front-left looking back-right, side
    sits on the robot's right looking along its left axis, so a fore-aft push crosses the side
    view. The main view aims MAIN_SHIFT_M to the robot's image-left, so the robot stands right
    of centre and the HUD box on the left does not cover it."""
    import mujoco  # noqa: PLC0415
    psi = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cams = {}
    for name, az in (("main", psi + 225.0), ("side", psi + 90.0)):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.distance, c.elevation, c.azimuth = 1.6, -12.0, az
        cams[name] = c
    return cams


def aim(cams: dict, data) -> None:
    """Point every camera at the pelvis subtree COM (what mjCAMERA_TRACKING does), the main
    one MAIN_SHIFT_M to the left of it in the image."""
    com = np.asarray(data.subtree_com[1], dtype=np.float64)
    for name, c in cams.items():
        a = np.radians(c.azimuth)
        right = np.array([np.sin(a), -np.cos(a), 0.0])     # image-right for this azimuth
        c.lookat[:] = com - (MAIN_SHIFT_M if name == "main" else 0.0) * right


def project(cam, pts, w: int, h: int) -> np.ndarray:
    """World points -> pixel (x, y) in a w x h render through mjvGLCamera `cam`. For a mono
    view pass mujoco.mjv_averageCamera of the scene's two eyes: that is what mjr_render draws
    (either eye alone is off by half the stereo baseline, ~14 px here). Points behind the
    camera come back as nan."""
    pos, fwd, up = (np.asarray(v, dtype=np.float64) for v in (cam.pos, cam.forward, cam.up))
    right = np.cross(fwd, up)
    v = np.atleast_2d(np.asarray(pts, dtype=np.float64)) - pos
    z = v @ fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (v @ right) * cam.frustum_near / z
        y = (v @ up) * cam.frustum_near / z
    span = cam.frustum_top - cam.frustum_bottom
    half_w = cam.frustum_width if cam.frustum_width > 0 else 0.5 * (w / h) * span
    px = ((x - cam.frustum_center) / half_w + 1.0) / 2.0 * w
    py = (1.0 - (2.0 * (y - cam.frustum_bottom) / span - 1.0)) / 2.0 * h
    out = np.stack([px, py], axis=1)
    out[z <= 0] = np.nan
    return out


def draw_xray(img, cam, arrows, s: float) -> None:
    """Each arrow again, flat and translucent, on top of the rendered view: a push from the
    far side of the robot is otherwise hidden behind its own torso."""
    import cv2  # noqa: PLC0415
    if not arrows:
        return
    h, w = img.shape[:2]
    over = img.copy()
    for tail, head, rgba, _ in arrows:
        (x0, y0), (x1, y1) = project(cam, [tail, head], w, h)
        n = float(np.hypot(x1 - x0, y1 - y0))
        if not np.isfinite(n) or n < 2:
            continue
        col = tuple(int(round(255 * c * rgba[3] + 255 * (1 - rgba[3]) * 0.5)) for c in rgba[:3])
        cv2.arrowedLine(over, (int(x0), int(y0)), (int(x1), int(y1)), col, max(1, round(3 * s)),
                        cv2.LINE_AA, tipLength=min(0.45, 12 * s / n))
    cv2.addWeighted(over, XRAY_ALPHA, img, 1.0 - XRAY_ALPHA, 0.0, dst=img)


# ------------------------------------------------------------------------------ support map

def _foot_geoms(wenv) -> List[tuple]:
    """(geom, foot index) for the foot plates: thin box collision geoms (sim_cop._foot_geoms'
    rule) under each foot body. Cached on the wrapper."""
    cached = getattr(wenv, "_showcase_feet", None)
    if cached is not None:
        return cached
    import mujoco  # noqa: PLC0415
    base = wenv.base
    m = base.env.unwrapped.model
    feet = [int(b) for b in list(getattr(base.model_spec, "foot_body_ids", []) or [])[:2]]

    def foot_of(b):
        while b > 0:
            if b in feet:
                return feet.index(b)
            b = int(m.body_parentid[b])
        return None

    out = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or not m.geom_contype[g]:
            continue
        sz = np.sort(m.geom_size[g])
        i = foot_of(int(m.geom_bodyid[g]))
        if sz[0] < 0.01 and sz[1] > 0.02 and sz[2] > 0.04 and i is not None:
            out.append((g, i))
    wenv._showcase_feet = out
    return out


def support_geometry(wenv) -> dict:
    """Everything the support map needs, in world metres (no drawing here)."""
    base = wenv.base
    m, d = base.env.unwrapped.model, base.env.unwrapped.data
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    feet = []
    for g, i in _foot_geoms(wenv):
        corners = d.geom_xpos[g] + (signs * m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3).T
        feet.append((corners[:, :2].copy(), i))
    fwd = robot_forward(m, d)
    centre = (np.mean([c.mean(axis=0) for c, _ in feet], axis=0) if feet
              else np.asarray(d.subtree_com[1][:2], dtype=np.float64))
    side = {}
    for c, i in feet:     # which contact flag is the left foot: the one further along +left
        side[i] = float((c.mean(axis=0) - centre) @ robot_left(fwd)[:2])
    left_index = max(side, key=side.get) if side else 0
    return {"feet": feet, "centre": centre, "com": np.asarray(d.subtree_com[1][:2], dtype=np.float64),
            "fwd": fwd, "contacts": [float(x) for x in base._foot_contacts()], "left_index": left_index}


def draw_support(sup: dict, frame: dict, w: int, h: int, s: float) -> np.ndarray:
    import cv2  # noqa: PLC0415
    img = np.full((h, w, 3), (30, 33, 38), np.uint8)
    k = MAP_PX_PER_MM * 1000.0 * s
    cx, cy = w / 2.0, h / 2.0 + 6 * s
    fwd = np.asarray(sup["fwd"], dtype=np.float64)[:2]
    left = robot_left(np.append(fwd, 0.0))[:2]

    def px(p):
        q = np.asarray(p, dtype=np.float64) - sup["centre"]
        return (cx - (q @ left) * k, cy - (q @ fwd) * k)

    def vec_px(v):
        v = np.asarray(v, dtype=np.float64)[:2]
        return np.array([-(v @ left), -(v @ fwd)])

    step = 0.05 * k                                    # 5 cm grid, in robot axes
    for i in range(-8, 9):
        x, y = int(round(cx + i * step)), int(round(cy + i * step))
        col = (62, 66, 74) if i else (84, 88, 98)
        cv2.line(img, (x, 0), (x, h), col, 1)
        cv2.line(img, (0, y), (w, y), col, 1)

    allpts = []
    for c, i in sup["feet"]:
        pts = np.array([px(p) for p in c], dtype=np.float32)
        hull = cv2.convexHull(pts).astype(np.int32)
        allpts.append(pts)
        on = sup["contacts"][i] > 0.5 if i < len(sup["contacts"]) else False
        if on:
            cv2.fillPoly(img, [hull], (70, 130, 80))
        cv2.polylines(img, [hull], True, (170, 200, 170) if on else _GREY, max(1, round(2 * s)),
                      cv2.LINE_AA)
    inside = True
    if allpts:
        hull = cv2.convexHull(np.concatenate(allpts)).astype(np.float32)
        cv2.polylines(img, [hull.astype(np.int32)], True, (110, 116, 128), 1, cv2.LINE_AA)
        inside = cv2.pointPolygonTest(hull, tuple(float(v) for v in px(sup["com"])), False) >= 0

    com = np.array(px(sup["com"]))
    t = frame["t"]
    p = frame.get("push")
    if p is not None and t - p["t0"] < AFTERIMAGE_S:
        live = t - p["t0"] <= p.get("duration_s", 0.2) + 1e-9
        n = float(np.linalg.norm(p["vec"]))
        u = vec_px(p["vec"]) / max(1e-9, np.linalg.norm(vec_px(p["vec"])))
        head = com - u * 9 * s
        tail = head - u * MAP_PX_PER_N * n * s
        col = _rgb255(force_colour(n))
        if not live:
            col = tuple(int(0.45 * c + 0.55 * b) for c, b in zip(col, (30, 33, 38)))
        cv2.arrowedLine(img, tuple(int(v) for v in tail), tuple(int(v) for v in head), col,
                        max(1, round(3 * s)), cv2.LINE_AA, tipLength=0.25)
        lab = f"{n:.1f} N"                                   # beside the shaft, not on it
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, 1)
        perp = np.array([-u[1], u[0]]) if u[1] <= 0 else np.array([u[1], -u[0]])
        at = 0.5 * (head + tail) + perp * 10 * s
        at = at + np.array([0 if perp[0] >= 0 else -tw, th / 2 if abs(perp[1]) < 0.7 else
                            (th if perp[1] > 0 else 0)])
        _text(img, lab, (int(at[0]), int(at[1])), 0.45 * s, col)
    kk = frame.get("kick")
    if kk is not None and t - kk["t0"] < AFTERIMAGE_S:
        head = com + vec_px(kk["vec"]) * MAP_PX_PER_MPS * s
        cv2.arrowedLine(img, tuple(int(v) for v in com), tuple(int(v) for v in head), _CYAN,
                        max(1, round(3 * s)), cv2.LINE_AA, tipLength=0.25)
    cv2.circle(img, tuple(int(v) for v in com), max(2, round(5 * s)), _WHITE if inside else _RED, -1,
               cv2.LINE_AA)
    _text(img, "top-down . forward up . 5 cm grid", (round(6 * s), round(15 * s)), 0.4 * s, _GREY)
    _text(img, "COM" + ("" if inside else " OUTSIDE FEET"), (round(6 * s), h - round(7 * s)), 0.4 * s,
          _WHITE if inside else _RED)
    return img


# ------------------------------------------------------------------------------ 2D overlays

def _text(img, s, org, scale, colour, thick: int = 1, outline: bool = False) -> None:
    """Text; with `outline` a dark rim too, for text over the 3D view with no box behind it
    (putText is most of the overlay cost, so only there)."""
    import cv2  # noqa: PLC0415
    if scale <= 0.05:
        return
    org = (int(org[0]), int(org[1]))
    if outline:
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _shade(img, x0, y0, x1, y1, colour=(0, 0, 0), alpha=0.55) -> None:
    """Blend a translucent box into img in place."""
    import cv2  # noqa: PLC0415
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img.shape[1], int(x1)), min(img.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(roi, 1.0 - alpha, np.full_like(roi, colour), alpha, 0.0, dst=roi)


def _hud(img, frame, sup, L: Layout) -> None:
    import cv2  # noqa: PLC0415
    s = L.s
    x0, y0 = round(8 * s), L.top + round(64 * s)     # under the event label and the fault line
    lh = round(19 * s)
    _shade(img, x0, y0, x0 + round(290 * s), y0 + 5 * lh + round(10 * s), alpha=0.6)
    fs = 0.48 * s
    x, y = x0 + round(8 * s), y0 + round(18 * s)
    _text(img, f"pitch {frame['pitch']:+5.1f} deg (fwd +)", (x, y), fs, _WHITE)
    _text(img, f"roll  {frame['roll']:+5.1f} deg (left +)", (x, y + lh), fs, _WHITE)
    _text(img, f"lean {frame['lean']:4.1f} deg   COM z {frame['com_z']:.3f} m", (x, y + 2 * lh), fs, _WHITE)
    state, sec = frame["hud"]
    msg, col = {"settled": ("settled", _GREEN), "recovered": (f"recovered in {sec:.2f} s", _GREEN),
                "recovering": (f"recovering {sec:.2f} s", _AMBER), "fallen": ("FALLEN", _RED)}[state]
    _text(img, f"|w| {frame['rate']:4.2f} rad/s  {msg}", (x, y + 3 * lh), fs, col)
    _text(img, "feet", (x, y + 4 * lh), fs, _WHITE)
    li = sup.get("left_index", 0)
    c = frame["contacts"]
    for j, (name, idx) in enumerate((("L", li), ("R", 1 - li))):
        cx = x + round((60 + 46 * j) * s)
        on = idx < len(c) and c[idx] > 0.5
        _text(img, name, (cx, y + 4 * lh), fs, _WHITE)
        cv2.circle(img, (cx + round(20 * s), y + 4 * lh - round(5 * s)), max(2, round(6 * s)),
                   _GREEN if on else _GREY, -1 if on else 1, cv2.LINE_AA)


def _timeline(img, frame, L: Layout) -> None:
    import cv2  # noqa: PLC0415
    s, y0 = L.s, L.H - L.tl
    img[y0:, :] = (16, 17, 20)
    T = max(frame["seconds"], frame["t"], 1e-6)
    x0, x1 = round(10 * s), L.W - round(10 * s)
    ym = y0 + L.tl // 2

    def X(t):
        return int(round(x0 + (x1 - x0) * min(max(t / T, 0.0), 1.0)))

    beats = frame["beats"]
    on = [b for b in beats if b["kind"] == "fault_on"]
    off = [b for b in beats if b["kind"] == "fault_off"]
    for a in on:
        z = next((b["t"] for b in off if b["t"] >= a["t"]), T)
        _shade(img, X(a["t"]), ym - round(9 * s), X(z), ym + round(5 * s), _AMBER, 0.35)
    for f in frame["falls"]:       # the 2 s a fall costs: its own physics, reset and settle
        _shade(img, X(f["t"]), ym - round(9 * s), X(f["t"] + 2.0), ym + round(5 * s), (90, 90, 90), 0.5)
    cv2.line(img, (x0, ym), (x1, ym), (70, 72, 80), 1)
    for b in beats:
        col = {"push": _RED, "kick": _CYAN, "start": _WHITE}.get(b["kind"])
        if col is None:
            continue
        x = X(b["t"])
        cv2.line(img, (x, ym - round(10 * s)), (x, ym + round(5 * s)), col, max(1, round(2 * s)))
        short = b["name"].split(" ")[0] + ("N" if b["kind"] == "push" else "")
        if b["kind"] == "push" and "pelvis" in b["name"]:
            short += "p"
        _text(img, short if b["kind"] != "start" else "", (x + round(3 * s), ym - round(8 * s)), 0.36 * s, col)
    for f in frame["falls"]:
        x, r = X(f["t"]), max(2, round(5 * s))
        cv2.line(img, (x - r, ym - r), (x + r, ym + r), _RED, max(1, round(2 * s)), cv2.LINE_AA)
        cv2.line(img, (x - r, ym + r), (x + r, ym - r), _RED, max(1, round(2 * s)), cv2.LINE_AA)
    xp = X(frame["t"])
    cv2.line(img, (xp, y0 + 2), (xp, L.H - 2), _WHITE, max(1, round(2 * s)))
    _text(img, "push", (x0, L.H - round(4 * s)), 0.34 * s, _RED)
    _text(img, "kick", (x0 + round(40 * s), L.H - round(4 * s)), 0.34 * s, _CYAN)
    _text(img, "gyro fault", (x0 + round(78 * s), L.H - round(4 * s)), 0.34 * s, _AMBER)
    _text(img, "X fall", (x0 + round(150 * s), L.H - round(4 * s)), 0.34 * s, _RED)


def compose_frame(main: np.ndarray, side: np.ndarray, frame: dict, sup: dict, header: str,
                  L: Optional[Layout] = None) -> np.ndarray:
    """One RGB video frame from the two rendered views and the driver's frame dict."""
    L = L or Layout.of(main.shape[1] + side.shape[1], main.shape[0] + 68)
    s = L.s
    img = np.zeros((L.H, L.W, 3), np.uint8)
    img[:] = _DARK
    img[L.top:L.top + L.main_h, :L.main_w] = main[:L.main_h, :L.main_w]
    img[L.top:L.top + L.side_h, L.main_w:] = side[:L.side_h, :L.side_w]
    img[L.top + L.side_h:L.top + L.main_h, L.main_w:] = draw_support(sup, frame, L.side_w, L.map_h, s)
    img[L.top + L.side_h:L.top + L.side_h + 1, L.main_w:] = (70, 72, 80)
    img[L.top:L.top + L.main_h, L.main_w:L.main_w + 1] = (70, 72, 80)
    _text(img, "side . from the robot's right", (L.main_w + round(6 * s), L.top + round(15 * s)), 0.4 * s,
          _GREY, outline=True)

    # top bar
    _text(img, f"{header} . t = {frame['t']:5.2f} / {frame['seconds']:.1f} s",
          (round(8 * s), round(19 * s)), 0.52 * s, _WHITE)

    # event label (2 s after a beat) and the persistent fault line, under the top bar
    y = L.top
    if frame.get("label"):
        _shade(img, 0, y, L.main_w, y + round(30 * s), alpha=0.6)
        lab = frame["label"]
        col = _RED if lab.startswith("PUSH") else _CYAN if lab.startswith("KICK") else \
            _AMBER if lab.startswith("FAULT") else _WHITE
        _text(img, lab, (round(10 * s), y + round(21 * s)), 0.6 * s, col, max(1, round(1.5 * s)))
    card = frame.get("end_card")
    if frame.get("fault") and not card:
        _text(img, frame["fault"], (round(10 * s), y + round(52 * s)), 0.5 * s, _AMBER, outline=True)
    if not card:
        _hud(img, frame, sup, L)

    if frame.get("banner"):
        yb = L.top + round(L.main_h * 0.70)
        _shade(img, 0, yb, L.main_w, yb + round(32 * s), (150, 25, 20), 0.85)
        _text(img, frame["banner"], (round(10 * s), yb + round(22 * s)), 0.55 * s, _WHITE)

    if card:
        rows = card
        x0, y0 = round(8 * s), L.top + round(40 * s)
        lh = round(22 * s)
        _shade(img, x0, y0, L.main_w - round(8 * s), y0 + (len(rows) + 3) * lh, alpha=0.8)
        _text(img, "VALIDATION: the fixed showcase timeline", (x0 + round(12 * s), y0 + lh), 0.55 * s, _WHITE)
        for i, r in enumerate(rows):
            col = _RED if "FELL" in r else _GREY if "not reached" in r else _GREEN
            _text(img, r, (x0 + round(12 * s), y0 + (i + 2) * lh), 0.48 * s, col)
        if frame.get("summary"):
            _text(img, frame["summary"], (x0 + round(12 * s), y0 + (len(rows) + 2) * lh + round(4 * s)),
                  0.55 * s, _WHITE)
    _timeline(img, frame, L)
    return img


# ------------------------------------------------------------------------------ encoding

# (extension, fourcc, backend name) in order of preference. H.264 via Media Foundation is the
# clean path on Windows; opencv's bundled ffmpeg has no H.264 encoder (it looks for an
# openh264 DLL), so elsewhere it is VP9/VP8 webm, and mp4v as the last resort (local players
# only: browsers and W&B do not play MPEG-4 Part 2).
WRITERS = [(".mp4", "avc1", "CAP_MSMF"), (".webm", "VP90", "CAP_FFMPEG"),
           (".webm", "VP80", "CAP_FFMPEG"), (".mp4", "mp4v", "CAP_FFMPEG")]


def open_writer(stem, fps: float, w: int, h: int):
    """(cv2.VideoWriter, path) for the first encoder that really writes here, or (None, None).
    Each candidate first writes a 2-frame probe file, which must come out non-empty: a writer
    can report isOpened() and still produce nothing."""
    import cv2  # noqa: PLC0415
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    blank = np.zeros((h, w, 3), np.uint8)
    for ext, cc, api in WRITERS:
        if not hasattr(cv2, api):
            continue
        probe = stem.with_name(stem.name + ".probe" + ext)
        try:
            wr = cv2.VideoWriter(str(probe), getattr(cv2, api), cv2.VideoWriter_fourcc(*cc), fps, (w, h))
            ok = wr.isOpened()
            if ok:
                wr.write(blank)
                wr.write(blank)
            wr.release()
            ok = ok and probe.exists() and probe.stat().st_size > 0
        except cv2.error:
            ok = False
        finally:
            with contextlib.suppress(OSError):
                probe.unlink()
        if not ok:
            continue
        path = stem.with_suffix(ext) if stem.suffix in ("", ".mp4", ".webm") else Path(str(stem) + ext)
        wr = cv2.VideoWriter(str(path), getattr(cv2, api), cv2.VideoWriter_fourcc(*cc), fps, (w, h))
        if wr.isOpened():
            return wr, path
        wr.release()
    return None, None


# ------------------------------------------------------------------------------ top level

_GL_PROBE = "import mujoco; mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'), 8, 8)"


def gl_available(timeout: float = 20.0):
    """(ok, why): can this machine make an offscreen GL context? Tried in a SUBPROCESS (~1 s),
    because a broken GL driver can segfault, and in the training process that would end the
    run before wandb.finish(). Inherits MUJOCO_GL, so it tests what the renderer will use."""
    try:
        r = subprocess.run([sys.executable, "-c", _GL_PROBE], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"GL probe timed out after {timeout:.0f} s"
    except OSError as e:
        return False, f"GL probe could not run ({e})"
    if r.returncode != 0:
        lines = (r.stderr or "").strip().splitlines()
        return False, lines[-1] if lines else f"GL probe exited {r.returncode}"
    return True, ""


def pick_checkpoint(run_root, which: str = "best") -> Path:
    """best/ when the robustness eval made one, else final/; or a named dir ('final',
    'checkpoints/00035.00M')."""
    run_root = Path(run_root)
    if which == "best":
        return run_root / "best" if (run_root / "best" / "model.zip").exists() else run_root / "final"
    return run_root / which


def checkpoint_label(ck) -> str:
    """'best@45.00M (score 74.8)' from its eval.json, else 'final@60.00M (unscored)'."""
    ck = Path(ck)
    ev = ck / "eval.json"
    if ev.exists():
        try:
            d = json.loads(ev.read_text())
            at = f"@{d['step'] / 1e6:.2f}M" if d.get("step") is not None else ""
            return f"{ck.name}{at} (score {d['score']:.1f})"
        except (ValueError, KeyError, TypeError):
            pass
    try:
        import zipfile  # noqa: PLC0415
        steps = int(json.loads(zipfile.ZipFile(ck / "model.zip").read("data"))["num_timesteps"])
        return f"{ck.name}@{steps / 1e6:.2f}M (unscored)"
    except Exception:  # noqa: BLE001 -- a label is not worth failing over
        return f"{ck.name} (unscored)"


@dataclass
class VideoResult:
    path: Path
    json_path: Path
    frames: int
    seconds: float        # of video
    falls: int
    beats_held: int
    beats_graded: int
    wall_s: float


def make_validation_video(policy, env_cfg: dict, out_stem, label: str, width: int = 960,
                          height: int = 540, fps: float = 20, seed: int = 12345,
                          render_every: int = 2, timeline=None, seconds: float = SHOWCASE_SECONDS,
                          progress=print) -> VideoResult:
    """Play the showcase on `policy`, film it to <out_stem>.mp4 (or .webm) and write the event
    log and per-beat outcomes to <out_stem>.json. At 20 fps with render_every=2 it is real time."""
    import cv2  # noqa: PLC0415
    import mujoco  # noqa: PLC0415
    t0 = time.time()
    width, height = int(width) // 2 * 2, int(height) // 2 * 2     # encoders want even sizes
    with contextlib.redirect_stdout(io.StringIO()):                # the env prints its config
        wenv = make_eval_env(env_cfg, seconds=60.0)
    m = wenv.base.env.unwrapped.model
    # Render-only: the offscreen buffer must hold the largest view. No physics effect.
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), width)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), height)
    L = Layout.of(width, height)
    out_stem = Path(out_stem)
    writer, path = open_writer(out_stem, fps, width, height)
    if writer is None:
        raise RuntimeError("no working video encoder (tried " +
                           ", ".join(f"{e}/{c}" for e, c, _ in WRITERS) + ")")
    st = {"frames": 0, "cams": None, "resets": -1}
    data = wenv.base.env.unwrapped.data
    views_r = []          # (renderer, camera name)

    def on_frame(w, fr):
        n_resets = len(fr["falls"]) if fr["phase"] != "falling" else st["resets"]
        if st["cams"] is None or n_resets != st["resets"]:     # fixed at each reset
            st["cams"], st["resets"] = make_cameras(robot_forward(m, data)), n_resets
        aim(st["cams"], data)
        arrows = scene_arrows(data, fr)
        views = []
        for r, name in views_r:
            r.update_scene(data, camera=st["cams"][name])
            for a in arrows:
                _add_arrow(r.scene, *a)
            img = r.render()
            draw_xray(img, mujoco.mjv_averageCamera(r.scene.camera[0], r.scene.camera[1]), arrows, L.s)
            views.append(img)
        img = cv2.cvtColor(compose_frame(views[0], views[1], fr, support_geometry(w), label, L),
                           cv2.COLOR_RGB2BGR)
        for _ in range(max(1, round(fr.get("hold_s", 0.0) * fps))):    # the end card, held
            writer.write(img)
            st["frames"] += 1

    try:
        for name, (h, w) in (("main", (L.main_h, L.main_w)), ("side", (L.side_h, L.side_w))):
            r = mujoco.Renderer(m, height=h, width=w)
            views_r.append((r, name))
            # The CAD meshes are ~760k vertices; the shadow pass draws them all again and was
            # 3/4 of each render (~100 ms -> ~25 ms per view measured 10/25). Look only.
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        res = run_showcase(policy, env_cfg, seed=seed, on_frame=on_frame, render_every=render_every,
                           timeline=timeline, seconds=seconds, wenv=wenv, label=label)
    except BaseException:
        writer.release()
        with contextlib.suppress(OSError):
            path.unlink()                # a half-written video is not the video
        raise
    finally:
        writer.release()
        for r, _ in views_r:
            r.close()
    wall = time.time() - t0
    res.settings.update(video=str(path), frames=st["frames"], fps=fps, size=[width, height],
                        wall_s=round(wall, 2))
    json_path = res.save(out_stem.with_suffix(".json"))
    if progress:
        progress(f"validation video: {st['frames']} frames ({st['frames'] / fps:.1f} s of video) in "
                 f"{wall:.1f} s, {res.summary()} -> {path}")
    return VideoResult(path=path, json_path=json_path, frames=st["frames"], seconds=st["frames"] / fps,
                       falls=len(res.falls), beats_held=res.beats_held, beats_graded=len(res.graded),
                       wall_s=wall)
