#!/usr/bin/env python3
# video_pose.py  --  dev box only (OpenCV). Never needed on the Pi.
"""Phone video of the tagged robot -> every link's pose over time, in the floor's frame.

WHY. The robot reports one pelvis IMU and its encoders. Neither can say where the body
actually bent (10/25: ~75% of pelvis pitch never appeared in the encoders -- it was
flex and servo play), whether a foot rolled, lifted or slid, or which way a fall twisted.
A camera watching printed tags on each link measures all of that independently, so a
hardware log can be checked against what physically happened instead of argued about.

    python scripts/camera/video_pose.py run.mp4 --camera config/camera/pixel7_4k60.yaml
    python scripts/camera/video_pose.py frames_dir/ --camera ... --fps 60   # image folder

WRITES, next to the video (or --out-dir):
  <stem>.tags.csv   one row per frame per configured tag:
      missing        1 = not detected in this frame (pose columns blank)
      blurred        sharpness below 40% of that tag's median (motion blur: trust less)
      ambiguous      both IPPE poses fit; the choice came from continuity / the upright prior
      jump           rotated faster than JUMP_DEG_S since its last detection: likely a flip
      px_cx, px_cy   tag centre in the image (sync.py uses these for the tap)
      reproj_px      RMS corner reprojection error of the chosen pose; alt_reproj_px = the other
      cam_*          the TAG's pose in the CAMERA frame (x right, y down, z out of the lens), m / deg
      world_*        the LINK's pose in WORLD implied by this tag (mount from tags.yaml), m / deg
  <stem>.links.csv  one row per frame: per link x/y/z (m) roll/pitch/yaw (deg) in WORLD, the tag
                    used and its error; per joint pair (tags.yaml `pairs`) the child's rotation in
                    the parent's frame, in the joint chain's axis order, comparable with encoders.
  <stem>.pose.json  detection rate, error and flag counts per tag, and the floor-pose stability.

THE WORLD FRAME is the floor tag's frame (z up), estimated ONCE from every frame that sees
it: the camera is on a tripod, so averaging is far less noisy than re-solving each frame.
The spread of the per-frame floor poses is reported; a large spread means the camera moved
and the world-frame numbers are suspect (relative pair angles are still fine -- they never
use the world).

ANGLES: roll/pitch/yaw intrinsic Z-Y-X (camlib.py), pitch > 0 = leaning forward, the same
sign as pg_x in the frame log. Time: t_video, seconds from the first frame of the file
(container timestamps when the decoder gives them, else frame / fps); sync.py maps it onto
the log's t_wall.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camlib  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
GAP_S = 0.5             # continuity across a dropout shorter than this; longer -> re-initialise
JUMP_DEG_S = 600.0      # a link turning faster than this between detections is a pose flip, not motion
BLUR_FRAC = 0.4         # sharpness below this fraction of the tag's median -> blurred
WORLD_WARN_DEG = 1.0    # floor-pose spread (p95) past these -> the camera probably moved
WORLD_WARN_MM = 10.0
# World up in the CAMERA frame when the floor tag is never seen: the phone stands upright on
# a tripod, so image-up (camera -y) is within a tilt of true up -- far better than nothing.
CAMERA_UP = np.array([0.0, -1.0, 0.0])

TAG_COLUMNS = ["frame", "t_video", "tag_id", "link", "missing", "blurred", "ambiguous", "jump",
               "px_cx", "px_cy", "px_size", "sharpness", "reproj_px", "alt_reproj_px",
               "cam_x", "cam_y", "cam_z", "cam_roll", "cam_pitch", "cam_yaw",
               "world_x", "world_y", "world_z", "world_roll", "world_pitch", "world_yaw"]
LINK_FIELDS = ["x", "y", "z", "roll", "pitch", "yaw", "tag", "reproj_px"]
PAIR_FIELDS = ["roll", "pitch", "yaw"]
_AXIS_NAME = {"x": "roll", "y": "pitch", "z": "yaw"}


# ---------------------------------------------------------------- frames

def iter_frames(path, fps=None, start=0.0, end=None, stride=1):
    """Yield (frame_index, t_video, bgr) from a video file or a folder of images.

    Phone video is often variable frame rate, so the container timestamp is used when the
    decoder provides a sane one; otherwise frame / fps.
    """
    cv2 = camlib.cv2_aruco()
    path = Path(path)
    stride = max(1, int(stride))
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise SystemExit(f"{path}: no images ({', '.join(sorted(IMAGE_EXTS))})")
        rate = float(fps or 30.0)
        if not fps:
            print(f"WARNING: no --fps for an image folder; assuming {rate:g}")
        for i, f in enumerate(files):
            t = i / rate
            if t < start or i % stride:
                continue
            if end is not None and t > end:
                break
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                print(f"WARNING: unreadable image {f}")
                continue
            yield i, t, img
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"{path}: cannot open video")
    rate = float(fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    i, last_t = -1, -1.0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            i += 1
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            t = pos if (fps is None and pos > last_t and (pos > 0 or i == 0)) else i / rate
            last_t = t
            if t < start or i % stride:
                continue
            if end is not None and t > end:
                break
            yield i, t, img
    finally:
        cap.release()


# ---------------------------------------------------------------- detection

@dataclass
class Detection:
    frame: int
    t: float
    tag: int
    corners: np.ndarray         # (4, 2) px, cv2.aruco order
    cands: list                 # camlib.PoseCandidate, best error first
    sharp: float
    choice: int = 0
    ambiguous: bool = False
    jump: bool = False
    blurred: bool = False


def detect(frames, cfg, K, dist, progress_every=600):
    """Run the detector over every frame. Returns (frame list, detections, image size, unknown ids)."""
    cv2 = camlib.cv2_aruco()
    detector = camlib.make_detector(cfg.dictionary)
    frame_list, dets, unknown = [], [], {}
    size = None
    for n, (idx, t, img) in enumerate(frames):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        frame_list.append((idx, t))
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            for c, tid in zip(corners, np.ravel(ids)):
                tid = int(tid)
                tag = cfg.tags.get(tid)
                if tag is None:
                    unknown[tid] = unknown.get(tid, 0) + 1
                    continue
                c, _ = camlib.refine_corners(gray, np.asarray(c, dtype=np.float64).reshape(4, 2))
                cands = camlib.solve_tag(c, tag.size, K, dist)
                if cands:
                    dets.append(Detection(idx, t, tid, c, cands, camlib.sharpness(gray, c)))
        if progress_every and n and n % progress_every == 0:
            print(f"  {n} frames, t={t:.1f} s, {len(dets)} detections")
    return frame_list, dets, size, unknown


# ---------------------------------------------------------------- world + choice

def _T(c):
    return camlib.make_T(c.R, c.t)


def resolve(dets, cfg, prior_up=None):
    """Choose one IPPE pose per detection with temporal continuity; set ambiguous/jump."""
    by_tag = {}
    for d in dets:
        by_tag.setdefault(d.tag, []).append(d)
    for tid, ds in by_tag.items():
        ds.sort(key=lambda d: d.frame)
        R_lt = cfg.tags[tid].T_link_tag[:3, :3]
        is_world = tid == cfg.world_tag
        prev = None
        for d in ds:
            prev_R = prev.cands[prev.choice].R if prev is not None and d.t - prev.t <= GAP_S else None
            # The floor tag has no "upright link" prior: its normal is world up by definition.
            up = None if is_world else prior_up
            d.choice, d.ambiguous = camlib.choose_pose(d.cands, prev_R=prev_R, prior_up=up, R_link_tag=R_lt)
            if prev_R is not None and d.t > prev.t:
                rate = math.degrees(camlib.rotation_angle(prev_R, d.cands[d.choice].R)) / (d.t - prev.t)
                d.jump = rate > JUMP_DEG_S
                if d.jump and d.ambiguous and up is not None:
                    # Continuity just failed: either the link really spun, or the track has
                    # been on the flipped branch and this frame's candidates are both far
                    # from it. Let the upright prior re-pick, so one bad start does not lock
                    # the whole track onto the wrong tilt (jump stays flagged either way).
                    d.choice, _ = camlib.choose_pose(d.cands, prior_up=up, R_link_tag=R_lt)
            prev = d
        sharp = np.array([d.sharp for d in ds], dtype=float)
        med = np.nanmedian(sharp) if np.isfinite(sharp).any() else float("nan")
        for d in ds:
            d.blurred = bool(np.isfinite(med) and np.isfinite(d.sharp) and d.sharp < BLUR_FRAC * med)


@dataclass
class World:
    T_cam_world: np.ndarray
    n_frames: int
    rot_p95_deg: float
    trans_p95_mm: float


def estimate_world(dets, cfg):
    """One static camera->world transform from every sighting of the floor tag."""
    if cfg.world_tag is None:
        return None
    ws = [d for d in dets if d.tag == cfg.world_tag]
    if not ws:
        return None
    resolve(ws, cfg)
    T_world_tag = cfg.tags[cfg.world_tag].T_link_tag
    Ts = [_T(d.cands[d.choice]) @ camlib.inv_T(T_world_tag) for d in ws if not d.blurred] or \
         [_T(d.cands[d.choice]) @ camlib.inv_T(T_world_tag) for d in ws]
    Rs = np.array([T[:3, :3] for T in Ts])
    ts = np.array([T[:3, 3] for T in Ts])
    R0 = camlib.mean_rotation(Rs)
    ang = np.array([camlib.rotation_angle(R0, R) for R in Rs])
    keep = ang <= max(3.0 * np.median(ang), math.radians(0.5))
    R = camlib.mean_rotation(Rs[keep])
    t = np.median(ts[keep], axis=0)
    ang = np.degrees([camlib.rotation_angle(R, Ri) for Ri in Rs])
    dt = np.linalg.norm(ts - t, axis=1) * 1000.0
    return World(camlib.make_T(R, t), len(Ts), float(np.percentile(ang, 95)), float(np.percentile(dt, 95)))


# ---------------------------------------------------------------- tables

def _rpy_deg(R):
    return [math.degrees(a) for a in camlib.matrix_to_rpy(R)]


def build_tables(frame_list, dets, cfg, world):
    """tags.csv rows (one per frame per tag) and links.csv rows (one per frame)."""
    T_world_cam = camlib.inv_T(world.T_cam_world) if world is not None else np.eye(4)
    by_frame = {}
    for d in dets:
        by_frame.setdefault(d.frame, {})[d.tag] = d
    links = cfg.links
    tag_rows, link_rows = [], []
    for idx, t in frame_list:
        fd = by_frame.get(idx, {})
        best = {}                                # link -> (detection, T_cam_link)
        for tid in sorted(cfg.tags):
            tag = cfg.tags[tid]
            d = fd.get(tid)
            row = {"frame": idx, "t_video": t, "tag_id": tid, "link": tag.link, "missing": int(d is None)}
            if d is not None:
                c = d.cands[d.choice]
                T_cam_tag = _T(c)
                T_cam_link = T_cam_tag @ camlib.inv_T(tag.T_link_tag)
                T_w = T_world_cam @ T_cam_link
                ctr = d.corners.mean(axis=0)
                edge = np.mean(np.linalg.norm(d.corners - np.roll(d.corners, 1, axis=0), axis=1))
                alt = [k.err for i, k in enumerate(d.cands) if i != d.choice]
                row.update(blurred=int(d.blurred), ambiguous=int(d.ambiguous), jump=int(d.jump),
                           px_cx=ctr[0], px_cy=ctr[1], px_size=edge, sharpness=d.sharp,
                           reproj_px=c.err, alt_reproj_px=alt[0] if alt else float("nan"),
                           cam_x=c.t[0], cam_y=c.t[1], cam_z=c.t[2])
                row.update(zip(["cam_roll", "cam_pitch", "cam_yaw"], _rpy_deg(c.R)))
                row.update(world_x=T_w[0, 3], world_y=T_w[1, 3], world_z=T_w[2, 3])
                row.update(zip(["world_roll", "world_pitch", "world_yaw"], _rpy_deg(T_w[:3, :3])))
                if tag.link != "world":
                    # Fuse a link's tags by trust: sharp before blurred, then lowest error.
                    key = (d.blurred or d.jump, c.err)
                    if tag.link not in best or key < best[tag.link][2]:
                        best[tag.link] = (d, T_cam_link, key)
            tag_rows.append(row)
        lrow = {"frame": idx, "t_video": t, "world_ref": int(world is not None)}
        Rcam = {}
        for link in links:
            if link not in best:
                continue
            d, T_cam_link, _ = best[link]
            Rcam[link] = T_cam_link[:3, :3]
            T_w = T_world_cam @ T_cam_link
            r, p, y = _rpy_deg(T_w[:3, :3])
            lrow.update({f"{link}.x": T_w[0, 3], f"{link}.y": T_w[1, 3], f"{link}.z": T_w[2, 3],
                         f"{link}.roll": r, f"{link}.pitch": p, f"{link}.yaw": y,
                         f"{link}.tag": d.tag, f"{link}.reproj_px": d.cands[d.choice].err})
        for pr in cfg.pairs:
            if pr.parent in Rcam and pr.child in Rcam:
                e = camlib.euler_from_matrix(Rcam[pr.parent].T @ Rcam[pr.child], pr.order)
                for ax, v in e.items():
                    lrow[f"{pr.name}.{_AXIS_NAME[ax]}"] = math.degrees(v)
        link_rows.append(lrow)
    return tag_rows, link_rows


def link_columns(cfg):
    cols = ["frame", "t_video", "world_ref"]
    for link in cfg.links:
        cols += [f"{link}.{f}" for f in LINK_FIELDS]
    for pr in cfg.pairs:
        cols += [f"{pr.name}.{f}" for f in PAIR_FIELDS]
    return cols


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(v) else f"{v:.6g}"
    return v


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            # Time keeps fixed microseconds: 6 significant figures would round t_video to
            # 10 ms after 1000 s, coarser than a 60 fps frame.
            w.writerow([f"{r[c]:.6f}" if c == "t_video" else _fmt(r.get(c)) for c in columns])


# ---------------------------------------------------------------- driver

@dataclass
class Result:
    tag_rows: list
    link_rows: list
    summary: dict
    world: World | None


def process(frames, cfg, K, dist):
    """The whole pipeline on any (frame, t, image) iterable -- the CLI and the tests share it."""
    frame_list, dets, size, unknown = detect(frames, cfg, K, dist)
    world = estimate_world(dets, cfg)
    # Without a floor tag the first-frame IPPE choice would fall back to the lower error,
    # which for a near-frontal 32 mm tag differs by ~0.02-0.05 px: a coin flip that
    # continuity then keeps. Image-up is a good enough "up" to break it.
    prior_up = world.T_cam_world[:3, 2] if world is not None else CAMERA_UP
    body = [d for d in dets if d.tag != cfg.world_tag]
    resolve(body, cfg, prior_up=prior_up)
    tag_rows, link_rows = build_tables(frame_list, dets, cfg, world)

    n = len(frame_list)
    per_tag = {}
    for tid, tag in sorted(cfg.tags.items()):
        ds = [d for d in dets if d.tag == tid]
        errs = [d.cands[d.choice].err for d in ds]
        per_tag[tid] = {"link": tag.link, "detected_frac": len(ds) / n if n else 0.0,
                        "median_reproj_px": float(np.median(errs)) if errs else None,
                        "blurred": sum(d.blurred for d in ds), "ambiguous": sum(d.ambiguous for d in ds),
                        "jumps": sum(d.jump for d in ds)}
    ts = [t for _, t in frame_list]
    summary = {"frames": n, "image_size": size,
               "fps_est": (n - 1) / (ts[-1] - ts[0]) if n > 1 and ts[-1] > ts[0] else None,
               "t_video": [ts[0], ts[-1]] if ts else None,
               "world": None if world is None else {
                   "frames": world.n_frames, "rot_p95_deg": world.rot_p95_deg,
                   "trans_p95_mm": world.trans_p95_mm,
                   "T_cam_world": world.T_cam_world.round(6).tolist()},
               "tags": per_tag, "unknown_ids": unknown}
    return Result(tag_rows, link_rows, summary, world)


def print_summary(s, cfg):
    print(f"\n{s['frames']} frames, {s['image_size']}, ~{(s['fps_est'] or 0):.1f} fps")
    w = s["world"]
    if w is None:
        print("WORLD: floor tag never seen -> links.csv is in the CAMERA frame (world_ref=0); "
              "pair angles are still valid")
    else:
        flag = "  <- CAMERA MOVED? world numbers suspect" if (
            w["rot_p95_deg"] > WORLD_WARN_DEG or w["trans_p95_mm"] > WORLD_WARN_MM) else ""
        print(f"WORLD: floor tag in {w['frames']} frames, spread p95 {w['rot_p95_deg']:.2f} deg / "
              f"{w['trans_p95_mm']:.1f} mm{flag}")
    print(f"{'tag':>4} {'link':<8} {'seen':>6} {'err px':>7} {'blur':>5} {'ambig':>6} {'jump':>5}  where")
    for tid, r in s["tags"].items():
        e = "-" if r["median_reproj_px"] is None else f"{r['median_reproj_px']:.2f}"
        print(f"{tid:>4} {r['link']:<8} {100 * r['detected_frac']:5.1f}% {e:>7} {r['blurred']:>5} "
              f"{r['ambiguous']:>6} {r['jumps']:>5}  {cfg.tags[tid].where[:60]}")
    if s["unknown_ids"]:
        print(f"ids seen but not in tags.yaml (ignored): {s['unknown_ids']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", help="video file, or a folder of images (then give --fps)")
    ap.add_argument("--camera", required=True, help="intrinsics YAML from calibrate_camera.py")
    ap.add_argument("--tags", default=str(camlib.DEFAULT_TAGS))
    ap.add_argument("--fps", type=float, default=None, help="image folders; overrides container timestamps")
    ap.add_argument("--start", type=float, default=0.0, help="skip video before this time (s)")
    ap.add_argument("--end", type=float, default=None, help="stop at this video time (s)")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--out-dir", default=None, help="default: next to the video")
    a = ap.parse_args(argv)

    cfg = camlib.load_tag_config(a.tags)
    src = Path(a.video)
    frames = iter_frames(src, fps=a.fps, start=a.start, end=a.end, stride=a.stride)
    first = next(frames, None)
    if first is None:
        raise SystemExit(f"{src}: no frames")
    h, w = first[2].shape[:2]
    K, dist, _ = camlib.load_camera(a.camera, image_size=(w, h))

    def chained():
        yield first
        yield from frames

    print(f"{src}: {w}x{h}, tags {sorted(cfg.tags)} ({cfg.dictionary})")
    res = process(chained(), cfg, K, dist)
    out_dir = Path(a.out_dir) if a.out_dir else (src if src.is_dir() else src.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.name if src.is_dir() else src.stem
    tags_path, links_path = out_dir / f"{stem}.tags.csv", out_dir / f"{stem}.links.csv"
    write_csv(tags_path, res.tag_rows, TAG_COLUMNS)
    write_csv(links_path, res.link_rows, link_columns(cfg))
    res.summary.update(video=str(src), camera=str(a.camera), tags_yaml=str(cfg.path))
    (out_dir / f"{stem}.pose.json").write_text(json.dumps(res.summary, indent=2, default=str))
    print_summary(res.summary, cfg)
    print(f"\nwrote {tags_path}\n      {links_path}\n      {out_dir / f'{stem}.pose.json'}")
    return res


if __name__ == "__main__":
    main()
