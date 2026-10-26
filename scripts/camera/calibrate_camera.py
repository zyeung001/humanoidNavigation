#!/usr/bin/env python3
# calibrate_camera.py  --  dev box only (OpenCV). Once per phone + video mode.
"""Phone video of the printed ChArUco board -> config/camera/<name>.yaml (intrinsics).

WHY. A tag's 3D pose is only as right as the camera model behind it: a focal length off
by 3% puts every link 3% too far away, and an uncorrected wide-angle lens bends a tag near
the frame edge enough to read as a few degrees of tilt. Phones do not publish these
numbers for the video mode, so they are measured once, from the board page that
make_targets.py prints.

    python scripts/camera/calibrate_camera.py board.mp4 --name pixel7_4k60
    python scripts/camera/calibrate_camera.py board_photos/ --name pixel7_4k60

A 30 s video is plenty: wave the board slowly through the whole frame -- centre, all four
corners and edges, near and far, tilted 20-40 deg each way. Frames are chosen for you:
sharp ones, with most of the board visible, spread over as many different board
positions/tilts as possible (farthest-point sampling), because 40 views of the board in
the middle of the frame teach the lens model nothing about the corners.

THE RESULT IS ONLY VALID FOR the same phone, lens, resolution and zoom, with stabilisation
OFF (stabilisation re-warps every frame, so there is no fixed camera to calibrate) and
focus locked where it will be when filming the robot.

The RMS reprojection error is printed; above 1 px something is wrong (blur, a curled
print, autofocus hunting, stabilisation on). The printed square size does not affect the
intrinsics -- a uniform print scale error only rescales the board, which the calibration
absorbs -- but pass the measured --square-mm anyway so the saved board extrinsics are right.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camlib  # noqa: E402
from video_pose import iter_frames  # noqa: E402

RMS_WARN_PX = 1.0
MIN_CORNERS = 12            # of 24 interior corners on the 5 x 7 board
SCAN_FRAMES = 400           # a long video is subsampled to about this many frames before detection
COVERAGE_GRID = (8, 6)      # image cells that should each see a board corner


def make_board(board_cfg, square_mm=None):
    cv2 = camlib.cv2_aruco()
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board_cfg["dictionary"]))
    sq = float(square_mm or board_cfg["square_mm"])
    mk = float(board_cfg["marker_mm"]) * sq / float(board_cfg["square_mm"])
    board = cv2.aruco.CharucoBoard((int(board_cfg["squares_x"]), int(board_cfg["squares_y"])),
                                   sq / 1000.0, mk / 1000.0, d)
    return board, d


def _view_feature(obj, img, size, board_wh):
    """Where the board's four outer corners land in the image, normalised to [0, 1]: one
    8-D point per view, so 'different views' = far apart in this space (position, scale
    and tilt all move these corners)."""
    cv2 = camlib.cv2_aruco()
    Hm, _ = cv2.findHomography(obj[:, :2].astype(np.float64), img.astype(np.float64))
    if Hm is None:
        return None
    w, h = board_wh
    quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
    p = cv2.perspectiveTransform(quad, Hm).reshape(-1, 2) / np.asarray(size, dtype=float)
    return p.ravel()


def select_views(cands, max_frames):
    """Farthest-point sampling over view features, starting from the fullest view."""
    if len(cands) <= max_frames:
        return list(range(len(cands)))
    F = np.array([c["feat"] for c in cands])
    chosen = [int(np.argmax([len(c["obj"]) for c in cands]))]
    dmin = np.linalg.norm(F - F[chosen[0]], axis=1)
    while len(chosen) < max_frames:
        i = int(np.argmax(dmin))
        chosen.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(F - F[i], axis=1))
    return sorted(chosen)


def calibrate(frames, board_cfg, square_mm=None, max_frames=40, min_corners=MIN_CORNERS):
    """Detect the board in every given frame, pick diverse sharp views, calibrate."""
    cv2 = camlib.cv2_aruco()
    board, _ = make_board(board_cfg, square_mm)
    detector = cv2.aruco.CharucoDetector(board)
    sq = float(square_mm or board_cfg["square_mm"]) / 1000.0
    board_wh = (int(board_cfg["squares_x"]) * sq, int(board_cfg["squares_y"]) * sq)
    cands, size, scanned = [], None, 0
    for idx, t, img in frames:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        scanned += 1
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < min_corners:
            continue
        obj, imgp = board.matchImagePoints(ch_corners, ch_ids)
        obj, imgp = obj.reshape(-1, 3), imgp.reshape(-1, 2)
        feat = _view_feature(obj, imgp, size, board_wh)
        if feat is None:
            continue
        cands.append({"frame": idx, "t": t, "obj": obj, "img": imgp, "feat": feat,
                      "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var())})
    if len(cands) < 6:
        raise SystemExit(f"board found in only {len(cands)} of {scanned} frames (need >= 6 with "
                         f">= {min_corners} corners). Better light, slower motion, closer board?")
    med = np.median([c["sharp"] for c in cands])
    sharp = [c for c in cands if c["sharp"] >= 0.5 * med]
    pool = sharp if len(sharp) >= 6 else cands
    views = [pool[i] for i in select_views(pool, max_frames)]

    def run(vs):
        objs = [v["obj"].astype(np.float32) for v in vs]
        imgs = [v["img"].astype(np.float32) for v in vs]
        out = cv2.calibrateCameraExtended(objs, imgs, size, None, None)
        rms, K, dist = out[0], out[1], out[2]
        per_view = np.ravel(out[7])
        return float(rms), K, dist, per_view

    rms, K, dist, per_view = run(views)
    # One outlier pass: a view much worse than the rest is a misdetection or a blurred frame.
    bad = per_view > max(2.5 * float(np.median(per_view)), 1.0)
    dropped = [views[i]["frame"] for i in np.flatnonzero(bad)]
    if bad.any() and (~bad).sum() >= 8:
        views = [v for v, b in zip(views, bad) if not b]
        rms, K, dist, per_view = run(views)
    else:
        dropped = []

    gx, gy = COVERAGE_GRID
    cells = set()
    for v in views:
        for x, y in v["img"]:
            cells.add((min(int(x / size[0] * gx), gx - 1), min(int(y / size[1] * gy), gy - 1)))
    return {"image_size": [int(size[0]), int(size[1])],
            "camera_matrix": np.asarray(K).round(4).tolist(),
            "dist_coeffs": np.ravel(dist).round(6).tolist(),
            "rms_px": round(rms, 4),
            "per_view_px": [round(float(e), 3) for e in per_view],
            "frames_used": [int(v["frame"]) for v in views],
            "frames_dropped_outlier": [int(f) for f in dropped],
            "n_frames_scanned": scanned, "n_frames_with_board": len(cands),
            "coverage": round(len(cells) / (gx * gy), 3),
            "board": {**board_cfg, "square_mm": float(square_mm or board_cfg["square_mm"])}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", help="video of the board, or a folder of photos of it")
    ap.add_argument("--name", default="phone",
                    help="camera name -> config/camera/<name>.yaml (e.g. pixel7_4k60: phone + mode)")
    ap.add_argument("--tags", default=str(camlib.DEFAULT_TAGS), help="holds calibration_board")
    ap.add_argument("--square-mm", type=float, default=None, help="measured printed square size")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=None, help="default: scan ~400 frames of a video")
    ap.add_argument("--out", default=None, help="default: config/camera/<name>.yaml")
    a = ap.parse_args(argv)

    cfg = camlib.load_tag_config(a.tags)
    src = Path(a.source)
    stride = a.stride
    if stride is None:
        stride = 1
        if src.is_file():
            cv2 = camlib.cv2_aruco()
            cap = cv2.VideoCapture(str(src))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            stride = max(1, n // SCAN_FRAMES)
    print(f"{src}: scanning every {stride} frame(s) for the {cfg.board['squares_x']}x"
          f"{cfg.board['squares_y']} ChArUco board ...")
    # A photo folder has no frame rate and needs none: times are only labels here.
    cal = calibrate(iter_frames(src, fps=30.0 if src.is_dir() else None, stride=stride),
                    cfg.board, a.square_mm, a.max_frames)

    out = Path(a.out) if a.out else camlib.CAMERA_DIR / f"{a.name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"camera_name": a.name, "source": str(src), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "opencv": camlib.cv2_aruco().__version__, **cal}
    header = ("# Camera intrinsics from scripts/camera/calibrate_camera.py. Valid only for this phone,\n"
              "# lens, resolution and zoom, stabilisation OFF, focus locked. dist = k1 k2 p1 p2 k3.\n")
    out.write_text(header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))

    K = np.array(cal["camera_matrix"])
    print(f"\nused {len(cal['frames_used'])} of {cal['n_frames_with_board']} board frames "
          f"({cal['n_frames_scanned']} scanned); dropped as outliers: {cal['frames_dropped_outlier'] or 'none'}")
    print(f"image {cal['image_size'][0]}x{cal['image_size'][1]}  fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  "
          f"cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}  dist {cal['dist_coeffs']}")
    print(f"RMS reprojection error {cal['rms_px']:.3f} px   image coverage {100 * cal['coverage']:.0f}%")
    if cal["rms_px"] > RMS_WARN_PX:
        print(f"WARNING: RMS > {RMS_WARN_PX} px -- blur, a curled print, autofocus or stabilisation? Redo it.")
    if cal["coverage"] < 0.5:
        print("WARNING: the board never reached much of the frame; sweep it into the corners and edges too.")
    if len(cal["frames_used"]) < 10:
        print("WARNING: fewer than 10 usable views; film longer, with more tilt variety.")
    print(f"wrote {out}")
    return cal


if __name__ == "__main__":
    main()
