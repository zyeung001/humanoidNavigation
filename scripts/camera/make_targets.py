#!/usr/bin/env python3
# make_targets.py  --  dev box only. Writes the printable tag sheets.
"""tags.yaml -> printable PDFs (A4 and US Letter): every tag at its exact physical size,
labelled with id / link / face, a 100 mm scale bar on every page, and the ChArUco
calibration board.

WHY EXACT SIZE MATTERS. Pose from a square tag scales with the size it is told: a tag
printed 3% small reads 3% farther away (angles are unaffected). Printers and PDF viewers
love to "fit to page", which silently shrinks everything ~3-6%. So every page carries a
100 mm bar: measure it with a ruler; if it is not 100 mm, reprint at 100% / "Actual size".

Everything is drawn as vector rectangles in millimetres -- no raster scaling, so the
cell edges are as sharp as the printer allows.

    python scripts/camera/make_targets.py                    # -> data/camera/camera_targets_{A4,Letter}.pdf
    python scripts/camera/make_targets.py --copies 2         # spares for re-mounting
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camlib  # noqa: E402

PAPER_MM = {"A4": (210.0, 297.0), "Letter": (215.9, 279.4)}
MARGIN = 14.0
GAP = 6.0
LABEL_H = 13.0
SEAM_MM = 0.03          # neighbouring cells overlap by this much so viewers draw no hairline seams
DEFAULT_OUT = camlib.ROOT / "data" / "camera"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def draw_bitmap(ax, bmp, x0, y_top, cell_mm):
    """Black pixels of a 0/255 bitmap (row 0 = top) as merged vector rectangles."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    black = np.asarray(bmp) < 128
    rects, open_runs = [], {}
    for r in range(black.shape[0] + 1):
        runs = set()
        if r < black.shape[0]:
            row = np.concatenate([[False], black[r], [False]])
            d = np.diff(row.astype(int))
            runs = set(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
        # Close runs that do not continue into this row (vertical merge of identical runs).
        for run in list(open_runs):
            if run not in runs:
                r0 = open_runs.pop(run)
                rects.append((run[0], r0, run[1] - run[0], r - r0))
        for run in runs:
            open_runs.setdefault(run, r)
    patches = [Rectangle((x0 + c * cell_mm - SEAM_MM, y_top - (r + h) * cell_mm - SEAM_MM),
                         w * cell_mm + 2 * SEAM_MM, h * cell_mm + 2 * SEAM_MM)
               for c, r, w, h in rects]
    ax.add_collection(PatchCollection(patches, facecolor="black", edgecolor="none", linewidth=0))


def tag_bitmap(dictionary, tag_id):
    """8 x 8 cells: the 6 x 6 code plus its black border, one pixel per cell."""
    cv2 = camlib.cv2_aruco()
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    return cv2.aruco.generateImageMarker(d, int(tag_id), 8, borderBits=1)


def scale_bar(ax, x0, y0):
    ax.plot([x0, x0 + 100], [y0, y0], color="black", lw=1.2, solid_capstyle="butt")
    for k in range(11):
        h = 3.0 if k % 5 == 0 else 1.6
        ax.plot([x0 + 10 * k] * 2, [y0, y0 + h], color="black", lw=0.6)
    ax.text(x0, y0 - 2.5, "0", fontsize=6, ha="center", va="top")
    ax.text(x0 + 100, y0 - 2.5, "100 mm", fontsize=6, ha="center", va="top")
    ax.text(x0 + 104, y0, "<- must measure exactly 100 mm. If not: reprint at 100% / Actual size,\n"
            "   no 'fit to page'. Tag positions scale with print size.", fontsize=6, va="center")


def new_page(plt, paper, title):
    w, h = PAPER_MM[paper]
    fig = plt.figure(figsize=(w / 25.4, h / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(MARGIN, h - MARGIN + 4, title, fontsize=9, weight="bold", va="top")
    scale_bar(ax, MARGIN, MARGIN)
    return fig, ax, w, h


def draw_tag(ax, cfg, tag, x0, y_top):
    """Tag (black square + one-cell white border, dotted cut line) with its label below."""
    cell = tag.size * 1000.0 / 8.0
    foot = 10 * cell
    ax.add_patch(_rect(x0, y_top - foot, foot, foot, ls=":", ec="0.6"))
    draw_bitmap(ax, tag_bitmap(cfg.dictionary, tag.id), x0 + cell, y_top - cell, cell)
    ax.text(x0, y_top - foot - 1.5, f"id {tag.id}  {tag.link}  ({tag.face} face)",
            fontsize=6.5, weight="bold", va="top")
    where = "\n".join(textwrap.wrap(f"{tag.size * 1000:g} mm. {tag.where}", max(24, int(foot / 1.35))))
    ax.text(x0, y_top - foot - 5.0, where, fontsize=5, va="top", color="0.25")


def _rect(x, y, w, h, ls="-", ec="black"):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fill=False, ls=ls, ec=ec, lw=0.4)


def floor_page(plt, cfg, paper):
    tag = cfg.tags[cfg.world_tag]
    fig, ax, w, h = new_page(plt, paper, f"FLOOR tag (id {tag.id}) - lay flat, arrow = the way the robot faces")
    foot = tag.size * 1000.0 * 10 / 8
    x0, y_top = (w - foot) / 2, h - MARGIN - 12
    draw_tag(ax, cfg, tag, x0, y_top)
    ya = y_top - foot - 26
    ax.annotate("", xy=(x0 + foot, ya), xytext=(x0, ya),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black", mutation_scale=14))
    ax.text(x0 + foot / 2, ya + 2, "ROBOT FACES THIS WAY (world +x)", fontsize=8, ha="center", va="bottom")
    ax.text(x0 + foot / 2, ya - 3, "label edge = robot's RIGHT side, top edge = robot's LEFT side",
            fontsize=6, ha="center", va="top")
    return fig


def body_pages(plt, cfg, paper, copies):
    tags = [t for tid, t in sorted(cfg.tags.items()) if tid != cfg.world_tag] * max(1, copies)
    tags.sort(key=lambda t: (-t.size, t.id))       # big ones first packs a page tighter
    figs, fig, ax = [], None, None
    x = y_top = row_h = 0.0
    w = PAPER_MM[paper][0]
    for tag in tags:
        foot = tag.size * 1000.0 * 10 / 8
        box_w = max(foot, 40.0)
        if fig is not None and x + box_w > w - MARGIN:
            x, y_top, row_h = MARGIN, y_top - row_h - GAP, 0.0
        if fig is None or y_top - foot - LABEL_H < MARGIN + 12:
            fig, ax, w, h = new_page(plt, paper, f"BODY tags ({cfg.dictionary})\nKeep the white border. "
                                                 "Glue flat on a stiff card. Label edge toward the floor.")
            figs.append(fig)
            x, y_top, row_h = MARGIN, h - MARGIN - 14, 0.0
        draw_tag(ax, cfg, tag, x, y_top)
        x += box_w + GAP
        row_h = max(row_h, foot + LABEL_H)
    return figs


def board_page(plt, cfg, paper):
    from calibrate_camera import make_board
    b = cfg.board
    board, _ = make_board(b)
    px_per_mm = 2
    nx, ny, sq = int(b["squares_x"]), int(b["squares_y"]), float(b["square_mm"])
    img = board.generateImage((int(nx * sq * px_per_mm), int(ny * sq * px_per_mm)), marginSize=0, borderBits=1)
    fig, ax, w, h = new_page(plt, paper, f"CAMERA CALIBRATION board: ChArUco {nx}x{ny}, {sq:g} mm squares, "
                                         f"{b['marker_mm']:g} mm markers, {b['dictionary']}")
    bw, bh = nx * sq, ny * sq
    x0, y_top = (w - bw) / 2, h - MARGIN - 14
    draw_bitmap(ax, img, x0, y_top, 1.0 / px_per_mm)
    ax.add_patch(_rect(x0, y_top - bh, bw, bh, ls=":", ec="0.6"))
    ax.text(w / 2, y_top - bh - 4,
            "Glue flat to stiff board (no curl). Measure one square: if it is not "
            f"{sq:g} mm, pass --square-mm <measured>.\n"
            "Film 30 s: sweep slowly through the centre, all corners and edges, near and far, "
            "tilted 20-40 deg each way.", fontsize=6.5, ha="center", va="top")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", default=str(camlib.DEFAULT_TAGS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--paper", nargs="+", default=list(PAPER_MM), choices=list(PAPER_MM))
    ap.add_argument("--copies", type=int, default=1, help="copies of each body tag (spares)")
    a = ap.parse_args(argv)

    plt = _plt()
    from matplotlib.backends.backend_pdf import PdfPages
    cfg = camlib.load_tag_config(a.tags)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for paper in a.paper:
        path = out_dir / f"camera_targets_{paper}.pdf"
        figs = []
        if cfg.world_tag is not None:
            figs.append(floor_page(plt, cfg, paper))
        figs += body_pages(plt, cfg, paper, a.copies)
        if cfg.board:
            figs.append(board_page(plt, cfg, paper))
        with PdfPages(path) as pdf:
            for f in figs:
                pdf.savefig(f)
                plt.close(f)
        paths.append(path)
        print(f"wrote {path} ({len(figs)} pages)")
    print("Print at 100% / 'Actual size', then check every page's 100 mm bar with a ruler.")
    return paths


if __name__ == "__main__":
    main()
