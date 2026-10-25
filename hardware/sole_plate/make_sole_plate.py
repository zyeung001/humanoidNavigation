#!/usr/bin/env python3
"""Sole plate for the current robot's feet: 120 x 80 mm foot -> 160 x 80 mm, +20 mm at toe and heel.

Why: on 12 cm feet every controller tried stalls at ~75 on the robustness suite; with 16 cm feet,
holding the 'knee2' or 'flat' stance (config/stances/) scores 90.7 / 90.8 on held-out episodes
(+21.4 +- 2.0 / +21.3 +- 2.5) and the smallest shove that topples it rises ~38%. The sim model of
the plated robot is models/humanoid_real_v2_compliant_soleplate.xml.

Shape: a flat base with a low stop across each end, so the foot drops in between them and cannot
slide toe-to-heel. Sides are open, so the plate is exactly as wide as the foot.

    python hardware/sole_plate/make_sole_plate.py          # writes sole_plate_160x80.stl next to it

Print TWO (left and right are the same part):
  - PETG, flat side down on the bed, no supports. 100% infill (or 5 walls + 50% gyroid).
  - Fix with thin, strong double-sided tape (VHB-type, ~1 mm) over the whole pocket. Line the
    foot's toe and heel up against the stops and its sides flush with the plate's sides.
  - If anything sticks out under the foot (sensor pads, screw heads), keep the tape around it so the
    foot still sits flat, not rocking on it.
After fitting, re-run the same pendulum test as before the plates.
"""
import struct
from pathlib import Path

import numpy as np

FOOT_L, FOOT_W = 120.0, 80.0      # measured on the robot 10/18 (matches the sim's foot box)
EXT = 20.0                        # added at each end
BASE_T = 3.0                      # plate under the foot
STOP_H, STOP_W = 2.0, 3.0         # end stops: height above the base, thickness
CLEAR = 0.5                       # per end, so the foot drops in

PLATE_L = FOOT_L + 2 * EXT
OUT = Path(__file__).with_name(f"sole_plate_{PLATE_L:.0f}x{FOOT_W:.0f}.stl")


def profile():
    """Side profile in (x, z), counter-clockwise; x along the foot, z up."""
    h, p = PLATE_L / 2, FOOT_L / 2 + CLEAR
    top = BASE_T + STOP_H
    return np.array([(-h, 0), (h, 0), (h, BASE_T), (p + STOP_W, BASE_T), (p + STOP_W, top), (p, top),
                     (p, BASE_T), (-p, BASE_T), (-p, top), (-p - STOP_W, top), (-p - STOP_W, BASE_T),
                     (-h, BASE_T)], dtype=float)


def ear_clip(poly):
    """Triangulate a simple counter-clockwise polygon; returns index triples."""
    idx, tris = list(range(len(poly))), []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    while len(idx) > 3:
        for k in range(len(idx)):
            i, j, m = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i], poly[j], poly[m]
            if cross(a, b, c) <= 1e-12:
                continue
            if any(inside(poly[q], a, b, c) for q in idx if q not in (i, j, m)):
                continue
            tris.append((i, j, m))
            idx.pop(k)
            break
        else:
            raise ValueError("polygon is not simple")
    tris.append(tuple(idx))
    return tris


def extrude(poly, width):
    """Closed solid: the (x, z) profile swept along y from 0 to width. Triangles as (3, 3) arrays."""
    n = len(poly)

    def v(i, y):
        return np.array([poly[i % n][0], y, poly[i % n][1]])

    out = []
    for i, j, m in ear_clip(poly):
        out.append([v(i, 0), v(j, 0), v(m, 0)])                    # y = 0 face, normal -y
        out.append([v(i, width), v(m, width), v(j, width)])        # y = width face, normal +y
    for i in range(n):                                             # walls, outward for a CCW profile
        a0, b0, a1, b1 = v(i, 0), v(i + 1, 0), v(i, width), v(i + 1, width)
        out += [[a0, a1, b0], [b0, a1, b1]]
    return np.array(out)


def write_stl(tris, path):
    with open(path, "wb") as f:
        f.write(b"sole plate".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            nrm = np.cross(t[1] - t[0], t[2] - t[0])
            nrm = nrm / (np.linalg.norm(nrm) or 1.0)
            f.write(struct.pack("<12fH", *nrm, *t[0], *t[1], *t[2], 0))


def main():
    tris = extrude(profile(), FOOT_W)
    write_stl(tris, OUT)
    print(f"wrote {OUT} ({len(tris)} triangles): {PLATE_L:.0f} x {FOOT_W:.0f} mm, base {BASE_T:g} mm, "
          f"pocket {FOOT_L + 2 * CLEAR:g} mm between {STOP_H:g} mm end stops")


if __name__ == "__main__":
    main()
