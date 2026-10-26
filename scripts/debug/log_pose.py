#!/usr/bin/env python3
"""Turn a hardware frame log into body-pose numbers: where every link was, where the COM was
over the feet, and how much of the pelvis attitude the encoders cannot explain.

WHY. diagnose_run.py says WHEN something happened and pose_viewer.py lets a person WATCH it,
but neither answers the balance question in numbers: how close the centre of mass came to the
edge of the feet, and which way. That needs forward kinematics, which needs MuJoCo, which the
Pi does not have -- so it is done here, after the fact, from the raw columns frame_log.py
records. The output is a CSV Claude can read row by row and a JSON summary it can read whole.

    python scripts/debug/log_pose.py logs/pi/20261025_031319_standing.csv
    python scripts/debug/log_pose.py <log.csv> --model models/humanoid_real_v2_compliant.xml
    -> <stem>.pose.csv + <stem>.pose_summary.json  (next to the log, or --out-dir)

HOW A ROW BECOMES A POSE, and the assumption behind each step:
  1. Joint angles: u_meas through the servo map, via frame_log.pose_from_row() -- the same
     conversion the deploy loop used. A rejected read (NaN) holds the previous frame's value
     and the row is flagged, rather than posing a joint at NaN.
  2. Pelvis attitude: the logged projected gravity pg (already in the base frame, already the
     EMA-filtered value the controller saw), yaw 0. It is applied as the smallest rotation that
     takes the body's measured "up" to world up, so pitch/roll read back from it with the same
     atan2 formulas diagnose_run.py and pose_sweep.py use.
  3. Ground: there are no ankles, so a leg's attitude is the pelvis attitude plus its encoders.
     The STANCE foot is whichever sole centre the IMU-tilted legs put lowest; its sole centre
     sits on the floor (z = 0). The other foot counts as supporting while it is within
     CONTACT_MM of the floor. Each supporting foot's support box is its sole rectangle (from the
     model's foot geom) laid FLAT on the floor at the sole centre, with the foot's heading.
     That is the "flex lumped at the sole" picture the flex plant uses (servo play modelled as
     a passive hinge at each foot): the feet are flat, the body above them is tilted by
     whatever the IMU says. Where the encoders and the IMU disagree, the IMU wins for the body.
  4. COM = mass-weighted body COMs; its floor projection is compared against each support box
     (signed mm to toe / heel / inner / outer edge, + = inside) and against the support
     polygon (convex hull of the supporting boxes).

HIDDEN FLEX (the 10/25 finding: ~75% of pelvis pitch is invisible to the encoders). The
encoders alone imply a pelvis attitude: the one that puts both soles flat (their mean normal
vertical). The IMU measures another. flex = IMU - encoder-implied, in degrees. The encoder side
is passed through the SAME EMA the deploy loop applied to pg (meta projgrav_alpha), so the
filter lag is not reported as flex. A static offset in flex also contains IMU mounting error
and the model's own sole tilt (~0.75 deg of roll at the straight pose); the dynamic part
(deviation from the median) is the cleaner measure, and the summary reports both.

CAVEAT, the accelerometer: pg is the accelerometer's direction (EMA-filtered), i.e. specific
force, not gravity. During a push or a fast swing the body's own acceleration tilts it, so pitch,
flex and the COM margin are inflated exactly when they are most interesting. Each row carries
quiet = 1 when the body rate |av| is under diagnose_run's CALM_RATE; numbers read on quiet rows
are the static truth, the rest are an upper bound on the excursion. On v2 logs acc_norm_g far
from 1 says the same thing directly.

COMMANDED vs MEASURED: where the COM would have been had every servo reached its command (u_cmd
by default -- what the servos were actually told; --cmd q_cmd for the policy's pre-quantisation
action). The feet stay flat and the stance sole stays where it was, so a joint that fell short
moves the body ABOVE the feet: the pelvis attitude becomes the commanded pose's feet-flat
attitude plus the same hidden flex the row measured (IMU o enc_meas^-1 o enc_cmd). Holding the
pelvis attitude fixed instead would get the sign backwards for the knees -- a knee that sagged
with the pelvis held still pushes the feet back, not the COM. com_err = commanded COM - measured
COM, so servo tracking error shows up in mm at the floor, where it matters.

Schema: works on schema-v1 logs and tolerates v2. Unknown columns are ignored and listed in the
summary. acc_x/y/z (raw g, axis_remap applied), when present, add an unfiltered accelerometer
attitude next to the filtered one; nothing here REQUIRES a v2 column.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "deploy"))
sys.path.insert(0, str(ROOT / "scripts" / "debug"))

import frame_log  # noqa: E402
from diagnose_run import CALM_RATE, TIP_COS  # noqa: E402
from sim_real_map import DEFAULT_MAP, SimRealMap  # noqa: E402

# Which MJCF matches the robot on a given day. The 16 cm sole plates went on 10/24 (they were
# designed 10/18; every log before 10/24 is the 12 cm foot). Both models carry the 10/16 phantom-
# collision fix and the compliant waist. Neither has the flex plant's passive foot hinges: those
# hinges are exactly what the encoders cannot see, so posing from encoders would hold them at 0
# anyway, and the plain model keeps the foot mass where the CAD put it.
MODEL_16CM = ROOT / "models" / "humanoid_real_v2_compliant_soleplate.xml"
MODEL_12CM = ROOT / "models" / "humanoid_real_v2_compliant.xml"
SOLE_PLATES_FROM = "2026-10-24"

CONTACT_MM = 10.0       # the other foot counts as supporting within this of the floor
UNITS_MIN, UNITS_MAX = 0, 1023   # SCS15 positions are 10-bit
FILL_WARN = 0.01        # flag in the summary when more than this share of rows had a NaN read

LINKS = ["pelvis", "torso", "thigh_R", "shin_R", "foot_R", "thigh_L", "shin_L", "foot_L"]
FOOT_EDGES = ["toe", "heel", "inner", "outer"]


# ---------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------

def load_log(path):
    """(rows, meta). rows are csv dicts; meta is the sidecar or {} when it is missing."""
    path = Path(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    meta_path = path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            meta = {}
    return rows, meta


def column(rows, name):
    """One column as float, NaN where blank or absent (diagnose_run.col, list-of-dicts form)."""
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        v = r.get(name)
        if v not in ("", None):
            try:
                out[i] = float(v)
            except ValueError:
                pass
    return out


def has_column(rows, name):
    return name in rows[0]


def file_md5(path):
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def choose_model(meta, override=None):
    """(xml path, why). Explicit beats inferred; the reason is written into the summary."""
    if override:
        return Path(override), "given with --model"
    for key in ("model_xml", "xml_file", "mjcf"):
        if meta.get(key):
            p = Path(meta[key])
            p = p if p.is_absolute() else ROOT / p
            if p.exists():
                return p, f"named by the log's meta sidecar ({key})"
    started = str(meta.get("started_iso") or "")[:10]
    if started and started < SOLE_PLATES_FROM:
        return MODEL_12CM, (f"log started {started}, before the 16 cm sole plates went on "
                            f"({SOLE_PLATES_FROM}): 12 cm feet")
    why = (f"log started {started}, on/after {SOLE_PLATES_FROM}: 16 cm sole plates" if started
           else "no start date in the meta sidecar: assuming today's robot (16 cm sole plates)")
    return MODEL_16CM, why


# ---------------------------------------------------------------------------------------
# attitude helpers (same conventions as diagnose_run / sim_cop.pelvis_attitude)
# ---------------------------------------------------------------------------------------

def pitch_roll_deg(pg):
    """pg (..., 3) in the base frame -> (pitch, roll) deg. + pitch = forward, + roll = LEFT."""
    pg = np.asarray(pg, dtype=float)
    return (np.degrees(np.arctan2(pg[..., 0], -pg[..., 2])),
            np.degrees(np.arctan2(pg[..., 1], -pg[..., 2])))


def tilt_rotation(pg):
    """Body->world rotation with zero yaw that makes the body's measured gravity point down.

    The smallest rotation taking body-up (-pg) onto world +z. Reading pitch/roll back out of
    R^T @ (0, 0, -1) returns exactly the pitch/roll of pg, so nothing is lost in the round trip.
    """
    a = -np.asarray(pg, dtype=float)
    n = np.linalg.norm(a)
    if not np.isfinite(n) or n < 1e-9:
        return np.eye(3)
    a = a / n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z)
    s, c = np.linalg.norm(v), float(a @ z)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def ema_unit(vecs, alpha):
    """The deploy loop's pg filter: EMA then renormalise (deploy_standing.read_sensors)."""
    if alpha >= 1.0:
        return vecs.copy()
    out = np.empty_like(vecs)
    g = None
    for i, v in enumerate(vecs):
        if g is None or not np.all(np.isfinite(g)):
            g = v.copy()
        elif np.all(np.isfinite(v)):
            g = (1.0 - alpha) * g + alpha * v
            n = np.linalg.norm(g)
            if n > 1e-6:
                g = g / n
        out[i] = g
    return out


# ---------------------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------------------

class Kinematics:
    """MuJoCo forward kinematics with the base at the origin and identity orientation.

    Everything the pose needs is a rigid transform of these base-frame quantities, so the
    model is evaluated once per joint vector and the attitude/anchoring is applied with
    numpy afterwards.
    """

    def __init__(self, xml, smap):
        import mujoco
        from sim_cop import _foot_geoms

        self.mj = mujoco
        self.xml = Path(xml)
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = {}
        for j in smap.joints:
            k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.mjcf)
            if k < 0:
                raise SystemExit(f"{xml} has no joint '{j.mjcf}' ({j.dof}) -- wrong model for this map")
            jid[j.dof] = k
        # By joint NAME through the servo map, not qpos[7:]: the compliant and flex models
        # insert passive hinges, so actuated qpos addresses are not contiguous.
        self.qadr = np.array([m.jnt_qposadr[jid[j.dof]] for j in smap.joints])
        self.free_adr = int(m.jnt_qposadr[0])
        body = {name: int(m.jnt_bodyid[jid[dof]]) for name, dof in (
            ("torso", "waist_yaw"), ("thigh_R", "R_hip_pitch"), ("shin_R", "R_knee"),
            ("thigh_L", "L_hip_pitch"), ("shin_L", "L_knee"))}
        body["pelvis"] = int(m.jnt_bodyid[0])
        self.body = body
        self.mass = m.body_mass.copy()
        self.total_mass = float(self.mass.sum())

        # Feet: the thin box geoms, assigned to a side by which shin they hang from.
        self.foot = {}
        for g in _foot_geoms(m):
            b = int(m.geom_bodyid[g])
            while b not in (body["shin_R"], body["shin_L"], 0):
                b = int(m.body_parentid[b])
            side = "R" if b == body["shin_R"] else "L" if b == body["shin_L"] else None
            if side:
                self.foot[side] = g
        if set(self.foot) != {"R", "L"}:
            raise SystemExit(f"{xml}: could not find one sole box under each shin")
        # Sole axes, fixed once at the straight pose: which box axis is the normal (thinnest)
        # and the long one, and the signs that make them point DOWN and FORWARD there.
        self._reset(np.zeros(len(self.qadr)))
        self.sole = {}
        for side, g in self.foot.items():
            size = m.geom_size[g]
            order = np.argsort(size)
            ax_n, ax_w, ax_l = int(order[0]), int(order[1]), int(order[2])
            R = self.d.geom_xmat[g].reshape(3, 3)
            self.sole[side] = dict(
                ax_n=ax_n, sn=-np.sign(R[2, ax_n]) or 1.0,
                ax_l=ax_l, sl=np.sign(R[0, ax_l]) or 1.0,
                half_t=float(size[ax_n]), half_len=float(size[ax_l]), half_w=float(size[ax_w]))

    def _reset(self, q):
        d = self.d
        d.qpos[:] = 0.0
        d.qpos[self.free_adr + 3] = 1.0
        d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.m, d)

    def fk(self, q):
        """Base-frame geometry for joint vector q (servo-map order, sim rad)."""
        self._reset(q)
        d = self.d
        out = {"com": (self.mass[:, None] * d.xipos).sum(0) / self.total_mass}
        for name, b in self.body.items():
            out[name] = d.xipos[b].copy()
        up = np.zeros(3)
        for side, g in self.foot.items():
            s = self.sole[side]
            R = d.geom_xmat[g].reshape(3, 3)
            down = s["sn"] * R[:, s["ax_n"]]
            out[f"sole_{side}"] = d.geom_xpos[g] + s["half_t"] * down   # bottom-face centre
            out[f"down_{side}"] = down
            out[f"fwd_{side}"] = s["sl"] * R[:, s["ax_l"]]
            out[f"foot_{side}"] = out[f"sole_{side}"]
            up -= down
        out["sole_up"] = up / np.linalg.norm(up)    # mean sole normal: "both feet flat" = this vertical
        return out


# ---------------------------------------------------------------------------------------
# support geometry
# ---------------------------------------------------------------------------------------

def foot_box(center_xy, fwd_xyz, half_len, half_w):
    """Sole rectangle flat on the floor: centre, unit forward, unit left, corners (4, 2)."""
    f = np.asarray(fwd_xyz[:2], dtype=float)
    n = np.linalg.norm(f)
    f = f / n if n > 1e-9 else np.array([1.0, 0.0])
    left = np.array([-f[1], f[0]])
    c = np.asarray(center_xy, dtype=float)
    corners = np.array([c + sx * half_len * f + sy * half_w * left
                        for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    return c, f, left, corners


def foot_margins(p_xy, box, half_len, half_w, side):
    """Signed mm from the COM projection to each sole edge, + = inside that edge."""
    c, f, left, _ = box
    x = float((p_xy - c) @ f)
    y = float((p_xy - c) @ left)
    # inner = toward the other foot: +left for the right foot, -left for the left foot
    inner = half_w - y if side == "R" else half_w + y
    outer = half_w + y if side == "R" else half_w - y
    return {"toe": 1000 * (half_len - x), "heel": 1000 * (half_len + x),
            "inner": 1000 * inner, "outer": 1000 * outer}


def convex_hull(points):
    """Monotone chain; counter-clockwise, no repeated end point."""
    pts = sorted(map(tuple, np.asarray(points, dtype=float)))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_margin(p, hull, fwd=np.array([1.0, 0.0])):
    """(signed distance mm, direction of the nearest edge). + = COM inside the polygon.

    Direction is the nearest edge's outward normal named in the robot's frame (forward/back/
    left/right), so "min margin 3 mm, forward" reads as "nearly over the toes".
    """
    best, best_n = np.inf, None
    inside = True
    k = len(hull)
    for i in range(k):
        a, b = hull[i], hull[(i + 1) % k]
        e = b - a
        L = np.linalg.norm(e)
        if L < 1e-12:
            continue
        out_n = np.array([e[1], -e[0]]) / L         # CCW polygon: outward normal is to the right
        if (p - a) @ out_n > 0:
            inside = False
        t = np.clip((p - a) @ e / (L * L), 0.0, 1.0)
        dist = np.linalg.norm(p - (a + t * e))
        if dist < best:
            best, best_n = dist, out_n
    if best_n is None:
        return float("nan"), ""
    left = np.array([-fwd[1], fwd[0]])
    fx, fy = float(best_n @ fwd), float(best_n @ left)
    way = (("forward" if fx > 0 else "back") if abs(fx) >= abs(fy)
           else ("left" if fy > 0 else "right"))
    return 1000 * best * (1.0 if inside else -1.0), way


def place(fk, R, stance=None, anchor=None):
    """Put fk (base frame) in the world with body rotation R.

    The stance foot (lowest sole unless given) gets its sole centre on the floor; `anchor`
    (world xyz) fixes that sole centre somewhere specific, else the origin is the midpoint
    between the two soles on the floor. Returns world positions plus the stance choice.
    """
    soles = {s: R @ fk[f"sole_{s}"] for s in ("R", "L")}
    if stance is None:
        stance = "R" if soles["R"][2] <= soles["L"][2] else "L"
    if anchor is None:
        mid = 0.5 * (soles["R"] + soles["L"])
        t = np.array([-mid[0], -mid[1], -soles[stance][2]])
    else:
        t = np.asarray(anchor, dtype=float) - soles[stance]
    w = {k: R @ v + t for k, v in fk.items() if not k.startswith(("down_", "fwd_", "sole_up"))}
    for s in ("R", "L"):
        w[f"fwd_{s}"] = R @ fk[f"fwd_{s}"]
    other = "L" if stance == "R" else "R"
    w["stance"] = stance
    w["clear_mm"] = 1000 * float(w[f"sole_{other}"][2])
    return w


def support(w, kin):
    """Per-foot margins and the support-polygon margin for one placed pose."""
    supporting = [w["stance"]]
    other = "L" if w["stance"] == "R" else "R"
    if w["clear_mm"] <= CONTACT_MM:
        supporting.append(other)
    p = w["com"][:2]
    out, corners = {}, []
    for s in ("R", "L"):
        sp = kin.sole[s]
        box = foot_box(w[f"sole_{s}"][:2], w[f"fwd_{s}"], sp["half_len"], sp["half_w"])
        for edge, v in foot_margins(p, box, sp["half_len"], sp["half_w"], s).items():
            out[f"{s}_{edge}_mm"] = v
        if s in supporting:
            corners.append(box[3])
    hull = convex_hull(np.vstack(corners))
    # robot heading = mean of the two foot headings
    h = w["fwd_R"][:2] + w["fwd_L"][:2]
    h = h / np.linalg.norm(h) if np.linalg.norm(h) > 1e-9 else np.array([1.0, 0.0])
    out["support_margin_mm"], out["support_nearest"] = polygon_margin(p, hull, h)
    out["support"] = "both" if len(supporting) == 2 else supporting[0]
    return out


# ---------------------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------------------

def units_matrix(rows, smap, block):
    """(n_rows, 17) raw servo units for u_meas/u_cmd, NaN where missing OR impossible.

    A read can pass the bus checksum and still be garbage: on 10/25 01:28 (t=42.3 s) thirteen
    joints read 14338 in two frames, during a near-fall. SCS15 positions are 10-bit, so anything
    outside 0..1023 is treated like a rejected read instead of posing a leg 4000 deg away.
    """
    dofs = [j.dof for j in smap.joints]
    raw = np.column_stack([column(rows, f"{block}.{d}") for d in dofs])
    raw[(raw < UNITS_MIN) | (raw > UNITS_MAX)] = np.nan
    return raw


def joint_matrix(rows, smap, block):
    """(n_rows, 17) sim rad for a block, NaN where missing. u_* via the servo map."""
    if block.startswith("u_"):
        return np.asarray(smap.units_to_rad(units_matrix(rows, smap, block)), dtype=float)
    dofs = [j.dof for j in smap.joints]
    return np.column_stack([column(rows, f"{block}.{d}") for d in dofs])


def forward_fill(q):
    """Hold the last good value per joint through NaN reads; returns (q, filled_row_mask)."""
    q = q.copy()
    bad = ~np.isfinite(q)
    filled = bad.any(axis=1)
    for j in range(q.shape[1]):
        last = 0.0
        good = np.flatnonzero(~bad[:, j])
        if good.size:
            last = q[good[0], j]          # leading NaNs take the first good value
        for i in range(q.shape[0]):
            if bad[i, j]:
                q[i, j] = last
            else:
                last = q[i, j]
    return q, filled


def known_columns(meta, rows, smap):
    """Columns this tool understands; anything else in the log is listed as ignored."""
    dofs = [j.dof for j in smap.joints]
    known = set(frame_log.build_columns(dofs))
    known |= {"acc_x", "acc_y", "acc_z", "posture_lean", "posture_zero", "posture_corr"}
    for block in ("spd_raw", "load_raw", "volt_raw", "temp_raw", "q_prior"):
        known |= {f"{block}.{d}" for d in dofs}
    known |= {c for c in rows[0] if c.startswith("fsr_v")}
    return known


def analyze(log_path, model=None, cmd="u_cmd", map_path=DEFAULT_MAP):
    """Pose every row. Returns (columns, table rows as dicts, summary dict, arrays dict)."""
    rows, meta = load_log(log_path)
    smap = SimRealMap(map_path)
    log_dofs = meta.get("dofs")
    map_dofs = [j.dof for j in smap.joints]
    if log_dofs and list(log_dofs) != map_dofs:
        raise SystemExit(f"{log_path}: joint order {log_dofs} differs from the servo map {map_dofs}")
    if not has_column(rows, f"u_meas.{map_dofs[0]}"):
        raise SystemExit(f"{log_path}: no u_meas columns -- is this a deploy frame log?")

    xml, why = choose_model(meta, model)
    kin = Kinematics(xml, smap)

    # 1. joints: measured through pose_from_row (the same map the deploy loop used)
    q_meas = np.array([frame_log.pose_from_row(r, smap)["q_meas"] for r in rows], dtype=float)
    u_ok = np.isfinite(units_matrix(rows, smap, "u_meas"))
    bad_reads = int((~u_ok).sum())
    q_meas[~u_ok] = np.nan
    q_meas, filled = forward_fill(q_meas)
    cmd_used = cmd
    q_cmd = joint_matrix(rows, smap, cmd) if has_column(rows, f"{cmd}.{map_dofs[0]}") else None
    if q_cmd is None or not np.isfinite(q_cmd).any():
        alt = "q_cmd" if cmd == "u_cmd" else "u_cmd"
        q_cmd = joint_matrix(rows, smap, alt) if has_column(rows, f"{alt}.{map_dofs[0]}") else None
        cmd_used = alt if q_cmd is not None else None
    if q_cmd is not None:
        q_cmd, _ = forward_fill(q_cmd)

    t = column(rows, "t_rel")
    step = column(rows, "step")
    pg = np.column_stack([column(rows, f"pg_{a}") for a in "xyz"])
    pg_ok = np.all(np.isfinite(pg), axis=1)
    pg[~pg_ok] = (0.0, 0.0, -1.0)
    imu_pitch, imu_roll = pitch_roll_deg(pg)
    imu_pitch[~pg_ok] = np.nan
    imu_roll[~pg_ok] = np.nan
    up = column(rows, "upright_cos")
    av = np.column_stack([column(rows, f"av_{a}") for a in "xyz"])
    rate = np.linalg.norm(av, axis=1)

    has_acc = all(has_column(rows, f"acc_{a}") for a in "xyz")
    if has_acc:
        acc = np.column_stack([column(rows, f"acc_{a}") for a in "xyz"])
        acc_norm = np.linalg.norm(acc, axis=1)
        acc_pitch, acc_roll = pitch_roll_deg(-acc / np.maximum(acc_norm, 1e-9)[:, None])

    # 2. encoder-implied attitude: the one with the mean sole normal vertical
    fks = [kin.fk(q) for q in q_meas]
    pg_enc_raw = np.array([-f["sole_up"] for f in fks])
    alpha = float(meta.get("projgrav_alpha", 1.0) or 1.0)
    pg_enc = ema_unit(pg_enc_raw, alpha)
    if q_cmd is not None:
        fks_cmd = [kin.fk(q) for q in q_cmd]
        pg_cmd = ema_unit(np.array([-f["sole_up"] for f in fks_cmd]), alpha)
    enc_pitch, enc_roll = pitch_roll_deg(pg_enc)
    flex_pitch, flex_roll = imu_pitch - enc_pitch, imu_roll - enc_roll

    # tip: everything after it is not a standing measurement (diagnose_run's rule)
    tipped = np.isfinite(up) & (up < TIP_COS)
    end = int(np.argmax(tipped)) if tipped.any() else len(rows)
    fallen = np.arange(len(rows)) >= end

    # 3/4. place, measure support
    cols = (["t_rel", "step", "fallen", "quiet", "u_meas_filled", "stance", "support", "swing_clear_mm",
             "imu_pitch_deg", "imu_roll_deg", "enc_pitch_deg", "enc_roll_deg",
             "flex_pitch_deg", "flex_roll_deg"]
            + (["acc_pitch_deg", "acc_roll_deg", "acc_norm_g"] if has_acc else [])
            + [f"{lk}_{a}_mm" for lk in LINKS for a in "xyz"]
            + ["com_x_mm", "com_y_mm", "com_z_mm"]
            + (["com_cmd_x_mm", "com_cmd_y_mm", "com_err_x_mm", "com_err_y_mm"] if q_cmd is not None else [])
            + ["com_flat_x_mm", "com_flat_y_mm", "support_margin_flat_mm"]
            + [f"{s}_{e}_mm" for s in ("R", "L") for e in FOOT_EDGES]
            + ["support_margin_mm", "support_nearest"])
    table = []
    for i in range(len(rows)):
        R = tilt_rotation(pg[i])
        w = place(fks[i], R)
        sup = support(w, kin)
        rec = {"t_rel": t[i], "step": step[i], "fallen": int(fallen[i]),
               "quiet": int(np.isfinite(rate[i]) and rate[i] < CALM_RATE), "u_meas_filled": int(filled[i]),
               "stance": w["stance"], "support": sup["support"], "swing_clear_mm": w["clear_mm"],
               "imu_pitch_deg": imu_pitch[i], "imu_roll_deg": imu_roll[i],
               "enc_pitch_deg": enc_pitch[i], "enc_roll_deg": enc_roll[i],
               "flex_pitch_deg": flex_pitch[i], "flex_roll_deg": flex_roll[i]}
        if has_acc:
            rec.update(acc_pitch_deg=acc_pitch[i], acc_roll_deg=acc_roll[i], acc_norm_g=acc_norm[i])
        for lk in LINKS:
            for k, a in enumerate("xyz"):
                rec[f"{lk}_{a}_mm"] = 1000 * w[lk][k]
        for k, a in enumerate("xyz"):
            rec[f"com_{a}_mm"] = 1000 * w["com"][k]
        if q_cmd is not None:
            # flex held, joints at their commands: R_imu R_enc_meas^T R_enc_cmd
            R_c = R @ tilt_rotation(pg_enc[i]).T @ tilt_rotation(pg_cmd[i])
            wc = place(fks_cmd[i], R_c, stance=w["stance"], anchor=w[f"sole_{w['stance']}"])
            rec.update(com_cmd_x_mm=1000 * wc["com"][0], com_cmd_y_mm=1000 * wc["com"][1],
                       com_err_x_mm=1000 * (wc["com"][0] - w["com"][0]),
                       com_err_y_mm=1000 * (wc["com"][1] - w["com"][1]))
        wf = place(fks[i], tilt_rotation(pg_enc_raw[i]), stance=w["stance"], anchor=w[f"sole_{w['stance']}"])
        rec.update(com_flat_x_mm=1000 * wf["com"][0], com_flat_y_mm=1000 * wf["com"][1],
                   support_margin_flat_mm=support(wf, kin)["support_margin_mm"])
        rec.update({k: v for k, v in sup.items() if k != "support"})
        table.append(rec)

    arrays = {c: np.array([r[c] for r in table]) for c in cols}
    summary = summarize(log_path, rows, meta, arrays, kin, xml, why, cmd_used, alpha, end, t, up,
                        pg, rate, filled, has_acc, known_columns(meta, rows, smap))
    summary["u_meas_bad_values"] = bad_reads
    return cols, table, summary, arrays


# ---------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------

def _stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    return {"median": round(float(np.median(x)), 2), "min": round(float(x.min()), 2),
            "max": round(float(x.max()), 2), "std": round(float(x.std()), 2),
            "p05": round(float(np.percentile(x, 5)), 2), "p95": round(float(np.percentile(x, 95)), 2)}


def missed_share(imu, enc):
    """Share of the IMU's attitude excursion the encoders do not explain.

    1 - (projection of the encoder excursion onto the IMU excursion) / |IMU excursion|^2, both
    measured from their medians. 0 = encoders explain all of it; 1 = none of it; > 1 = the
    encoders moved the opposite way.
    """
    ok = np.isfinite(imu) & np.isfinite(enc)
    if ok.sum() < 10:
        return None
    a = imu[ok] - np.median(imu[ok])
    b = enc[ok] - np.median(enc[ok])
    den = float(a @ a)
    return None if den < 1e-12 else round(1.0 - float(a @ b) / den, 3)


def summarize(log_path, rows, meta, A, kin, xml, why, cmd_used, alpha, end, t, up, pg, rate,
              filled, has_acc, known):
    n = len(rows)
    win = slice(0, end)
    tt = t[win]
    dur = float(np.nanmax(t) - np.nanmin(t)) if np.isfinite(t).any() else None
    fell = end < n
    fall = None
    if fell:
        p, r = A["imu_pitch_deg"][end], A["imu_roll_deg"][end]
        fall = {"t_rel": round(float(t[end]), 2), "upright_cos": round(float(up[end]), 3),
                "direction": ("forward" if p > abs(r) else "backward" if -p > abs(r)
                              else "left" if r > 0 else "right")}
    quiet = np.zeros(n, dtype=bool)
    quiet[:end] = np.isfinite(rate[:end]) & (rate[:end] < CALM_RATE)

    sm = A["support_margin_mm"][win]
    k_min = int(np.nanargmin(sm)) if np.isfinite(sm).any() else None
    edge_cols = [f"{s}_{e}_mm" for s in ("R", "L") for e in FOOT_EDGES]
    st = A["stance"][win]
    foot_min = {}
    for s in ("R", "L"):
        on = st == s
        if on.any():
            vals = {e: float(np.nanmin(A[f"{s}_{e}_mm"][win][on])) for e in FOOT_EDGES}
            e = min(vals, key=vals.get)
            foot_min[s] = {"edge": e, "mm": round(vals[e], 1)}

    sw = A["support"][win]
    sumry = {
        "log": str(log_path),
        "schema_version": meta.get("schema_version"),
        "rows": n,
        "duration_s": None if dur is None else round(dur, 2),
        "model": {"xml": str(Path(xml).resolve().relative_to(ROOT)) if Path(xml).resolve().is_relative_to(ROOT)
                  else str(xml),
                  "md5": file_md5(xml), "why": why,
                  "foot_len_mm": round(2000 * kin.sole["R"]["half_len"], 1),
                  "foot_width_mm": round(2000 * kin.sole["R"]["half_w"], 1),
                  "mass_kg": round(kin.total_mass, 4)},
        "conventions": {
            "frame": "x forward (pelvis heading, yaw 0), y = robot's left, z up; origin = floor point midway "
                     "between the sole centres; all positions mm",
            "attitude": "+pitch = forward, +roll = robot's LEFT (atan2 of pg, as diagnose_run.py)",
            "pelvis_attitude_from": "logged pg (EMA-filtered, what the controller saw), yaw 0",
            "stance_rule": "stance = lower sole centre with the IMU-tilted legs; its sole centre on the floor; "
                           f"the other foot also supports while within {CONTACT_MM:g} mm of the floor",
            "support_box": "each sole box laid flat on the floor at its sole centre (flex lumped at the sole)",
            "margins": "signed mm, + = COM floor projection inside that edge / polygon",
            "quiet": f"1 when |av| < {CALM_RATE} rad/s; pg is accelerometer-based, so attitude/flex/margins on "
                     "non-quiet rows include the body's own acceleration (an upper bound on the excursion)",
            "flex": "IMU attitude - encoder-implied attitude (both soles flat), encoder side EMA-filtered "
                    f"with alpha {alpha:g} to match the deploy pg filter",
            "com_err": f"COM had the joints reached their commands ({cmd_used}) - COM measured; feet flat, same "
                       "stance sole, same hidden flex (pelvis = commanded feet-flat attitude + measured flex)",
            "com_flat": "COM placed by the encoders alone (pelvis at the encoder-implied attitude)",
        },
        "pg_ema_alpha": alpha,
        "fell": fell,
        "fall": fall,
        "standing_window_s": round(float(tt[-1] - tt[0]), 2) if len(tt) > 1 else 0.0,
        "quiet_frames": int(quiet.sum()),
        "imu_pitch_deg": _stats(A["imu_pitch_deg"][win]),
        "imu_roll_deg": _stats(A["imu_roll_deg"][win]),
        "enc_pitch_deg": _stats(A["enc_pitch_deg"][win]),
        "enc_roll_deg": _stats(A["enc_roll_deg"][win]),
        "flex_pitch_deg": _stats(A["flex_pitch_deg"][win]),
        "flex_roll_deg": _stats(A["flex_roll_deg"][win]),
        "pitch_missed_by_encoders": missed_share(A["imu_pitch_deg"][win], A["enc_pitch_deg"][win]),
        "roll_missed_by_encoders": missed_share(A["imu_roll_deg"][win], A["enc_roll_deg"][win]),
        "pitch_excursion_ratio_imu_over_enc": (
            round(float(np.nanstd(A["imu_pitch_deg"][win]) / np.nanstd(A["enc_pitch_deg"][win])), 2)
            if np.nanstd(A["enc_pitch_deg"][win]) > 1e-9 else None),
        "com_x_mm": _stats(A["com_x_mm"][win]),
        "com_y_mm": _stats(A["com_y_mm"][win]),
        "com_flat_x_mm": _stats(A["com_flat_x_mm"][win]),
        "support_margin_mm": _stats(sm),
        "support_margin_quiet_median_mm": (round(float(np.nanmedian(A["support_margin_mm"][quiet])), 1)
                                           if quiet.any() else None),
        "support_margin_min": None if k_min is None else {
            "mm": round(float(sm[k_min]), 1), "t_rel": round(float(tt[k_min]), 2),
            "nearest": str(A["support_nearest"][win][k_min]), "stance": str(st[k_min])},
        "share_outside_support": round(float(np.mean(sm < 0)), 3) if len(sm) else None,
        "support_margin_flat_mm": _stats(A["support_margin_flat_mm"][win]),
        "foot_edge_min_while_stance": foot_min,
        "stance_frames": {s: int((st == s).sum()) for s in ("R", "L")},
        "support_frames": {s: int((sw == s).sum()) for s in ("both", "R", "L")},
        "u_meas_filled_rows": int(filled.sum()),
        "u_meas_filled_at_t": [round(float(x), 2) for x in t[filled][:20]],
        "optional_columns_used": (["acc_x", "acc_y", "acc_z"] if has_acc else []),
        "ignored_columns": sorted(c for c in rows[0] if c not in known),
        "edge_columns": edge_cols,
    }
    if "com_err_x_mm" in A:
        ex, ey = A["com_err_x_mm"][win], A["com_err_y_mm"][win]
        sumry["com_err_x_mm"] = _stats(ex)
        sumry["com_err_y_mm"] = _stats(ey)
        sumry["com_err_abs_p95_mm"] = round(float(np.nanpercentile(np.hypot(ex, ey), 95)), 1)
    if filled.mean() > FILL_WARN:
        sumry["warning"] = (f"{100 * filled.mean():.1f}% of rows had a NaN encoder read; "
                            "those joints were held at their previous value")
    return sumry


# ---------------------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------------------

def _fmt(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"


def write_outputs(log_path, cols, table, summary, out_dir=None):
    log_path = Path(log_path)
    d = Path(out_dir) if out_dir else log_path.parent
    d.mkdir(parents=True, exist_ok=True)
    pose_csv = d / f"{log_path.stem}.pose.csv"
    with open(pose_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for rec in table:
            w.writerow([_fmt(rec[c]) for c in cols])
    js = d / f"{log_path.stem}.pose_summary.json"
    js.write_text(json.dumps(summary, indent=2, default=str))
    return pose_csv, js


def main():
    p = argparse.ArgumentParser(description="Frame log -> per-row body pose, COM and support margins")
    p.add_argument("log")
    p.add_argument("--model", default=None,
                   help="MJCF to pose. Default: chosen from the log's date (16 cm sole plates from "
                        f"{SOLE_PLATES_FROM}); the choice and why are written to the summary")
    p.add_argument("--cmd", choices=["u_cmd", "q_cmd"], default="u_cmd",
                   help="commanded angles for com_err: u_cmd = what the servos were told (default), "
                        "q_cmd = the policy's action before quantisation and rate limit")
    p.add_argument("--map", default=str(DEFAULT_MAP))
    p.add_argument("--out-dir", default=None, help="default: next to the log")
    args = p.parse_args()

    cols, table, summary, _ = analyze(args.log, args.model, args.cmd, args.map)
    pose_csv, js = write_outputs(args.log, cols, table, summary, args.out_dir)
    s = summary
    print(f"{args.log}: {s['rows']} rows, {s['duration_s']} s, model {s['model']['xml']} ({s['model']['why']})")
    if s["fell"]:
        print(f"  FELL {s['fall']['direction']} at t={s['fall']['t_rel']} s")
    fp, ep, xp = s["flex_pitch_deg"], s["enc_pitch_deg"], s["imu_pitch_deg"]
    if fp and ep and xp:
        print(f"  pelvis pitch: IMU median {xp['median']:+.2f} (std {xp['std']:.2f}) | encoders "
              f"{ep['median']:+.2f} (std {ep['std']:.2f}) | flex median {fp['median']:+.2f}; "
              f"{s['pitch_missed_by_encoders']} of the IMU excursion missed by the encoders")
    sm = s["support_margin_mm"]
    if sm:
        mn = s["support_margin_min"]
        print(f"  COM support margin: median {sm['median']:.1f} mm (quiet {s['support_margin_quiet_median_mm']}), "
              f"min {mn['mm']:.1f} mm at t={mn['t_rel']} s toward {mn['nearest']}")
    print(f"  -> {pose_csv}\n  -> {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
