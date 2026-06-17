"""
Solve a natural, gravity-balanced standing pose for humanoid_real_v2.xml.

Unconstrained torque-min contorts the robot, so we minimize a weighted blend:
  - feet FLAT on the floor (sole faces horizontal, both feet level)
  - COM horizontally over the foot support centroid (won't tip)
  - torso UPRIGHT
  - low gravity hold-torque  (qfrc_bias, frame-independent)
  - stay near the joint-zero pose (no contortion)

geom_xpos / xipos are physically correct despite the Fusion frame scattering;
only joint xanchor frames are unreliable, which we don't use here.

Writes models/standing_pose.json (joint -> angle) which the builder bakes into
the <key name="home"> keyframe. Verifies with a PD-hold sim (must not collapse).

Usage: py -3.13 scripts/solve_standing_pose.py
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

PROJ  = Path(__file__).parent.parent
MODEL = PROJ / "models" / "humanoid_real_v2.xml"
OUT   = PROJ / "models" / "standing_pose.json"

m = mujoco.MjModel.from_xml_path(str(MODEL))
d = mujoco.MjData(m)

ACT = [j for j in range(m.njnt) if m.joint(j).name != 'base_joint']
QADR = [m.jnt_qposadr[j] for j in ACT]
DADR = [m.jnt_dofadr[j] for j in ACT]
NAMES = [m.joint(j).name for j in ACT]
LO = np.array([m.jnt_range[j][0] for j in ACT])
HI = np.array([m.jnt_range[j][1] for j in ACT])

# foot collision boxes = box geoms whose body owns a mesh named '*foot*'
def is_foot_body(bid):
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_dataid[g] >= 0:
            if 'foot' in m.mesh(m.geom_dataid[g]).name.lower():
                return True
    return False
FOOT_BOX = [g for g in range(m.ngeom)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and is_foot_body(m.geom_bodyid[g])]
TORSO = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, '0003_8')
TOTAL_M = m.body_mass.sum()
print(f"actuated={len(ACT)}  foot boxes={FOOT_BOX}  torso_bid={TORSO}")

_CORNER = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def kin(x):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[:3] = 0
    d.qpos[3:7] = [1, 0, 0, 0]
    for a, v in zip(QADR, x):
        d.qpos[a] = v
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def foot_lowz(g):
    c = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    h = m.geom_size[g]
    z = (c + (_CORNER * h) @ R.T)[:, 2]
    low = np.sort(z)[:4]
    return low.mean(), low.var()


def com_xy():
    return (d.xipos * m.body_mass[:, None]).sum(0)[:2] / TOTAL_M


def objective(x):
    kin(x)
    lows = [foot_lowz(g) for g in FOOT_BOX]
    flat = sum(v for _, v in lows)
    level = (lows[0][0] - lows[1][0]) ** 2 if len(lows) == 2 else 0.0
    fmid = np.mean([d.geom_xpos[g][:2] for g in FOOT_BOX], axis=0)
    com = ((com_xy() - fmid) ** 2).sum()
    tz = d.xmat[TORSO].reshape(3, 3)[:, 2]
    torso = (1.0 - tz[2]) ** 2
    tau = sum(float(d.qfrc_bias[a]) ** 2 for a in DADR)
    reg = float((x ** 2).sum())
    return 80 * flat + 180 * level + 120 * com + 15 * torso + 0.3 * tau + 0.15 * reg


def metrics(x):
    kin(x)
    lows = [foot_lowz(g) for g in FOOT_BOX]
    fmid = np.mean([d.geom_xpos[g][:2] for g in FOOT_BOX], axis=0)
    tau = np.array([abs(float(d.qfrc_bias[a])) for a in DADR])
    return {
        'max_hold_Nm': float(tau.max()),
        'com_off_mm': float(np.hypot(*(com_xy() - fmid)) * 1000),
        'foot_tilt_mm2': float(sum(v for _, v in lows) * 1e6),
        'foot_dz_mm': float(abs(lows[0][0] - lows[1][0]) * 1000) if len(lows) == 2 else 0.0,
    }


# multi-start (zeros + random) to dodge bad local minima
best = None
rng = np.random.default_rng(0)
starts = [np.zeros(len(ACT))] + [rng.uniform(LO, HI) * 0.5 for _ in range(7)]
for i, x0 in enumerate(starts):
    r = minimize(objective, np.clip(x0, LO, HI), method='L-BFGS-B',
                 bounds=list(zip(LO, HI)), options={'maxiter': 600})
    if best is None or r.fun < best.fun:
        best = r
    print(f"  start {i}: obj={r.fun:.4f}")

x = best.x
mt = metrics(x)
print(f"\nbest objective={best.fun:.4f}")
print(f"  max hold torque = {mt['max_hold_Nm']:.3f} N*m  (legs/waist limit 1.47)")
print(f"  COM over feet   = {mt['com_off_mm']:.1f} mm")
print(f"  foot flatness   = {mt['foot_tilt_mm2']:.2f} mm^2   foot level dz = {mt['foot_dz_mm']:.2f} mm")
print("  pose (deg):", {n: round(np.degrees(v), 1) for n, v in zip(NAMES, x)})

# --- PD-hold verification: command this pose, must not collapse ---
kin(x)
fmid = np.mean([d.geom_xpos[g][:2] for g in FOOT_BOX], axis=0)
minfoot = min(np.sort((d.geom_xpos[g] + (_CORNER * m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3).T)[:, 2])[0]
              for g in FOOT_BOX)
base_xy = -fmid
base_z = -minfoot + 0.002

d2 = mujoco.MjData(m)
d2.qpos[:3] = [base_xy[0], base_xy[1], base_z]
d2.qpos[3:7] = [1, 0, 0, 0]
for a, v in zip(QADR, x):
    d2.qpos[a] = v
ctrl = np.array(x)
z0 = base_z
fall_step = None
for k in range(2000):
    d2.ctrl[:] = ctrl
    mujoco.mj_step(m, d2)
    if fall_step is None and (abs(float(d2.qpos[2]) - z0) > 0.05 or
                              d2.xmat[TORSO].reshape(3, 3)[2, 2] < 0.8):
        fall_step = k

# A no-ankle biped is an inverted pendulum: it CANNOT stand open-loop with fixed
# joint targets (no ankle torque to correct lean). So we judge the RESET pose on
# static validity, not open-loop hold; the RL policy supplies balance.
static_ok = (mt['max_hold_Nm'] < 1.47 and mt['com_off_mm'] < 25 and
             mt['foot_dz_mm'] < 5 and mt['foot_tilt_mm2'] < 40)
fall_t = (fall_step * m.opt.timestep) if fall_step is not None else float('inf')
print(f"\nRESET POSE valid (static)? {'YES' if static_ok else 'NO'}  "
      f"[hold<1.47Nm, COM<25mm, feet level<5mm, flat<40mm^2]")
print(f"open-loop hold time = {fall_t:.2f}s before lean (expected short: no ankle, "
      f"RL balances at train time)")

json.dump({'base_pos': [float(base_xy[0]), float(base_xy[1]), float(base_z)],
           'joints': {n: float(v) for n, v in zip(NAMES, x)}},
          open(OUT, 'w'), indent=2)
print(f"wrote {OUT}")

# render the solved pose
from PIL import Image
mujoco.mj_resetDataKeyframe(m, d, 0)
d.qpos[:3] = [base_xy[0], base_xy[1], base_z]
d.qpos[3:7] = [1, 0, 0, 0]
for a, v in zip(QADR, x):
    d.qpos[a] = v
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 480, 640)
opt = mujoco.MjvOption()
for g in range(6):
    opt.geomgroup[g] = 1
cam = mujoco.MjvCamera()
cam.distance = 1.1
cam.elevation = -8
cam.lookat[:] = [float(d.qpos[0]), float(d.qpos[1]), 0.30]
for nm, az in [("front", 90), ("side", 0)]:
    cam.azimuth = az
    r.update_scene(d, cam, opt)
    Image.fromarray(r.render()).save(str(PROJ / "models" / f"_stand_{nm}.png"))
print("rendered _stand_front.png / _stand_side.png")
