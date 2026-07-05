"""Find the tilt->joint signs/gains that let BalanceBaseline alone recover a push (no RL).

Raw MuJoCo (position actuators), robot initialized standing straight (joints=0) on the floor,
perturbed with a root angular-velocity kick. For each sign/gain combo, run the baseline ALONE
(ctrl = nominal-straight 0 + baseline.delta) and score by mean upright_cos over the window. The
combo that keeps upright_cos high (vs the no-baseline control that falls) is the correct sign.
"""
import sys
from pathlib import Path
import numpy as np
import mujoco

PROJ = Path(r"C:\Users\zachc\humanoidnavigation")
sys.path.insert(0, str(PROJ))
from src.core.balance_baseline import BalanceBaseline  # noqa: E402

HIP_PITCH = [2, 6]   # R, L hip pitch (MJCF/action idx)
HIP_ROLL = [0, 4]    # R, L hip roll
M = mujoco.MjModel.from_xml_path(str(PROJ / "models/humanoid_real_v2.xml"))
LO, HI = M.actuator_ctrlrange[:, 0], M.actuator_ctrlrange[:, 1]


def proj_grav(d):
    R = d.qpos[3:7]
    Rm = np.zeros(9)
    mujoco.mju_quat2Mat(Rm, R)
    return Rm.reshape(3, 3).T @ np.array([0, 0, -1.0])


def init_straight(d):
    mujoco.mj_resetDataKeyframe(M, d, 0)
    d.qpos[7:] = 0.0                     # straight joints
    d.qpos[3:7] = [1, 0, 0, 0]           # upright
    d.qvel[:] = 0.0
    mujoco.mj_forward(M, d)
    zmin = min(d.geom_xpos[g][2] for g in range(M.ngeom)
              if M.geom_contype[g] and M.geom_bodyid[g] != 0)
    d.qpos[2] -= zmin - 0.002            # drop feet onto floor
    mujoco.mj_forward(M, d)


def run(baseline, tilt_axis, tilt_deg, steps=250):
    d = mujoco.MjData(M)
    init_straight(d)
    # start already leaning hard (past the foot edge) so the control falls and a correct
    # baseline must actively pull it back: tilt about x (roll) or y (pitch).
    a = np.deg2rad(tilt_deg) / 2.0
    q = [np.cos(a), 0, 0, 0]
    q[tilt_axis + 1] = np.sin(a)
    d.qpos[3:7] = q
    mujoco.mj_forward(M, d)
    ups = []
    for _ in range(steps):
        pg = proj_grav(d)
        delta = baseline.delta(pg, d.qvel[3:6], M.nu)
        d.ctrl[:] = np.clip(delta, LO, HI)
        mujoco.mj_step(M, d)
        ups.append(-float(proj_grav(d)[2]))
        if -float(proj_grav(d)[2]) < 0.2:
            break
    ups = np.array(ups)
    return float(ups[-50:].mean()), len(ups)


print("FORWARD lean 18deg (pitch axis) -- find sign_pitch:")
none = BalanceBaseline(HIP_PITCH, HIP_ROLL)  # all gains 0 = no baseline (control)
sc, n = run(none, tilt_axis=1, tilt_deg=18)
print(f"  no-baseline (control): mean_up={sc:.3f} survived={n}/250")
for sp in (+1.0, -1.0):
    for kp in (2.0, 4.0):
        b = BalanceBaseline(HIP_PITCH, HIP_ROLL, kp_pitch=kp, kd_pitch=0.3 * kp, sign_pitch=sp)
        sc, n = run(b, tilt_axis=1, tilt_deg=18)
        print(f"  sign_pitch={sp:+.0f} kp={kp}: mean_up={sc:.3f} survived={n}/250")

print("\nLATERAL lean 14deg (roll axis) -- find sign_roll & roll_mirror:")
sc, n = run(none, tilt_axis=0, tilt_deg=14)
print(f"  no-baseline (control): mean_up={sc:.3f} survived={n}/250")
for sr in (+1.0, -1.0):
    for mir in (+1.0, -1.0):
        b = BalanceBaseline(HIP_PITCH, HIP_ROLL, kp_roll=4.0, kd_roll=1.0,
                            sign_roll=sr, roll_mirror=mir)
        sc, n = run(b, tilt_axis=0, tilt_deg=14)
        print(f"  sign_roll={sr:+.0f} mirror={mir:+.0f}: mean_up={sc:.3f} survived={n}/250")
