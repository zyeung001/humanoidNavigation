"""Kick-recovery gate for a real-humanoid standing model (the pre-deploy sim gate).

Reproduces the scratchpad gate from the 6/24 + 7/1 deploy sessions as a permanent tool:
settle the policy standing, then inject a root-velocity kick at the TRAINED push magnitude
(lin 0.3 m/s horizontal + yaw +/-1.0 rad/s, matching push_lin_vel/push_ang_vel in
config/real_humanoid_balanced.yaml) and require the policy to recover. On the pre-rebuild
(wrong-mass) MJCF the balanced model passed this; on the corrected 2.046 kg model it failed
3/3 -- the missing recovery basin that shows up on hardware as the seizure. A retrain on the
corrected XML must PASS here before exporting an npz.

Tilt uses the yaw-invariant projected-gravity cosine (R22 of the root), never quat_w.

    python scripts/gate_standing_kick.py --model models/final_real_standing_balanced.zip \
        --vecnorm models/vecnorm_real_standing_balanced.pkl
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "src"))

from src.environments import make_standing_env  # noqa: E402

SETTLE_STEPS = 100
RECOVER_STEPS = 200
TILT_FAIL = 0.6      # R22 below this = tipped (env terminates around here anyway)


def tilt_cos(data, bid):
    w, x, y, z = data.xquat[bid]
    return float(1.0 - 2.0 * (x * x + y * y))   # R22: cos of tilt from vertical, yaw-invariant


def main():
    import mujoco
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/final_real_standing_balanced.zip")
    ap.add_argument("--vecnorm", default="models/vecnorm_real_standing_balanced.pkl")
    ap.add_argument("--config", default="config/real_humanoid_balanced.yaml")
    ap.add_argument("--lin", type=float, default=0.3, help="kick linear velocity (m/s, horizontal)")
    ap.add_argument("--yaw", type=float, default=1.0, help="kick yaw rate (rad/s)")
    ap.add_argument("--kicks", type=int, default=6, help="number of kick directions")
    args = ap.parse_args()

    cfg = (yaml.safe_load(open(PROJ / args.config)).get("standing") or {}).copy()
    # One fixed nominal body, clean obs; actuator_lag stays ON (the real servo has it).
    cfg.update(obs_noise=False, domain_rand=False, actuator_rand=False, push_enabled=False,
               max_episode_steps=SETTLE_STEPS + RECOVER_STEPS + 50)
    env = make_standing_env(config=cfg)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(str(PROJ / args.vecnorm), venv)
    venv.training = False
    venv.norm_reward = False
    # Bounds from the env so the gate exercises what the robot runs (see check_saturation).
    model = PPO.load(str(PROJ / args.model), device="cpu",
                     custom_objects={"action_space": env.action_space})

    m = env.unwrapped.model
    d = env.unwrapped.data
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if bid < 0:
        bid = 1
    print(f"Gate: {args.model} on {cfg.get('xml_file')} (total mass "
          f"{m.body_subtreemass[0]:.3f} kg)")
    print(f"Kick = lin {args.lin} m/s horizontal + yaw +/-{args.yaw} rad/s at step "
          f"{SETTLE_STEPS}, must survive {RECOVER_STEPS} more steps.\n")

    def run_trial(kick_dir_rad=None, yaw_sign=+1.0, label="quiet"):
        obs = venv.reset()
        min_tilt, kicked = 1.0, False
        for step in range(SETTLE_STEPS + RECOVER_STEPS):
            if kick_dir_rad is not None and step == SETTLE_STEPS:
                qvel = d.qvel
                qvel[0] += args.lin * np.cos(kick_dir_rad)
                qvel[1] += args.lin * np.sin(kick_dir_rad)
                qvel[5] += yaw_sign * args.yaw
                mujoco.mj_forward(m, d)
                kicked = True
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(a)
            if kicked:
                min_tilt = min(min_tilt, tilt_cos(d, bid))
            if bool(done[0]):
                print(f"  {label:<14} FELL at step {step} (tilt dip {min_tilt:.3f})")
                return False, min_tilt
        print(f"  {label:<14} STOOD  (tilt dip {min_tilt:.3f})")
        return True, min_tilt

    print("[quiet standing, no kick]")
    quiet_ok, _ = run_trial(None, label="quiet")

    print(f"\n[{args.kicks} kicks at trained magnitude]")
    passes = 0
    for k in range(args.kicks):
        ang = 2.0 * np.pi * k / args.kicks
        ok, _ = run_trial(ang, yaw_sign=(+1.0 if k % 2 == 0 else -1.0),
                          label=f"kick {np.degrees(ang):5.0f}deg")
        passes += int(ok)

    print(f"\nRESULT: quiet={'PASS' if quiet_ok else 'FAIL'}, "
          f"kicks {passes}/{args.kicks} survived")
    if quiet_ok and passes == args.kicks:
        print("GATE PASS -- recovery basin present on this MJCF. OK to export npz + deploy.")
    elif quiet_ok and passes >= args.kicks - 1:
        print("GATE MARGINAL -- one kick direction fails; inspect before deploying.")
    else:
        print("GATE FAIL -- do not deploy; the recovery basin is absent (HW seizure risk).")


if __name__ == "__main__":
    main()
