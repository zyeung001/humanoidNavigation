"""Measure waist-twist ratio (WTR) on the REAL-BODY turning policy (humanoid_real_v2).

Mirror of scripts/measure_waist_twist.py for the custom robot. WTR = std(torso_yaw - pelvis_yaw)
over the turning phases of a standardized turn-in-place schedule. Body names differ from
Humanoid-v5: torso = chest body '0003_8' (above the 3-joint waist), pelvis = freejoint root
'base_link'. The waist-twist DOF is Revolute 19 (waist_yaw, servo 1). Feet = '0003_2' (R) /
'0003_5' (L). WTR is read from body world quaternions (xquat), independent of the heading_source
code path the policy was rewarded on.

Usage:
    # pipeline check with a random policy (no trained model needed):
    python scripts/measure_waist_twist_real.py --random-policy --config config/turning/heading_torso.yaml
    # real measurement on a trained arm:
    python scripts/measure_waist_twist_real.py --model models/final_real_turn_hack.zip \
        --vecnorm models/vecnorm_real_turn_hack.pkl --config config/turning/heading_torso.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "src"))

from src.environments import make_real_turning_env  # noqa: E402

# turn-in-place schedule: (start, end, vx, vy, yaw_rate, phase)
SCHEDULE = [
    (0, 100, 0.0, 0.0, 0.0, "A"),
    (100, 300, 0.0, 0.0, +0.5, "B"),
    (300, 500, 0.0, 0.0, -0.5, "C"),
]
TURN_START = 100


def phase_for(step):
    for s, e, vx, vy, yr, lab in SCHEDULE:
        if s <= step < e:
            return vx, vy, yr, lab
    return SCHEDULE[-1][2], SCHEDULE[-1][3], SCHEDULE[-1][4], SCHEDULE[-1][5]


def yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def main():
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/turning/heading_torso.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--random-policy", action="store_true")
    args = ap.parse_args()

    cfg = (yaml.safe_load(open(PROJ / args.config)).get("turning") or {}).copy()
    cfg.update(obs_noise=False, domain_rand=False, actuator_rand=False, max_episode_steps=520)
    env = make_real_turning_env(config=cfg)

    m = env.unwrapped.model
    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
           for n in ("0003_8", "base_link", "0003_2", "0003_5")}
    missing = [n for n, i in bid.items() if i < 0]
    if missing:
        raise RuntimeError(f"bodies not found: {missing}")
    print(f"torso=0003_8(id{bid['0003_8']}) pelvis=base_link(id{bid['base_link']}) "
          f"feet=0003_2/0003_5 | heading_source={cfg.get('heading_source')}")

    if args.random_policy or args.model is None:
        predict = lambda o: env.action_space.sample()  # noqa: E731
        print("Policy: random (pipeline check)")
    else:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        venv = DummyVecEnv([lambda: env])
        venv = VecNormalize.load(str(PROJ / args.vecnorm), venv)
        venv.training = False
        venv.norm_reward = False
        model = PPO.load(str(PROJ / args.model), device="cpu")
        predict = None  # use venv path below
        print(f"Policy: {args.model}")

    wtrs = []
    for ep in range(args.episodes):
        if args.model and not args.random_policy:
            obs = venv.reset()
        else:
            obs, _ = env.reset(seed=ep)
        twist, pelvis_y, torso_y = [], [], []
        for step in range(520):
            vx, vy, yr, _ = phase_for(step)
            env.fixed_command = (vx, vy, yr)
            if args.model and not args.random_policy:
                a, _ = model.predict(obs, deterministic=True)
                obs, _, done, _ = venv.step(a)
                done = bool(done[0])
            else:
                a = predict(obs)
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
            d = env.unwrapped.data
            ty = yaw_from_quat(d.xquat[bid["0003_8"]])
            py = yaw_from_quat(d.xquat[bid["base_link"]])
            if step >= TURN_START:
                twist.append(wrap(ty - py))
                torso_y.append(ty)
                pelvis_y.append(py)
            if done:
                break
        if twist:
            w = float(np.std(twist))
            wtrs.append(w)
            tr = np.ptp(torso_y)
            pr = np.ptp(pelvis_y)
            print(f"  ep{ep}: WTR={w:.4f}  torso_yaw_range={tr:.3f}  pelvis_yaw_range={pr:.3f}  "
                  f"ratio={tr / (pr + 1e-6):.2f} (>1 => twist hack)")
    if wtrs:
        w = np.array(wtrs)
        print(f"\nWTR mean={w.mean():.4f} std={w.std():.4f} over {len(w)} eps "
              f"(higher = more waist twist = hack-like)")


if __name__ == "__main__":
    main()
