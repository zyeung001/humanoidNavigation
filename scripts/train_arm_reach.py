#!/usr/bin/env python3
# train_arm_reach.py
"""Train the arm pose-reach policy (the sim2real closed-loop stack demo).

Small task, small net: 96-dim obs (6 arm joints x [target|jpos|jvel|last_action] x 4-frame
history), 6-dim action. Trains in a few million steps. Runs fine on CPU, so it can run
CONCURRENTLY with a GPU standing retrain without contending for the GPU.

  python scripts/train_arm_reach.py                    # 3M steps, cpu, defaults
  python scripts/train_arm_reach.py --timesteps 5000000 --device cuda

Outputs: models/final_arm_reach.zip + models/vecnorm_arm_reach.pkl
Then export for the Pi:
  python scripts/deploy/export_policy.py --model models/final_arm_reach.zip \
      --vecnorm models/vecnorm_arm_reach.pkl --out models/arm_reach_policy.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from src.environments.arm_reach_env import ArmReachEnv  # noqa: E402


class LogStdClamp(BaseCallback):
    """Position-servo action space (radians): keep exploration std bounded so the policy
    can't 'stand in training, fail deterministic' the way wide-std standing runs did."""

    def __init__(self, lo=-2.5, hi=-0.7):
        super().__init__()
        self.lo, self.hi = lo, hi

    def _on_rollout_start(self):
        self.model.policy.log_std.data.clamp_(self.lo, self.hi)

    def _on_step(self):
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-model", default=str(ROOT / "models" / "final_arm_reach.zip"))
    p.add_argument("--out-vecnorm", default=str(ROOT / "models" / "vecnorm_arm_reach.pkl"))
    args = p.parse_args()

    def make():
        return Monitor(ArmReachEnv())

    venv = DummyVecEnv([make for _ in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", venv, device=args.device, seed=args.seed, verbose=1,
        learning_rate=3e-4, n_steps=1024, batch_size=2048, n_epochs=5,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.001,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                           activation_fn=nn.SiLU, log_std_init=-1.2),
    )
    model.learn(total_timesteps=args.timesteps, callback=LogStdClamp())

    model.save(args.out_model)
    venv.save(args.out_vecnorm)
    print(f"\nSaved final model: {args.out_model}\nSaved vecnorm:     {args.out_vecnorm}")

    # quick deterministic eval on a fresh env
    venv.training = False
    venv.norm_reward = False
    obs = venv.reset()
    errs = []
    for _ in range(800):
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, infos = venv.step(act)
        errs.extend(i["track_err"] for i in infos if "track_err" in i)
    errs = np.array(errs)
    print(f"deterministic eval: track_err mean={errs.mean():.3f} rad "
          f"p95={np.percentile(errs, 95):.3f} (includes post-retarget/push transients)")


if __name__ == "__main__":
    main()
