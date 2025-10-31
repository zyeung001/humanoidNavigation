"""
Advanced PPO training for humanoid standing with curriculum, schedules, and diagnostics.
"""

import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure project root & src on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import yaml
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList

from src.environments.standing_curriculum import make_standing_curriculum_env
from src.agents.diagnostics import DiagnosticsCallback


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def lr_schedule(initial_lr: float, final_lr: float, total_steps: int):
    def schedule(progress_remaining: float):
        # SB3 passes progress_remaining in [1..0]
        step = (1.0 - progress_remaining) * total_steps
        ratio = float(np.clip(step / max(total_steps, 1), 0.0, 1.0))
        return initial_lr * (1.0 - ratio) + final_lr * ratio
    return schedule


def clip_schedule(initial: float, final: float, total_steps: int):
    def schedule(progress_remaining: float):
        step = (1.0 - progress_remaining) * total_steps
        ratio = float(np.clip(step / max(total_steps, 1), 0.0, 1.0))
        return initial * (1.0 - ratio) + final * ratio
    return schedule


def make_env_fns(n_envs: int, seed: int, cfg: dict):
    def make(rank: int):
        def _init():
            os.environ.setdefault("MUJOCO_GL", "egl")
            env = make_standing_curriculum_env(render_mode=None, config=cfg)
            if hasattr(env, 'reset'):
                env.reset(seed=seed + rank)
            try:
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
            except Exception:
                pass
            return env
        return _init

    if n_envs > 1:
        return SubprocVecEnv([make(i) for i in range(n_envs)])
    return DummyVecEnv([make(0)])


def main():
    cfg = load_yaml('config/training_config.yaml')
    standing = cfg.get('standing', {}).copy()

    # Overrides / advanced defaults
    n_envs = int(standing.get('n_envs', 8))
    seed = int(standing.get('seed', 42))
    total_timesteps = int(standing.get('total_timesteps', 2_000_000))

    # Enable curriculum & enhanced env options by default here
    standing.setdefault('curriculum_start_stage', 0)
    standing.setdefault('curriculum_max_stage', 3)
    standing.setdefault('curriculum_advance_after', 10)
    standing.setdefault('curriculum_success_rate', 0.7)
    standing.setdefault('action_smoothing', True)
    standing.setdefault('action_smoothing_tau', 0.2)
    standing.setdefault('obs_include_com', True)
    standing.setdefault('obs_feature_norm', True)
    standing.setdefault('obs_history', 4)

    # Vectorized env
    vec = make_env_fns(n_envs, seed, standing)

    # Normalization
    vecnorm_path = standing.get('vecnormalize_path', 'models/saved_models/vecnorm.pkl')
    env = VecNormalize(
        vec,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=standing.get('gamma', 0.995),
    )

    # Schedules
    initial_lr = float(standing.get('learning_rate', 3e-4))
    final_lr = float(standing.get('final_learning_rate', 1e-4))
    lr_fn = lr_schedule(initial_lr, final_lr, total_timesteps)

    initial_clip = float(standing.get('clip_range', 0.2))
    final_clip = float(standing.get('final_clip_range', 0.1))
    clip_fn = clip_schedule(initial_clip, final_clip, total_timesteps)

    # Policy/net arch
    policy_kwargs = standing.get('policy_kwargs', {
        'net_arch': [dict(pi=[512, 512, 256], vf=[512, 512, 256])],
        'activation_fn': 'SiLU',
        'ortho_init': True,
    })

    # Convert activation if needed
    import torch.nn as nn
    act_map = {"relu":"ReLU","tanh":"Tanh","sigmoid":"Sigmoid","elu":"ELU","gelu":"GELU","leakyrelu":"LeakyReLU","silu":"SiLU","mish":"Mish"}
    if isinstance(policy_kwargs.get('activation_fn'), str):
        act = policy_kwargs['activation_fn'].lower()
        policy_kwargs['activation_fn'] = getattr(nn, act_map.get(act, 'ReLU'))

    # PPO model
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=lr_fn,
        n_steps=int(standing.get('n_steps', 2048)),
        batch_size=int(standing.get('batch_size', 256)),
        n_epochs=int(standing.get('n_epochs', 10)),
        gamma=float(standing.get('gamma', 0.995)),
        gae_lambda=float(standing.get('gae_lambda', 0.95)),
        clip_range=clip_fn,
        ent_coef=float(standing.get('ent_coef', 0.0)),
        vf_coef=float(standing.get('vf_coef', 0.5)),
        max_grad_norm=float(standing.get('max_grad_norm', 0.5)),
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=int(standing.get('verbose', 1)),
        device=standing.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
    )

    # Callbacks: diagnostics + periodic vecnorm save
    class SaveVecNormCallback(DiagnosticsCallback):
        def __init__(self, path: str, freq: int = 100_000):
            super().__init__(log_freq=freq, verbose=1)
            self.path = path
            self.freq = int(freq)

        def _on_step(self) -> bool:
            ok = super()._on_step()
            if self.freq > 0 and (self.num_timesteps % self.freq == 0):
                try:
                    self.model.get_env().save(self.path)
                    print(f"✓ VecNormalize saved: {self.path}")
                except Exception as e:
                    print(f"VecNormalize save failed: {e}")
            return ok

    callbacks = CallbackList([
        SaveVecNormCallback(vecnorm_path, freq=int(standing.get('save_freq', 100_000)))
    ])

    # Train
    print(f"Starting advanced standing training for {total_timesteps:,} timesteps (n_envs={n_envs})...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Save final model + vecnorm
    os.makedirs('models/saved_models', exist_ok=True)
    final_path = standing.get('final_model_path', 'models/saved_models/final_standing_model')
    model.save(final_path)
    print(f"✓ Final model saved: {final_path}.zip")
    try:
        env.save(vecnorm_path)
        print(f"✓ VecNormalize saved: {vecnorm_path}")
    except Exception as e:
        print(f"VecNormalize save failed: {e}")


if __name__ == "__main__":
    main()


