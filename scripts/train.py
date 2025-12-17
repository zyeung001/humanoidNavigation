#!/usr/bin/env python3
# train.py
"""
Unified training script for humanoid locomotion.

Supports both standing and walking tasks via --task flag.

Usage:
    python scripts/train.py --task walking --timesteps 15000000
    python scripts/train.py --task standing --timesteps 10000000
    python scripts/train.py --task walking --model models/walking/latest/model.zip --resume
"""

import os
import sys
import argparse
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor


def load_config(path: str = "config/training_config.yaml") -> dict:
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def linear_schedule(initial: float, final: float, total_steps: int):
    """Create linear schedule function for learning rate or clip range."""
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        return initial * (1.0 - progress) + final * progress
    return schedule


class EntropyScheduleCallback(BaseCallback):
    """Schedule entropy coefficient during training."""
    
    def __init__(self, initial_ent: float, final_ent: float, total_timesteps: int):
        super().__init__()
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.total_timesteps = total_timesteps
        
    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        self.model.ent_coef = self.initial_ent * (1 - progress) + self.final_ent * progress
        return True


class MetricsCallback(BaseCallback):
    """Log task-specific metrics."""
    
    def __init__(self, task: str, log_freq: int = 10000):
        super().__init__()
        self.task = task
        self.log_freq = log_freq
        self.metrics = {'velocity_error': [], 'height': [], 'ep_len': [], 'ep_rew': []}
        
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if 'velocity_error' in info:
                self.metrics['velocity_error'].append(info['velocity_error'])
            if 'height' in info:
                self.metrics['height'].append(info['height'])
            if 'episode' in info:
                self.metrics['ep_len'].append(info['episode']['l'])
                self.metrics['ep_rew'].append(info['episode']['r'])
                
                # Print curriculum info for walking
                if self.task == 'walking' and len(self.metrics['ep_len']) % 50 == 0:
                    stage = info.get('curriculum_stage', 0)
                    max_speed = info.get('curriculum_max_speed', 0)
                    print(f"  [Stage {stage}] Max speed: {max_speed:.1f} m/s | "
                          f"Ep len: {np.mean(self.metrics['ep_len'][-10:]):.0f}")
        
        if self.num_timesteps % self.log_freq == 0 and self.metrics['velocity_error']:
            print(f"\n[{self.num_timesteps:,}] "
                  f"vel_err: {np.mean(self.metrics['velocity_error'][-1000:]):.3f} | "
                  f"height: {np.mean(self.metrics['height'][-1000:]):.3f} | "
                  f"ep_len: {np.mean(self.metrics['ep_len'][-20:]):.0f}")
            
            # Cleanup old data
            if len(self.metrics['velocity_error']) > 5000:
                for k in self.metrics:
                    self.metrics[k] = self.metrics[k][-2000:]
        
        return True


class CheckpointCallback(BaseCallback):
    """Save model and VecNormalize periodically."""
    
    def __init__(self, save_dir: str, vecnorm_path: str, freq: int = 250000):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.vecnorm_path = vecnorm_path
        self.freq = freq
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0:
            # Save checkpoint
            ckpt_path = self.save_dir / f"model_{self.num_timesteps // 1000}k.zip"
            self.model.save(str(ckpt_path))
            
            # Save VecNormalize
            try:
                self.model.get_env().save(self.vecnorm_path)
            except Exception:
                pass
            
            print(f"✓ Checkpoint saved: {ckpt_path}")
        
        return True


def make_env(task: str, config: dict, rank: int = 0, seed: int = 42):
    """Create environment for given task."""
    def _init():
        os.environ.setdefault("MUJOCO_GL", "egl")
        
        if task == "walking":
            from src.environments import make_walking_curriculum_env
            env = make_walking_curriculum_env(render_mode=None, config=config)
        elif task == "standing":
            from src.environments import make_standing_curriculum_env
            env = make_standing_curriculum_env(render_mode=None, config=config)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    
    return _init


def create_vec_env(task: str, config: dict, n_envs: int, seed: int):
    """Create vectorized environment."""
    env_fns = [make_env(task, config, rank=i, seed=seed) for i in range(n_envs)]
    
    if n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def main():
    parser = argparse.ArgumentParser(description="Train humanoid locomotion")
    parser.add_argument('--task', type=str, required=True, choices=['walking', 'standing'],
                        help='Task to train (walking or standing)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps for training')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model for resuming (without .zip)')
    parser.add_argument('--vecnorm', type=str, default=None,
                        help='Path to VecNormalize for resuming')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--n-envs', type=int, default=None,
                        help='Override number of parallel environments')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Load config
    full_config = load_config(args.config)
    config = full_config.get(args.task, {}).copy()
    
    # Override from args
    n_envs = args.n_envs or int(config.get('n_envs', 8))
    seed = int(config.get('seed', 42))
    total_timesteps = args.timesteps or int(config.get('total_timesteps', 10_000_000))
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    model_dir = Path(f"models/{args.task}")
    model_dir.mkdir(parents=True, exist_ok=True)
    vecnorm_path = args.vecnorm or str(model_dir / "latest" / "vecnorm.pkl")
    checkpoint_dir = config.get('checkpoint_dir', f"data/checkpoints/{args.task}")
    
    print(f"\n{'='*60}")
    print(f"HUMANOID {args.task.upper()} TRAINING")
    print(f"{'='*60}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Environments: {n_envs}")
    print(f"  Device: {device}")
    print(f"  Resume: {args.resume}")
    print(f"{'='*60}\n")

    # Create environment
    vec_env = create_vec_env(args.task, config, n_envs, seed)
    
    # Load or create VecNormalize
    if args.resume and args.vecnorm and os.path.exists(args.vecnorm):
        print(f"Loading VecNormalize from: {args.vecnorm}")
        env = VecNormalize.load(args.vecnorm, vec_env)
    else:
        env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=float(config.get('vecnormalize_clip_obs', 10.0)),
            clip_reward=float(config.get('vecnormalize_clip_reward', 10.0)),
            gamma=float(config.get('gamma', 0.995)),
        )

    # Schedules
    initial_lr = float(config.get('learning_rate', 3e-4))
    final_lr = float(config.get('final_learning_rate', 5e-5))
    lr_fn = linear_schedule(initial_lr, final_lr, total_timesteps)
    
    initial_clip = float(config.get('clip_range', 0.2))
    final_clip = float(config.get('final_clip_range', 0.1))
    clip_fn = linear_schedule(initial_clip, final_clip, total_timesteps)
    
    # Policy architecture
    policy_kwargs = config.get('policy_kwargs', {
        'net_arch': [dict(pi=[512, 512, 256], vf=[512, 512, 256])],
    })
    
    # Convert activation string to class
    if isinstance(policy_kwargs.get('activation_fn'), str):
        import torch.nn as nn
        act_map = {'silu': nn.SiLU, 'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}
        policy_kwargs['activation_fn'] = act_map.get(
            policy_kwargs['activation_fn'].lower(), nn.SiLU
        )

    # Create or load model
    if args.resume and args.model:
        model_path = args.model if args.model.endswith('.zip') else f"{args.model}.zip"
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path, env=env, device=device)
        model.learning_rate = lr_fn
        model.clip_range = clip_fn
        
        remaining = total_timesteps - model.num_timesteps
        print(f"Resuming from {model.num_timesteps:,} timesteps, {remaining:,} remaining")
        learn_timesteps = remaining
        reset_num_timesteps = False
    else:
        print("Creating new PPO model...")
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=lr_fn,
            n_steps=int(config.get('n_steps', 2048)),
            batch_size=int(config.get('batch_size', 256)),
            n_epochs=int(config.get('n_epochs', 10)),
            gamma=float(config.get('gamma', 0.995)),
            gae_lambda=float(config.get('gae_lambda', 0.95)),
            clip_range=clip_fn,
            ent_coef=float(config.get('ent_coef', 0.02)),
            vf_coef=float(config.get('vf_coef', 0.5)),
            max_grad_norm=float(config.get('max_grad_norm', 0.5)),
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=1,
            device=device,
        )
        learn_timesteps = total_timesteps
        reset_num_timesteps = True

    # Callbacks
    initial_ent = float(config.get('ent_coef', 0.02))
    final_ent = float(config.get('final_ent_coef', 0.005))
    
    callbacks = CallbackList([
        EntropyScheduleCallback(initial_ent, final_ent, learn_timesteps),
        MetricsCallback(args.task, log_freq=int(config.get('wandb_log_freq', 10000))),
        CheckpointCallback(checkpoint_dir, vecnorm_path, 
                          freq=int(config.get('save_freq', 250000))),
    ])

    # Train
    print(f"\nStarting training for {learn_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=learn_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps
    )

    # Save final model
    final_dir = model_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(final_dir / "model"))
    env.save(str(final_dir / "vecnorm.pkl"))
    
    # Also save to latest
    latest_dir = model_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(latest_dir / "model"))
    env.save(str(latest_dir / "vecnorm.pkl"))
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"  Final model: {final_dir / 'model.zip'}")
    print(f"  VecNormalize: {final_dir / 'vecnorm.pkl'}")
    print(f"\nTo evaluate:")
    print(f"  python scripts/evaluate.py --task {args.task} --model {final_dir / 'model.zip'}")


if __name__ == "__main__":
    main()

