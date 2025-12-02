# train_walking.py
"""
Training script for humanoid walking controller.
Command-conditioned on desired world velocity (vx, vy).
Uses curriculum learning from standing (0 m/s) to fast walking (3 m/s).
"""

import os
import sys
import warnings
from datetime import datetime
import argparse

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
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.environments.walking_curriculum import make_walking_curriculum_env


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def lr_schedule(initial_lr: float, final_lr: float, total_steps: int):
    def schedule(progress_remaining: float):
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
    """Create vectorized environments."""
    def make(rank: int):
        def _init():
            os.environ.setdefault("MUJOCO_GL", "egl")
            env = make_walking_curriculum_env(render_mode=None, config=cfg)
            # CRITICAL: Wrap with Monitor to track episode statistics
            # This populates info['episode'] with 'l' (length) and 'r' (reward)
            env = Monitor(env)
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


class EntropyScheduleCallback(BaseCallback):
    """
    Custom callback to schedule entropy coefficient during training.
    """
    def __init__(self, initial_ent: float, final_ent: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.total_timesteps = total_timesteps
        
    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        current_ent = self.initial_ent * (1.0 - progress) + self.final_ent * progress
        self.model.ent_coef = current_ent
        
        if self.verbose and self.num_timesteps % 50000 == 0:
            print(f"Entropy coefficient updated to: {current_ent:.6f}")
        
        return True


class WalkingMetricsCallback(BaseCallback):
    """
    Callback to log walking-specific metrics.
    """
    def __init__(self, log_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.velocity_errors = []
        self.episode_lengths = []
        self.episode_rewards = []
        self.heights = []
        
    def _on_step(self) -> bool:
        # Collect data from infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if 'velocity_error' in info:
                self.velocity_errors.append(info['velocity_error'])
            if 'height' in info:
                self.heights.append(info['height'])
            if 'episode' in info:
                self.episode_lengths.append(info['episode']['l'])
                self.episode_rewards.append(info['episode']['r'])
                
                # Print curriculum info periodically
                if len(self.episode_lengths) % 50 == 0:
                    stage = info.get('curriculum_stage', 0)
                    max_speed = info.get('curriculum_max_speed', 0)
                    vel_err = info.get('curriculum_avg_vel_error', 0)
                    success_rate = info.get('curriculum_success_rate', 0)
                    print(f"  [Stage {stage}] Max speed: {max_speed:.1f} m/s | "
                          f"Avg vel error: {vel_err:.3f} m/s | "
                          f"Success rate: {success_rate:.1%} | "
                          f"Ep len: {np.mean(self.episode_lengths[-10:]):.0f}")
        
        # Log aggregated metrics
        if self.num_timesteps % self.log_freq == 0 and self.velocity_errors:
            avg_vel_err = np.mean(self.velocity_errors[-1000:])
            avg_height = np.mean(self.heights[-1000:]) if self.heights else 0
            avg_ep_len = np.mean(self.episode_lengths[-20:]) if self.episode_lengths else 0
            avg_reward = np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0
            
            print(f"\n[Step {self.num_timesteps:,}] Walking Metrics:")
            print(f"  Avg velocity error: {avg_vel_err:.4f} m/s")
            print(f"  Avg height: {avg_height:.3f} m")
            print(f"  Avg episode length: {avg_ep_len:.0f}")
            print(f"  Avg episode reward: {avg_reward:.1f}")
            
            # Clear old data to prevent memory issues
            if len(self.velocity_errors) > 5000:
                self.velocity_errors = self.velocity_errors[-2000:]
                self.heights = self.heights[-2000:]
        
        return True


class SaveVecNormCallback(BaseCallback):
    """Callback to periodically save VecNormalize stats and model checkpoints."""
    def __init__(self, vecnorm_path: str, checkpoint_dir: str, freq: int = 100_000):
        super().__init__(verbose=1)
        self.vecnorm_path = vecnorm_path
        self.checkpoint_dir = checkpoint_dir
        self.freq = int(freq)

        from pathlib import Path
        Path(self.vecnorm_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        print(f"✓ VecNormalize will save to: {self.vecnorm_path}")
        print(f"✓ Checkpoints will save to: {self.checkpoint_dir}")

    def _on_step(self) -> bool:
        if self.freq > 0 and (self.num_timesteps % self.freq == 0):
            try:
                # Save VecNormalize
                self.model.get_env().save(self.vecnorm_path)
                print(f"✓ VecNormalize saved: {self.vecnorm_path}")
                
                # Save checkpoint
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, 
                    f"walking_{self.num_timesteps}.zip"
                )
                self.model.save(checkpoint_path)
                print(f"✓ Checkpoint saved: {checkpoint_path}")
            except Exception as e:
                print(f"✗ Save failed: {e}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Train humanoid walking controller")
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to load model from (for resuming or transfer from standing)')
    parser.add_argument('--vecnorm', type=str, default=None, 
                        help='Path to load VecNormalize from')
    parser.add_argument('--timesteps', type=int, default=None, 
                        help='Total timesteps for training')
    parser.add_argument('--reset-vecnorm', action='store_true', 
                        help='Reset VecNormalize statistics (fresh start)')
    parser.add_argument('--from-standing', action='store_true',
                        help='Initialize from standing model (handles obs dimension mismatch)')
    args = parser.parse_args()

    # Load config
    cfg = load_yaml('config/training_config.yaml')
    walking = cfg.get('walking', {}).copy()

    # Overrides / defaults
    n_envs = int(walking.get('n_envs', 8))
    seed = int(walking.get('seed', 42))
    total_timesteps = int(walking.get('total_timesteps', 15_000_000)) if args.timesteps is None else args.timesteps

    # Ensure walking-specific settings
    walking.setdefault('curriculum_start_stage', 0)
    walking.setdefault('curriculum_max_stage', 6)
    walking.setdefault('curriculum_advance_after', 20)
    walking.setdefault('curriculum_success_rate', 0.70)
    walking.setdefault('action_smoothing', True)
    walking.setdefault('action_smoothing_tau', 0.2)  # CRITICAL: Match standing
    walking.setdefault('obs_include_com', True)
    walking.setdefault('obs_feature_norm', True)
    walking.setdefault('obs_history', 4)
    walking.setdefault('velocity_weight', 5.0)
    walking.setdefault('max_commanded_speed', 0.0)  # Curriculum starts at 0

    # Create vectorized environment
    vec = make_env_fns(n_envs, seed, walking)

    # VecNormalize paths
    vecnorm_path = walking.get('vecnormalize_path', 'models/vecnorm_walking.pkl')
    env_load_path = args.vecnorm if args.vecnorm else vecnorm_path

    env = None
    vecnorm_loaded = False
    
    if not args.reset_vecnorm and os.path.exists(env_load_path):
        try:
            print(f"Attempting to load VecNormalize from: {env_load_path}")
            env = VecNormalize.load(env_load_path, vec)
            vecnorm_loaded = True
            print(f"✓ Successfully loaded VecNormalize statistics")
            print(f"  - Mean: {env.obs_rms.mean[:5]}...")
            print(f"  - Var: {env.obs_rms.var[:5]}...")
        except Exception as e:
            print(f"✗ Failed to load VecNormalize: {e}")
            print(f"  Creating fresh VecNormalize wrapper instead...")
            env = None
    
    if env is None:
        print(f"Creating new VecNormalize wrapper")
        env = VecNormalize(
            vec,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=walking.get('gamma', 0.995),
        )
        vecnorm_loaded = False

    # Schedules
    initial_lr = float(walking.get('learning_rate', 3e-4))
    final_lr = float(walking.get('final_learning_rate', 5e-5))
    lr_fn = lr_schedule(initial_lr, final_lr, total_timesteps)

    initial_clip = float(walking.get('clip_range', 0.2))
    final_clip = float(walking.get('final_clip_range', 0.1))
    clip_fn = clip_schedule(initial_clip, final_clip, total_timesteps)

    # Entropy coefficient 
    initial_ent = float(walking.get('ent_coef', 0.02))
    final_ent = float(walking.get('final_ent_coef', 0.005))
    if final_ent <= 0:
        print(f"  WARNING: final_ent_coef={final_ent} is non-positive, forcing to 0.005")
        final_ent = 0.005

    # Policy/net arch
    policy_kwargs = walking.get('policy_kwargs', {
        'net_arch': [dict(pi=[512, 512, 256], vf=[512, 512, 256])],
        'activation_fn': 'SiLU',
        'ortho_init': True,
    })

    # Convert activation if needed
    import torch.nn as nn
    act_map = {
        "relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoid", 
        "elu": "ELU", "gelu": "GELU", "leakyrelu": "LeakyReLU", 
        "silu": "SiLU", "mish": "Mish"
    }
    if isinstance(policy_kwargs.get('activation_fn'), str):
        act = policy_kwargs['activation_fn'].lower()
        policy_kwargs['activation_fn'] = getattr(nn, act_map.get(act, 'ReLU'))

    device = walking.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    resume = args.model is not None

    if resume and not args.from_standing:
        # Resume from walking model
        try:
            print(f"Loading walking model from: {args.model}")
            model = PPO.load(args.model, env=env, device=device)
            
            model.learning_rate = lr_fn
            model.clip_range = clip_fn
            
            current_timesteps = model.num_timesteps
            remaining_timesteps = total_timesteps - current_timesteps
            
            if remaining_timesteps <= 0:
                print(f"✗ Model already trained for {current_timesteps:,} steps (target: {total_timesteps:,})")
                return
            
            learn_timesteps = remaining_timesteps
            reset_num_timesteps = False
            
            print(f"✓ Resuming from model with {current_timesteps:,} timesteps")
            print(f"  Will train for {remaining_timesteps:,} more steps")
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print(f"  Starting fresh training instead...")
            resume = False
    
    if not resume:
        # Create fresh model
        print("Creating new PPO model for walking task...")
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=lr_fn,
            n_steps=int(walking.get('n_steps', 2048)),
            batch_size=int(walking.get('batch_size', 256)),
            n_epochs=int(walking.get('n_epochs', 10)),
            gamma=float(walking.get('gamma', 0.995)),
            gae_lambda=float(walking.get('gae_lambda', 0.95)),
            clip_range=clip_fn,
            ent_coef=initial_ent, 
            vf_coef=float(walking.get('vf_coef', 0.5)),
            max_grad_norm=float(walking.get('max_grad_norm', 0.5)),
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=int(walking.get('verbose', 1)),
            device=device,
        )
        learn_timesteps = total_timesteps
        reset_num_timesteps = True

    # Create callbacks
    checkpoint_dir = walking.get('checkpoint_dir', 'data/checkpoints/walking')
    callbacks = CallbackList([
        EntropyScheduleCallback(initial_ent, final_ent, learn_timesteps, verbose=1),
        WalkingMetricsCallback(log_freq=int(walking.get('wandb_log_freq', 10000)), verbose=1),
        SaveVecNormCallback(
            vecnorm_path, 
            checkpoint_dir,
            freq=int(walking.get('save_freq', 250_000))
        )
    ])

    # Train
    print(f"\n{'='*60}")
    print(f"Starting WALKING training:")
    print(f"  Mode: {'RESUME' if resume else 'FRESH START'}")
    print(f"  Training steps: {learn_timesteps:,}")
    print(f"  Target total: {total_timesteps:,}")
    print(f"  Environments: {n_envs}")
    print(f"  Device: {device}")
    print(f"  VecNormalize: {'LOADED' if vecnorm_loaded else 'NEW'}")
    print(f"  Curriculum stages: 0-{walking.get('curriculum_max_stage', 6)}")
    print(f"  Max speed range: 0.0 - 3.0 m/s")
    print(f"  Velocity weight: {walking.get('velocity_weight', 5.0)}")
    print(f"  Action smoothing tau: {walking.get('action_smoothing_tau', 0.2)}")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=learn_timesteps, 
        callback=callbacks, 
        reset_num_timesteps=reset_num_timesteps
    )

    # Save final model + vecnorm
    os.makedirs('models', exist_ok=True)
    final_path = walking.get('final_model_path', 'models/final_walking_model')
    model.save(final_path)
    print(f"✓ Final model saved: {final_path}.zip")
    
    try:
        env.save(vecnorm_path)
        print(f"✓ VecNormalize saved: {vecnorm_path}")
    except Exception as e:
        print(f"✗ VecNormalize save failed: {e}")

    print(f"\n{'='*60}")
    print("WALKING TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nTo record demo videos, run:")
    print(f"  python scripts/record_video.py --task walking --model {final_path}.zip --vecnorm {vecnorm_path} --vx_target 1.0 --vy_target 0.0")


if __name__ == "__main__":
    main()

