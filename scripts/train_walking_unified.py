#!/usr/bin/env python3
"""
train_walking_unified.py

UNIFIED training script that automatically:
1. Trains STANDING first (until agent can balance)
2. Transfers to WALKING (with velocity tracking)
3. Continues walking curriculum

This is the CORRECT way to train humanoid walking - the agent must learn
to stand before it can learn to walk.

Usage:
    python scripts/train_walking_unified.py --timesteps 10000000
    
    # Skip standing if you have a pretrained model:
    python scripts/train_walking_unified.py --skip-standing --standing-model models/best_standing_model.zip
"""

import os
import sys
import warnings
from datetime import datetime
import argparse
from pathlib import Path

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
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.environments.standing_curriculum import make_standing_curriculum_env
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


def make_standing_envs(n_envs: int, seed: int, cfg: dict):
    """Create standing training environments."""
    def make(rank: int):
        def _init():
            os.environ.setdefault("MUJOCO_GL", "egl")
            env = make_standing_curriculum_env(render_mode=None, config=cfg)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init
    
    if n_envs > 1:
        return SubprocVecEnv([make(i) for i in range(n_envs)], start_method='forkserver')
    else:
        return DummyVecEnv([make(i) for i in range(n_envs)])


def make_walking_envs(n_envs: int, seed: int, cfg: dict):
    """Create walking training environments."""
    def make(rank: int):
        def _init():
            os.environ.setdefault("MUJOCO_GL", "egl")
            env = make_walking_curriculum_env(render_mode=None, config=cfg)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init
    
    if n_envs > 1:
        return SubprocVecEnv([make(i) for i in range(n_envs)], start_method='forkserver')
    else:
        return DummyVecEnv([make(i) for i in range(n_envs)])


class StandingMasteryCallback(BaseCallback):
    """
    Callback to detect when standing is mastered and training should transition to walking.
    
    Criteria for standing mastery:
    - Average episode length > min_episode_length
    - Average height > min_height
    - Low fall rate
    """
    def __init__(
        self, 
        min_episode_length: int = 500,
        min_height: float = 1.25,
        max_fall_rate: float = 0.15,
        check_freq: int = 10000,
        min_timesteps: int = 500000,  # Minimum training before checking
        patience: int = 3,  # Consecutive successes needed
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.min_episode_length = min_episode_length
        self.min_height = min_height
        self.max_fall_rate = max_fall_rate
        self.check_freq = check_freq
        self.min_timesteps = min_timesteps
        self.patience = patience
        
        self.episode_lengths = []
        self.episode_heights = []
        self.episode_terminated = []
        self.success_count = 0
        self.standing_mastered = False
        
    def _on_step(self) -> bool:
        # Collect episode data
        infos = self.locals.get("infos", [])
        for info in infos:
            if 'episode' in info:
                self.episode_lengths.append(info['episode']['l'])
                
            if 'height' in info:
                self.episode_heights.append(info['height'])
            
            # Track terminations (falls)
            dones = self.locals.get("dones", [])
            if any(dones):
                terminated = info.get('TimeLimit.truncated', False) == False
                self.episode_terminated.append(terminated)
        
        # Check mastery periodically
        if self.num_timesteps >= self.min_timesteps and self.num_timesteps % self.check_freq == 0:
            self._check_mastery()
        
        return True
    
    def _check_mastery(self):
        if len(self.episode_lengths) < 20:
            return
        
        recent_lengths = self.episode_lengths[-50:]
        recent_heights = self.episode_heights[-500:] if self.episode_heights else [1.4]
        recent_terminated = self.episode_terminated[-50:] if self.episode_terminated else [False]
        
        avg_length = np.mean(recent_lengths)
        avg_height = np.mean(recent_heights)
        fall_rate = np.mean(recent_terminated) if recent_terminated else 0.0
        
        length_ok = avg_length >= self.min_episode_length
        height_ok = avg_height >= self.min_height
        fall_ok = fall_rate <= self.max_fall_rate
        
        if self.verbose:
            print(f"\n[Standing Check @ {self.num_timesteps:,} steps]")
            print(f"  Avg episode length: {avg_length:.0f} (need ≥{self.min_episode_length}) {'✓' if length_ok else '✗'}")
            print(f"  Avg height: {avg_height:.3f}m (need ≥{self.min_height}m) {'✓' if height_ok else '✗'}")
            print(f"  Fall rate: {fall_rate:.1%} (need ≤{self.max_fall_rate:.0%}) {'✓' if fall_ok else '✗'}")
        
        if length_ok and height_ok and fall_ok:
            self.success_count += 1
            if self.verbose:
                print(f"  ✓ Mastery check passed ({self.success_count}/{self.patience})")
            
            if self.success_count >= self.patience:
                self.standing_mastered = True
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"🎉 STANDING MASTERED! Transitioning to walking...")
                    print(f"{'='*60}\n")
                return False  # Stop training
        else:
            self.success_count = 0
            if self.verbose:
                print(f"  ✗ Not yet mastered, continuing training...")
        
        return True


class WalkingMetricsCallback(BaseCallback):
    """Callback to log walking-specific metrics."""
    def __init__(self, log_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.velocity_errors = []
        self.episode_lengths = []
        self.episode_rewards = []
        self.heights = []
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if 'velocity_error' in info:
                self.velocity_errors.append(info['velocity_error'])
            if 'height' in info:
                self.heights.append(info['height'])
            if 'episode' in info:
                self.episode_lengths.append(info['episode']['l'])
                self.episode_rewards.append(info['episode']['r'])
                
                if len(self.episode_lengths) % 50 == 0:
                    stage = info.get('curriculum_stage', 0)
                    vel_err = info.get('curriculum_avg_vel_error', 0)
                    success_rate = info.get('curriculum_success_rate', 0)
                    print(f"  [Stage {stage}] Vel err: {vel_err:.3f} m/s | "
                          f"Success: {success_rate:.1%} | "
                          f"Ep len: {np.mean(self.episode_lengths[-10:]):.0f}")
        
        if self.num_timesteps % self.log_freq == 0 and self.velocity_errors:
            avg_vel_err = np.mean(self.velocity_errors[-1000:])
            avg_height = np.mean(self.heights[-1000:]) if self.heights else 0
            avg_ep_len = np.mean(self.episode_lengths[-20:]) if self.episode_lengths else 0
            
            print(f"\n[Step {self.num_timesteps:,}] Walking Metrics:")
            print(f"  Avg velocity error: {avg_vel_err:.4f} m/s")
            print(f"  Avg height: {avg_height:.3f} m")
            print(f"  Avg episode length: {avg_ep_len:.0f}")
            
            if len(self.velocity_errors) > 5000:
                self.velocity_errors = self.velocity_errors[-2000:]
                self.heights = self.heights[-2000:]
        
        return True


def transfer_weights(standing_model, walking_model, device):
    """Transfer compatible weights from standing to walking model."""
    standing_state = standing_model.policy.state_dict()
    walking_state = walking_model.policy.state_dict()
    
    transferred = 0
    for key in walking_state.keys():
        if key in standing_state:
            standing_tensor = standing_state[key]
            walking_tensor = walking_state[key]
            
            if standing_tensor.shape == walking_tensor.shape:
                walking_state[key] = standing_tensor
                transferred += 1
            elif len(standing_tensor.shape) == 2 and len(walking_tensor.shape) == 2:
                # Partial transfer for first layer (different input sizes)
                min_in = min(standing_tensor.shape[1], walking_tensor.shape[1])
                min_out = min(standing_tensor.shape[0], walking_tensor.shape[0])
                walking_state[key][:min_out, :min_in] = standing_tensor[:min_out, :min_in]
                
                # Initialize new dimensions with small random values
                if walking_tensor.shape[1] > standing_tensor.shape[1]:
                    new_dim = walking_tensor.shape[1] - standing_tensor.shape[1]
                    walking_state[key][:, -new_dim:] = torch.randn(
                        walking_tensor.shape[0], new_dim, device=device
                    ) * 0.01
                transferred += 1
            elif len(standing_tensor.shape) == 1 and len(walking_tensor.shape) == 1:
                min_size = min(standing_tensor.shape[0], walking_tensor.shape[0])
                walking_state[key][:min_size] = standing_tensor[:min_size]
                transferred += 1
    
    walking_model.policy.load_state_dict(walking_state)
    return transferred


def main():
    parser = argparse.ArgumentParser(description="Unified humanoid walking training (standing → walking)")
    parser.add_argument('--timesteps', type=int, default=10_000_000, 
                        help='Total timesteps for WALKING training (standing is automatic)')
    parser.add_argument('--standing-timesteps', type=int, default=3_000_000,
                        help='Maximum timesteps for standing phase (will stop early if mastered)')
    parser.add_argument('--skip-standing', action='store_true',
                        help='Skip standing training (use with --standing-model)')
    parser.add_argument('--standing-model', type=str, default=None,
                        help='Path to pretrained standing model (to skip standing training)')
    parser.add_argument('--n-envs', type=int, default=8,
                        help='Number of parallel environments')
    parser.add_argument('--debug', action='store_true',
                        help='Use DummyVecEnv for easier debugging')
    args = parser.parse_args()

    # Load config
    cfg = load_yaml('config/training_config.yaml')
    standing_cfg = cfg.get('standing', {}).copy()
    walking_cfg = cfg.get('walking', {}).copy()
    
    n_envs = args.n_envs
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup paths
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    standing_model_path = models_dir / 'standing_for_walking.zip'
    standing_vecnorm_path = models_dir / 'standing_vecnorm.pkl'
    
    print(f"\n{'='*70}")
    print("UNIFIED HUMANOID WALKING TRAINING")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Parallel environments: {n_envs}")
    print(f"Standing phase: {'SKIP' if args.skip_standing else f'up to {args.standing_timesteps:,} steps'}")
    print(f"Walking phase: {args.timesteps:,} steps")
    print(f"{'='*70}\n")
    
    standing_model = None
    
    # ========== PHASE 1: STANDING ==========
    if not args.skip_standing:
        print(f"\n{'='*70}")
        print("PHASE 1: STANDING TRAINING")
        print("Agent must learn to balance before it can walk")
        print(f"{'='*70}\n")
        
        # Configure standing environment
        standing_cfg['curriculum_start_stage'] = 0
        standing_cfg['curriculum_max_stage'] = 3
        standing_cfg['obs_history'] = 4
        standing_cfg['obs_include_com'] = True
        standing_cfg['action_smoothing'] = True
        standing_cfg['action_smoothing_tau'] = 0.3
        
        # Create standing environment
        standing_vec = make_standing_envs(n_envs, seed, standing_cfg)
        standing_env = VecNormalize(
            standing_vec,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.995,
        )
        
        # Create standing model
        policy_kwargs = {
            'net_arch': [dict(pi=[512, 512, 256], vf=[512, 512, 256])],
            'activation_fn': torch.nn.SiLU,
            'ortho_init': True,
        }
        
        lr_fn = lr_schedule(3e-4, 5e-5, args.standing_timesteps)
        
        standing_model = PPO(
            policy='MlpPolicy',
            env=standing_env,
            learning_rate=lr_fn,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=1,
            device=device,
        )
        
        # Standing mastery callback
        mastery_callback = StandingMasteryCallback(
            min_episode_length=400,
            min_height=1.20,
            max_fall_rate=0.20,
            check_freq=20000,
            min_timesteps=300000,
            patience=3,
            verbose=1
        )
        
        print("Training standing...")
        try:
            standing_model.learn(
                total_timesteps=args.standing_timesteps,
                callback=mastery_callback,
            )
        except KeyboardInterrupt:
            print("\nStanding training interrupted, continuing to walking...")
        
        if mastery_callback.standing_mastered:
            print("✓ Standing mastered via callback criteria")
        else:
            print("✓ Standing training completed (max timesteps reached)")
        
        # Save standing model
        standing_model.save(str(standing_model_path))
        standing_env.save(str(standing_vecnorm_path))
        print(f"✓ Standing model saved: {standing_model_path}")
        
        # Close standing environment
        standing_env.close()
        
    elif args.standing_model:
        # Load pretrained standing model
        print(f"Loading pretrained standing model: {args.standing_model}")
        standing_model = PPO.load(args.standing_model, device=device)
        print("✓ Standing model loaded")
    else:
        print("ERROR: --skip-standing requires --standing-model")
        print("Usage: python train_walking_unified.py --skip-standing --standing-model models/standing.zip")
        return
    
    # ========== PHASE 2: TRANSFER & WALKING ==========
    print(f"\n{'='*70}")
    print("PHASE 2: WALKING TRAINING")
    print("Transferring standing knowledge and learning velocity tracking")
    print(f"{'='*70}\n")
    
    # Configure walking environment
    walking_cfg['curriculum_start_stage'] = 0
    walking_cfg['curriculum_max_stage'] = 6
    walking_cfg['obs_history'] = 4
    walking_cfg['obs_include_com'] = True
    walking_cfg['action_smoothing'] = True
    walking_cfg['action_smoothing_tau'] = 0.2
    walking_cfg['push_enabled'] = False  # Start without pushes
    walking_cfg['domain_rand'] = False
    
    # Create walking environment
    walking_vec = make_walking_envs(n_envs, seed, walking_cfg)
    walking_env = VecNormalize(
        walking_vec,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995,
    )
    
    # Create walking model
    policy_kwargs = {
        'net_arch': [dict(pi=[512, 512, 256], vf=[512, 512, 256])],
        'activation_fn': torch.nn.SiLU,
        'ortho_init': True,
    }
    
    lr_fn = lr_schedule(2e-4, 5e-5, args.timesteps)
    clip_fn = clip_schedule(0.2, 0.1, args.timesteps)
    
    walking_model = PPO(
        policy='MlpPolicy',
        env=walking_env,
        learning_rate=lr_fn,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=clip_fn,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
        device=device,
    )
    
    # Transfer weights from standing to walking
    print("Transferring standing weights to walking model...")
    if standing_model is not None:
        transferred = transfer_weights(standing_model, walking_model, device)
        print(f"✓ Transferred {transferred} layers from standing model")
        print("  Walking model now has balance knowledge!")
    
    # Walking callbacks
    walking_callback = WalkingMetricsCallback(log_freq=10000, verbose=1)
    
    print(f"\nTraining walking for {args.timesteps:,} steps...")
    try:
        walking_model.learn(
            total_timesteps=args.timesteps,
            callback=walking_callback,
        )
    except KeyboardInterrupt:
        print("\nWalking training interrupted, saving model...")
    
    # Save final walking model
    final_path = models_dir / 'final_walking_unified.zip'
    vecnorm_path = models_dir / 'vecnorm_walking_unified.pkl'
    walking_model.save(str(final_path))
    walking_env.save(str(vecnorm_path))
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Final model saved: {final_path}")
    print(f"VecNormalize saved: {vecnorm_path}")
    print(f"\nTo evaluate:")
    print(f"  python scripts/evaluate.py --task walking --model {final_path} --vx 0.5 --vy 0.0")
    
    walking_env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        raise

