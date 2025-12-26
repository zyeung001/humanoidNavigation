# train_walking.py
"""
Training script for humanoid walking controller
Command-conditioned on desired world velocity (vx, vy)
Uses curriculum learning from standing (0 m/s) to fast walking (3 m/s)

Integrates:
- ModelManager for organized checkpoint storage
- VelocityTrackingWandBCallback for comprehensive logging
- RewardCalculator (via walking_env.py)
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
from src.training.model_manager import ModelManager
from src.training.callbacks import (
    VelocityTrackingWandBCallback, 
    CurriculumWandBCallback,
    init_wandb_run,
    finish_wandb_run
)


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


def make_env_fns(n_envs: int, seed: int, cfg: dict, use_subproc: bool = True):
    """
    Create vectorized environments.
    
    Args:
        n_envs: Number of parallel environments
        seed: Random seed
        cfg: Environment config
        use_subproc: If True, use SubprocVecEnv for parallelization (recommended)
                    If False, use DummyVecEnv (for debugging)
    """
    def make(rank: int):
        def _init():
            os.environ.setdefault("MUJOCO_GL", "egl")
            try:
                env = make_walking_curriculum_env(render_mode=None, config=cfg)
                # Wrap with Monitor to track episode statistics
                env = Monitor(env)
                if hasattr(env, 'reset'):
                    env.reset(seed=seed + rank)
                try:
                    env.action_space.seed(seed + rank)
                    env.observation_space.seed(seed + rank)
                except Exception:
                    pass
                return env
            except Exception as e:
                print(f"ERROR creating env {rank}: {e}")
                import traceback
                traceback.print_exc()
                raise
        return _init

    if n_envs > 1 and use_subproc:
        print(f"Creating {n_envs} parallel environments with SubprocVecEnv...")
        return SubprocVecEnv([make(i) for i in range(n_envs)], start_method='forkserver')
    else:
        print(f"Creating {n_envs} environments with DummyVecEnv (sequential)...")
        return DummyVecEnv([make(i) for i in range(n_envs)])


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


class SaveWithModelManagerCallback(BaseCallback):
    """Callback to save checkpoints using ModelManager."""
    def __init__(self, model_manager: ModelManager, freq: int = 100_000):
        super().__init__(verbose=1)
        self.model_manager = model_manager
        self.freq = int(freq)
        self.best_vel_error = float('inf')
        self.recent_vel_errors = []

    def _on_step(self) -> bool:
        # Track velocity errors for best model detection
        for info in self.locals.get("infos", []):
            if 'velocity_error' in info:
                self.recent_vel_errors.append(info['velocity_error'])
                if len(self.recent_vel_errors) > 1000:
                    self.recent_vel_errors = self.recent_vel_errors[-500:]
        
        if self.freq > 0 and (self.num_timesteps % self.freq == 0):
            try:
                env = self.model.get_env()
                
                # Get current curriculum stage from env
                stage = 0
                try:
                    stage = env.envs[0].stage if hasattr(env.envs[0], 'stage') else 0
                except Exception:
                    pass
                
                # Calculate average velocity error
                avg_vel_error = np.mean(self.recent_vel_errors) if self.recent_vel_errors else float('inf')
                
                # Save checkpoint with stage info
                self.model_manager.save_checkpoint(
                    self.model, env, 
                    timesteps=self.num_timesteps,
                    stage=stage,
                    velocity_error=avg_vel_error
                )
                
                # Save latest
                self.model_manager.save_latest(self.model, env, timesteps=self.num_timesteps)
                
                # Check if this is the best model
                if avg_vel_error < self.best_vel_error:
                    self.model_manager.save_best(
                        self.model, env,
                        metric=avg_vel_error,
                        timesteps=self.num_timesteps,
                        metric_name="velocity_error"
                    )
                    self.best_vel_error = avg_vel_error
                    
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
    parser.add_argument('--debug', action='store_true',
                        help='Use DummyVecEnv for easier debugging (no multiprocessing)')
    parser.add_argument('--n-envs', type=int, default=None,
                        help='Number of parallel environments (default: from config, use 4-8 for Colab)')
    args = parser.parse_args()

    # Load config
    cfg = load_yaml('config/training_config.yaml')
    walking = cfg.get('walking', {}).copy()

    # Overrides / defaults
    n_envs = args.n_envs if args.n_envs is not None else int(walking.get('n_envs', 8))
    seed = int(walking.get('seed', 42))
    total_timesteps = int(walking.get('total_timesteps', 15_000_000)) if args.timesteps is None else args.timesteps
    
    print(f"\n{'='*60}")
    print(f"WALKING TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Device: {walking.get('device', 'cuda')}")
    print(f"{'='*60}\n")

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
    use_subproc = not args.debug  # Use DummyVecEnv if --debug flag is set
    vec = make_env_fns(n_envs, seed, walking, use_subproc=use_subproc)

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
    
    # ========== INITIALIZE MODEL MANAGER ==========
    model_manager = ModelManager(task="walking", base_dir="models")
    model_manager.archive_config(walking, run_name=f"walking_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # ========== INITIALIZE WANDB (if enabled) ==========
    use_wandb = walking.get('use_wandb', False)
    if use_wandb:
        init_wandb_run(
            project=walking.get('wandb_project', 'humanoid_walking'),
            name=f"walking_{datetime.now().strftime('%m%d_%H%M')}",
            config=walking,
            tags=['walking', 'curriculum']
        )

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
    callback_list = [
        EntropyScheduleCallback(initial_ent, final_ent, learn_timesteps, verbose=1),
        WalkingMetricsCallback(log_freq=int(walking.get('wandb_log_freq', 10000)), verbose=1),
        SaveWithModelManagerCallback(
            model_manager=model_manager,
            freq=int(walking.get('save_freq', 250_000))
        )
    ]
    
    # Add WandB callbacks if enabled
    if use_wandb:
        callback_list.extend([
            VelocityTrackingWandBCallback(
                log_freq=int(walking.get('wandb_log_freq', 5000)),
                project_name=walking.get('wandb_project', 'humanoid_walking'),
                config=walking
            ),
            CurriculumWandBCallback(log_freq=5000)
        ])
    
    callbacks = CallbackList(callback_list)

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

    # Save final model using ModelManager
    model_manager.save_final(model, env)
    model_manager.save_latest(model, env, timesteps=model.num_timesteps)
    
    # Also save to legacy paths for backwards compatibility
    os.makedirs('models', exist_ok=True)
    final_path = walking.get('final_model_path', 'models/final_walking_model')
    model.save(final_path)
    try:
        env.save(vecnorm_path)
    except Exception:
        pass
    
    # Finish WandB run
    if use_wandb:
        finish_wandb_run()
    
    # Print summary
    best_info = model_manager.get_best_info()
    print(f"\n{'='*60}")
    print("WALKING TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nModel locations:")
    print(f"  Final:  {model_manager.final_dir / 'model.zip'}")
    print(f"  Best:   {model_manager.best_dir / 'model.zip'} (vel_error: {best_info['metric']:.4f})")
    print(f"  Latest: {model_manager.latest_dir / 'model.zip'}")
    print(f"\nTo record demo videos, run:")
    print(f"  python scripts/evaluate.py --task walking --model {model_manager.best_dir / 'model.zip'} --record --vx 1.0 --vy 0.0")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("   To resume, use: python scripts/train_walking.py --model models/walking/latest/model.zip")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

