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

from src.utils import configure_mujoco_gl, get_subprocess_start_method
configure_mujoco_gl()

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor

# Fix SB3 version mismatch: VecNormalize.__getstate__ crashes if class_attributes missing
# Must check __dict__ directly — hasattr triggers __getattr__ → infinite recursion
_orig_getstate = VecNormalize.__getstate__
def _safe_getstate(self):
    if 'class_attributes' not in self.__dict__:
        self.__dict__['class_attributes'] = {}
    return _orig_getstate(self)
VecNormalize.__getstate__ = _safe_getstate

from src.environments.walking_curriculum import make_walking_curriculum_env
from src.training.model_manager import ModelManager
from src.training.callbacks import (
    VelocityTrackingWandBCallback,
    CurriculumWandBCallback,
    RewardBreakdownWandBCallback,
    PPOMetricsWandBCallback,
    init_wandb_run,
    finish_wandb_run
)
from src.training.transfer_utils import (
    transfer_standing_to_walking,
)


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def lr_schedule(initial_lr: float, final_lr: float, total_steps: int = 0):
    """Linear decay from initial_lr to final_lr. Uses SB3's progress_remaining (1→0)."""
    def schedule(progress_remaining: float):
        return initial_lr * progress_remaining + final_lr * (1.0 - progress_remaining)
    return schedule


def clip_schedule(initial: float, final: float, total_steps: int = 0):
    """Linear decay from initial to final. Uses SB3's progress_remaining (1→0)."""
    def schedule(progress_remaining: float):
        return initial * progress_remaining + final * (1.0 - progress_remaining)
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
            configure_mujoco_gl()
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
        start_method = get_subprocess_start_method()
        print(f"Creating {n_envs} parallel environments with SubprocVecEnv (start_method={start_method})...")
        return SubprocVecEnv([make(i) for i in range(n_envs)], start_method=start_method)
    else:
        print(f"Creating {n_envs} environments with DummyVecEnv (sequential)...")
        return DummyVecEnv([make(i) for i in range(n_envs)])


class ValueFunctionWarmupCallback(BaseCallback):
    """
    Three-phase warmup for standing -> walking transfer, with extra VF training.

    Phase 1 (0 to warmup_steps): Policy frozen (requires_grad=False).
        Only value function trains. KL=0 since policy doesn't change.

    Phase 2 (warmup_steps to warmup_steps + rampup_steps): Gradual unfreeze.
        Policy params are unfrozen but updates are scaled by a factor that
        ramps from 0 to max_scale.

    Phase 3 (warmup_steps + rampup_steps onward): Permanent scaling.
        Policy updates are permanently scaled by max_scale. With 17 action
        dims, full-speed PPO updates overshoot the clip boundary at ANY
        learning rate (tested 3e-4 to 5e-6, all produce >95% clip fraction).
        Permanent scaling keeps applied updates conservative while letting
        Adam compute correct gradients and momentum.

    Extra VF Training (all phases):
        Full-batch PPO (batch_size=24576) gives clean policy gradients but
        only 3 gradient steps per cycle. Clipping only affects policy loss,
        NOT value loss (VF uses plain MSE). But with only 3 steps, the VF
        can't keep up with the changing policy — explained_variance declines.
        After each model.train(), we run extra_vf_epochs additional VF-only
        gradient steps using the same rollout buffer. This gives the VF
        3 + extra_vf_epochs steps per cycle while the policy stays at 3.

    The value function always trains at full speed (unaffected by scaling).
    log_std is excluded from interpolation (managed by LogStdClampCallback).
    """
    def __init__(self, warmup_steps: int = 250_000, rampup_steps: int = 500_000,
                 max_scale: float = 0.3, extra_vf_epochs: int = 7, verbose: int = 0):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps
        self.rampup_steps = rampup_steps
        self.max_scale = max_scale
        self.extra_vf_epochs = extra_vf_epochs
        self._phase = 'init'  # init -> frozen -> ramping -> steady
        self._policy_param_names = set()
        self._saved_state = None

    def _on_training_start(self) -> None:
        for name, param in self.model.policy.named_parameters():
            is_value = 'vf' in name or 'value' in name
            if not is_value:
                self._policy_param_names.add(name)
                param.requires_grad = False
        self._phase = 'frozen'
        n_total = sum(1 for _ in self.model.policy.parameters())
        print(f"  [VFWarmup] Frozen {len(self._policy_param_names)}/{n_total} params (policy network)")
        print(f"  [VFWarmup] Phase 1: Value-only training for {self.warmup_steps:,} steps")
        print(f"  [VFWarmup] Phase 2: Policy ramp 0->{self.max_scale:.1f} over {self.rampup_steps:,} steps")
        print(f"  [VFWarmup] Phase 3: Permanent scaling at {self.max_scale:.1f}")

        # Monkey-patch model.train() to add extra VF training after each PPO update.
        # After model.train() returns, the rollout buffer is still full (buffer.reset()
        # only happens at the START of the next collect_rollouts()). So we can call
        # buffer.get() again for additional VF-only gradient steps.
        # Use the unbound class method directly to avoid capturing an already-patched instance method
        if self.extra_vf_epochs > 0 and not getattr(self.model, '_extra_vf_patched', False):
            _ppo_train = type(self.model).train  # unbound class method — never patched
            _model = self.model
            _callback = self
            def _train_with_extra_vf():
                _ppo_train(_model)
                _callback._extra_vf_training()
            self.model.train = _train_with_extra_vf
            self.model._extra_vf_patched = True
            print(f"  [VFWarmup] Extra VF training: {self.extra_vf_epochs} epochs after each train()")

    def _extra_vf_training(self):
        """Run additional VF-only gradient steps using the current rollout buffer.

        Called after model.train() via monkey-patch. At this point:
        - The rollout buffer is still full (reset() hasn't been called yet)
        - PPO's train() has already done n_epochs gradient steps on both
          policy and VF
        - We do extra_vf_epochs more steps for the VF only, compensating
          for the reduced gradient steps from full-batch training

        SB3's PPO clipping only affects policy loss, not VF loss. So the VF
        would benefit from more gradient steps even when clip_fraction is high.
        But full-batch (batch_size=24576) limits both to 3 steps. This method
        gives the VF the extra updates it needs.
        """
        model = self.model
        buffer = model.rollout_buffer

        if not buffer.full:
            return

        # Save requires_grad state for all params, then freeze non-VF params.
        # This is safe across all phases:
        #   Phase 1: policy already frozen — we restore the frozen state after
        #   Phase 2/3: policy unfrozen — we freeze temporarily, then restore
        grad_state = {}
        for name, param in model.policy.named_parameters():
            grad_state[name] = param.requires_grad
            is_value = 'vf' in name or 'value' in name
            param.requires_grad = is_value

        model.policy.set_training_mode(True)

        total_loss = 0.0
        for _epoch in range(self.extra_vf_epochs):
            for data in buffer.get(batch_size=None):
                values = model.policy.predict_values(data.observations)
                vf_loss = F.mse_loss(data.returns, values.flatten())

                model.policy.optimizer.zero_grad()
                (model.vf_coef * vf_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
                model.policy.optimizer.step()

                total_loss += vf_loss.item()

        # Restore original requires_grad state
        for name, param in model.policy.named_parameters():
            param.requires_grad = grad_state[name]

        model.policy.set_training_mode(False)

        avg_loss = total_loss / max(self.extra_vf_epochs, 1)
        if self.verbose and self.num_timesteps % 100_000 < 25_000:
            print(f"  [VFWarmup] Extra VF: {self.extra_vf_epochs} epochs, "
                  f"avg_loss={avg_loss:.6f}")

    def _on_rollout_start(self) -> None:
        # Called AFTER train() of previous cycle — scale back policy updates
        if self._phase not in ('ramping', 'steady') or self._saved_state is None:
            return

        if self._phase == 'ramping':
            elapsed = self.num_timesteps - self.warmup_steps
            progress = min(elapsed / max(self.rampup_steps, 1), 1.0)
            scale = progress * self.max_scale  # 0 -> max_scale

            if progress >= 1.0:
                self._phase = 'steady'
                scale = self.max_scale
                print(f"\n  [VFWarmup] Phase 3: Steady scale={self.max_scale:.2f} "
                      f"at step {self.num_timesteps:,}")
                # Reset Adam state for policy params to clear ramp-era momentum.
                # During the ramp, Adam built momentum against scaled-back updates.
                # Without reset, that momentum causes overshoot at steady state.
                self._reset_policy_optimizer()
        else:
            # Phase 'steady' — fixed scale forever
            scale = self.max_scale

        with torch.no_grad():
            for name, param in self.model.policy.named_parameters():
                if name in self._saved_state:
                    old = self._saved_state[name]
                    param.data.copy_(old + scale * (param.data - old))
                    self._saved_state[name] = param.data.clone()

        if self.verbose and self.num_timesteps % 100_000 < 25_000:
            print(f"  [VFWarmup] scale={scale:.3f} step={self.num_timesteps:,} ({self._phase})")

    def _reset_policy_optimizer(self):
        """Reset Adam state for policy params to prevent momentum artifacts from ramp."""
        optimizer = self.model.policy.optimizer
        reset_count = 0
        for name, param in self.model.policy.named_parameters():
            if name in self._saved_state and param in optimizer.state:
                del optimizer.state[param]
                reset_count += 1
        print(f"  [VFWarmup] Reset optimizer state for {reset_count} policy params")

    def _on_step(self) -> bool:
        if self._phase == 'frozen' and self.num_timesteps >= self.warmup_steps:
            # Transition: frozen -> ramping
            self._saved_state = {}
            for name, param in self.model.policy.named_parameters():
                if name in self._policy_param_names:
                    param.requires_grad = True
                    if 'log_std' not in name:
                        self._saved_state[name] = param.data.clone()
            self._phase = 'ramping'
            print(f"\n  [VFWarmup] Phase 2: Ramp starting at step {self.num_timesteps:,}")
        return True


class PermanentPolicyScalingCallback(BaseCallback):
    """
    Permanently scales policy updates by max_scale after each optimizer step.

    Use when RESUMING a walking training run (not from-standing transfer).
    This is the Phase 3 behavior of VFWarmup, extracted as a standalone callback
    so it can be applied during resume without re-running VF warmup/ramp.

    Includes extra VF training (same as VFWarmup) to compensate for reduced
    gradient steps from full-batch training.

    The value function always gets full updates. log_std is excluded.
    """
    def __init__(self, max_scale: float = 0.3, extra_vf_epochs: int = 7, verbose: int = 0):
        super().__init__(verbose)
        self.max_scale = max_scale
        self.extra_vf_epochs = extra_vf_epochs
        self._policy_param_names = set()
        self._saved_state = None

    def _on_training_start(self) -> None:
        self._saved_state = {}
        for name, param in self.model.policy.named_parameters():
            is_value = 'vf' in name or 'value' in name
            if not is_value and 'log_std' not in name:
                self._policy_param_names.add(name)
                self._saved_state[name] = param.data.clone()
        print(f"  [PolicyScaling] Tracking {len(self._saved_state)} policy params, "
              f"scale={self.max_scale:.2f}")

        # Monkey-patch model.train() for extra VF training
        # Use the unbound class method directly to avoid capturing an already-patched instance method
        if self.extra_vf_epochs > 0 and not getattr(self.model, '_extra_vf_patched', False):
            _ppo_train = type(self.model).train  # unbound class method — never patched
            _model = self.model
            _callback = self
            def _train_with_extra_vf():
                _ppo_train(_model)
                _callback._extra_vf_training()
            self.model.train = _train_with_extra_vf
            self.model._extra_vf_patched = True
            print(f"  [PolicyScaling] Extra VF training: {self.extra_vf_epochs} epochs after each train()")

    def _extra_vf_training(self):
        """Run additional VF-only gradient steps. See VFWarmupCallback for details."""
        model = self.model
        buffer = model.rollout_buffer

        if not buffer.full:
            return

        grad_state = {}
        for name, param in model.policy.named_parameters():
            grad_state[name] = param.requires_grad
            is_value = 'vf' in name or 'value' in name
            param.requires_grad = is_value

        model.policy.set_training_mode(True)

        total_loss = 0.0
        for _epoch in range(self.extra_vf_epochs):
            for data in buffer.get(batch_size=None):
                values = model.policy.predict_values(data.observations)
                vf_loss = F.mse_loss(data.returns, values.flatten())

                model.policy.optimizer.zero_grad()
                (model.vf_coef * vf_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
                model.policy.optimizer.step()

                total_loss += vf_loss.item()

        for name, param in model.policy.named_parameters():
            param.requires_grad = grad_state[name]

        model.policy.set_training_mode(False)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if self._saved_state is None:
            return
        with torch.no_grad():
            for name, param in self.model.policy.named_parameters():
                if name in self._saved_state:
                    old = self._saved_state[name]
                    param.data.copy_(old + self.max_scale * (param.data - old))
                    self._saved_state[name] = param.data.clone()


class CommandStatsProtectorCallback(BaseCallback):
    """
    Periodically re-pin VecNormalize stats for command dims to identity.

    Commands are pre-normalized to [-1, 1] in the environment, so VecNormalize
    should pass them through unchanged (mean=0, var=1). Without this, the running
    stats drift during training and crush the command signal.
    """
    def __init__(self, body_dim: int = 1484, pin_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.body_dim = body_dim
        self.pin_freq = pin_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.pin_freq == 0:
            env = self.model.get_env()
            if hasattr(env, 'obs_rms'):
                env.obs_rms.mean[self.body_dim:] = 0.0
                env.obs_rms.var[self.body_dim:] = 1.0
                if self.verbose and self.num_timesteps % 50_000 == 0:
                    print(f"  [CommandProtector] Re-pinned command stats at step {self.num_timesteps:,}")
        return True


class LogStdClampCallback(BaseCallback):
    """
    Safety net: clamps policy log_std to prevent action distribution explosion.

    When entropy coefficient is too high or value function is unstable,
    log_std can grow exponentially, making all actions pure noise.

    Default bounds [-2, 1] give action std range [0.14, 2.7] - enough variance
    for exploration while preventing explosion. Clamp infrequently (every 2000
    steps) to let the policy explore between safety checks.
    """
    def __init__(self, log_std_min: float = -2.0, log_std_max: float = 1.0,
                 clamp_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.clamp_freq = clamp_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.clamp_freq == 0:
            if hasattr(self.model.policy, 'log_std'):
                with torch.no_grad():
                    log_std = self.model.policy.log_std
                    old_max = log_std.max().item()
                    log_std.clamp_(self.log_std_min, self.log_std_max)
                    new_max = log_std.max().item()
                    if self.verbose and old_max > self.log_std_max:
                        print(f"  [LogStdClamp] Clamped log_std: {old_max:.3f} -> {new_max:.3f} "
                              f"at step {self.num_timesteps:,}")
        return True


class EntropyScheduleCallback(BaseCallback):
    """
    Custom callback to schedule entropy coefficient during training.

    Tracks progress relative to training start, not absolute timesteps.
    This is critical for resume: if resuming at 10M with 7M remaining,
    progress should go 0→1 over those 7M steps, not start at 10M/7M=1.43.
    """
    def __init__(self, initial_ent: float, final_ent: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.total_timesteps = total_timesteps
        self._start_timesteps = None

    def _on_training_start(self) -> None:
        self._start_timesteps = self.model.num_timesteps

    def _on_step(self) -> bool:
        if self._start_timesteps is None:
            self._start_timesteps = self.num_timesteps

        elapsed = self.num_timesteps - self._start_timesteps
        progress = min(elapsed / max(self.total_timesteps, 1), 1.0)
        current_ent = self.initial_ent * (1.0 - progress) + self.final_ent * progress
        self.model.ent_coef = current_ent

        if self.verbose and self.num_timesteps % 50000 == 0:
            print(f"Entropy coefficient: {current_ent:.6f} (progress: {progress:.1%})")

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
    import platform
    import stable_baselines3
    print(f"Python {sys.version} on {platform.system()}")
    print(f"torch={torch.__version__} sb3={stable_baselines3.__version__} "
          f"cuda={torch.cuda.is_available()}")

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
    parser.add_argument('--standing-vecnorm', type=str, default='models/vecnorm.pkl',
                        help='Path to standing VecNormalize (for transfer learning)')
    parser.add_argument('--init-strategy', type=str, default='xavier',
                        choices=['zero', 'xavier', 'kaiming', 'small', 'velocity'],
                        help='Command feature weight initialization strategy')
    parser.add_argument('--warmup-steps', type=int, default=10000,
                        help='Warmup steps for VecNormalize before learning (0 to skip)')
    parser.add_argument('--no-warmup', action='store_true',
                        help='Skip warmup collection (faster but may degrade transfer)')
    parser.add_argument('--fresh-lr', action='store_true',
                        help='Reset LR schedule on resume (start from initial_lr instead of where model left off)')
    args = parser.parse_args()

    # Load config
    cfg = load_yaml('config/training_config.yaml')
    walking = cfg.get('walking', {}).copy()

    # Overrides / defaults
    n_envs = args.n_envs if args.n_envs is not None else int(walking.get('n_envs', 12))
    seed = int(walking.get('seed', 42))
    total_timesteps = int(walking.get('total_timesteps', 30_000_000)) if args.timesteps is None else args.timesteps
    
    learn_timesteps = total_timesteps
    reset_num_timesteps = True
    
    print(f"\n{'='*60}")
    print("WALKING TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Device: {walking.get('device', 'cuda')}")
    print(f"{'='*60}\n")

    # Ensure walking-specific settings
    walking.setdefault('curriculum_start_stage', 0)
    walking.setdefault('curriculum_max_stage', 1)
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
    vecnorm_explicitly_provided = args.vecnorm is not None

    env = None
    vecnorm_loaded = False

    # Skip VecNormalize loading if transferring from standing
    # The transfer_utils module will handle VecNormalize extension
    if args.from_standing and args.model:
        print("Skipping VecNormalize load - transfer_utils will handle it")
        # Keep env=None, the transfer function will set it up
    elif not args.reset_vecnorm:
        # Build list of candidate vecnorm paths to try
        candidates = []
        if args.vecnorm:
            candidates.append(args.vecnorm)
        if args.model:
            # Auto-detect: look next to the model file
            model_dir = os.path.dirname(args.model)
            candidates.append(os.path.join(model_dir, 'vecnorm.pkl'))
        candidates.append(vecnorm_path)  # Config default
        # Also try common ModelManager paths
        candidates.extend([
            'models/walking/latest/vecnorm.pkl',
            'models/walking/best/vecnorm.pkl',
            'models/walking/final/vecnorm.pkl',
        ])
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            normed = os.path.normpath(c)
            if normed not in seen:
                seen.add(normed)
                unique_candidates.append(c)

        for candidate_path in unique_candidates:
            if os.path.exists(candidate_path):
                try:
                    print(f"Attempting to load VecNormalize from: {candidate_path}")
                    env = VecNormalize.load(candidate_path, vec)
                    vecnorm_loaded = True
                    print("✓ Successfully loaded VecNormalize statistics")
                    print(f"  - Mean[:5]: {env.obs_rms.mean[:5]}")
                    print(f"  - Var[:5]: {env.obs_rms.var[:5]}")
                    print(f"  - ret_rms.var: {env.ret_rms.var:.4f}")
                    break
                except Exception as e:
                    print(f"✗ Failed to load VecNormalize from {candidate_path}: {e}")
                    env = None

        if env is None and vecnorm_explicitly_provided:
            print(f"\n{'!'*60}")
            print("ERROR: --vecnorm was explicitly provided but could not be loaded!")
            print(f"  Provided path: {args.vecnorm}")
            print(f"  Tried: {unique_candidates}")
            print("  Resuming without VecNormalize stats will destroy performance.")
            print(f"{'!'*60}\n")
            raise FileNotFoundError(
                f"VecNormalize file not found or failed to load. "
                f"Tried: {unique_candidates}"
            )

    if env is None and not (args.from_standing and args.model):
        print("Creating new VecNormalize wrapper")
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
    initial_lr = float(walking.get('learning_rate', 5e-6))
    final_lr = float(walking.get('final_learning_rate', 3e-6))
    lr_fn = lr_schedule(initial_lr, final_lr, total_timesteps)

    initial_clip = float(walking.get('clip_range', 0.2))
    final_clip = float(walking.get('final_clip_range', 0.2))
    clip_fn = clip_schedule(initial_clip, final_clip, total_timesteps)

    # Entropy coefficient
    initial_ent = float(walking.get('ent_coef', 0.005))
    final_ent = float(walking.get('final_ent_coef', 0.003))
    if final_ent <= 0:
        print(f"  WARNING: final_ent_coef={final_ent} is non-positive, forcing to 0.005")
        final_ent = 0.005

    # NOTE: Previously capped entropy to 0.01 during transfer, but this killed exploration
    # and prevented the agent from discovering walking. The log_std explosion was caused
    # by other issues (VecNormalize variance collapse, reward instability) which are now fixed.
    # Keep entropy at configured value (0.05) to enable exploration.
    if args.from_standing and args.model:
        print(f"  Transfer mode: using configured ent_coef={initial_ent:.4f} (exploration enabled)")

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

    if args.from_standing and args.model:
        # ========== IMPROVED TRANSFER FROM STANDING MODEL ==========
        # Uses new transfer_utils module that properly handles:
        # 1. VecNormalize dimension mismatch (1484 → 1496)
        # 2. Command feature weight initialization (xavier instead of 0.01 noise)
        # 3. Optional warmup for normalization statistics
        
        try:
            # Determine warmup steps
            warmup_steps = 0 if args.no_warmup else args.warmup_steps
            
            # Standing VecNormalize path
            standing_vecnorm_path = args.standing_vecnorm
            if not os.path.exists(standing_vecnorm_path):
                # Try common alternatives
                alt_paths = [
                    'models/vecnorm.pkl',
                    'models/standing/latest/vecnorm.pkl',
                    'models/standing/best/vecnorm.pkl',
                ]
                for alt in alt_paths:
                    if os.path.exists(alt):
                        standing_vecnorm_path = alt
                        break
            
            # Prepare model kwargs
            model_kwargs = {
                'policy': 'MlpPolicy',
                'learning_rate': lr_fn,
                'n_steps': int(walking.get('n_steps', 2048)),
                'batch_size': int(walking.get('batch_size', 2048)),
                'n_epochs': int(walking.get('n_epochs', 6)),
                'gamma': float(walking.get('gamma', 0.995)),
                'gae_lambda': float(walking.get('gae_lambda', 0.95)),
                'clip_range': clip_fn,
                'ent_coef': initial_ent,
                'vf_coef': float(walking.get('vf_coef', 0.5)),
                'max_grad_norm': float(walking.get('max_grad_norm', 0.5)),
                'target_kl': float(walking['target_kl']) if 'target_kl' in walking else None,
                'policy_kwargs': policy_kwargs,
                'seed': seed,
                'verbose': int(walking.get('verbose', 1)),
            }
            
            # Use the new comprehensive transfer function
            model, env = transfer_standing_to_walking(
                standing_model_path=args.model,
                standing_vecnorm_path=standing_vecnorm_path,
                walking_env=vec,  # Raw VecEnv before VecNormalize
                walking_model_kwargs=model_kwargs,
                device=device,
                init_strategy=args.init_strategy,
                warmup_steps=warmup_steps,
            )
            
            vecnorm_loaded = True  # We handled it in transfer
            learn_timesteps = total_timesteps
            reset_num_timesteps = True
            resume = True
            
            print(f"✓ Transfer complete! Init strategy: {args.init_strategy}")
            if warmup_steps > 0:
                print(f"  Warmup: {warmup_steps:,} steps collected")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"✗ Transfer from standing failed: {e}")
            import traceback
            traceback.print_exc()
            print("\n  Falling back to fresh model...")
            resume = False
            
            # Create fresh VecNormalize if transfer failed
            if env is None:
                env = VecNormalize(
                    vec,
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0,
                    clip_reward=10.0,
                    gamma=walking.get('gamma', 0.995),
                )
    
    elif resume:
        # Resume from walking model
        try:
            print(f"Loading walking model from: {args.model}")

            # Load model first to read num_timesteps, then build schedules.
            # Pass dummy schedule via custom_objects to avoid segfault from
            # cloudpickle-deserialized np.clip (numpy version mismatch).
            dummy_lr_fn = lr_schedule(initial_lr, final_lr, total_timesteps)
            dummy_clip_fn = clip_schedule(initial_clip, final_clip, total_timesteps)

            # Override ALL PPO hyperparameters from config, not just LR/clip.
            # PPO.load() restores saved hyperparameters — if resuming from a
            # checkpoint saved with old values (e.g. batch_size=512, n_epochs=10),
            # those would silently produce 480 gradient steps instead of 24,
            # causing KL explosion.
            config_batch_size = int(walking.get('batch_size', 2048))
            config_n_epochs = int(walking.get('n_epochs', 6))

            model = PPO.load(
                args.model, env=env, device=device,
                custom_objects={
                    'learning_rate': dummy_lr_fn,
                    'lr_schedule': dummy_lr_fn,
                    'clip_range': dummy_clip_fn,
                    'batch_size': config_batch_size,
                    'n_epochs': config_n_epochs,
                    'verbose': int(walking.get('verbose', 1)),
                },
            )

            loaded_timesteps = model.num_timesteps
            remaining_timesteps = total_timesteps - loaded_timesteps

            if remaining_timesteps <= 0:
                print(f"Model already trained for {loaded_timesteps:,} steps (target: {total_timesteps:,})")
                print(f"  Use --timesteps {loaded_timesteps + 5_000_000} to train more")
                return

            # Compute where LR/clip SHOULD be at this point in the global schedule,
            # then decay from there to final over the remaining steps.
            if getattr(args, 'fresh_lr', False):
                # --fresh-lr: reset schedule to initial values (use when resuming
                # a model whose num_timesteps exhausted the original schedule)
                resume_lr = initial_lr
                resume_clip = initial_clip
                print(f"  --fresh-lr: LR schedule reset to {initial_lr} → {final_lr} over {remaining_timesteps:,} steps")
            else:
                progress_done = loaded_timesteps / total_timesteps
                resume_lr = initial_lr * (1.0 - progress_done) + final_lr * progress_done
                resume_clip = initial_clip * (1.0 - progress_done) + final_clip * progress_done

            # Build schedules from resume point → final over remaining steps.
            # reset_num_timesteps=True so SB3's progress_remaining goes 1→0
            # over exactly learn_timesteps.
            remaining_lr_fn = lr_schedule(resume_lr, final_lr, remaining_timesteps)
            remaining_clip_fn = clip_schedule(resume_clip, final_clip, remaining_timesteps)
            model.learning_rate = remaining_lr_fn
            model.lr_schedule = remaining_lr_fn
            model.clip_range = remaining_clip_fn

            learn_timesteps = remaining_timesteps
            reset_num_timesteps = True

            # Verify PPO hyperparameters match config (catch stale saved values)
            if model.batch_size != config_batch_size:
                print(f"  FIX: batch_size {model.batch_size} → {config_batch_size} (from config)")
                model.batch_size = config_batch_size
            if model.n_epochs != config_n_epochs:
                print(f"  FIX: n_epochs {model.n_epochs} → {config_n_epochs} (from config)")
                model.n_epochs = config_n_epochs

            print(f"✓ Loaded model at {loaded_timesteps:,} timesteps")
            print(f"  Training {remaining_timesteps:,} more steps (target: {total_timesteps:,})")
            print(f"  LR schedule: {resume_lr:.6f} → {final_lr} over remaining steps")
            print(f"  Clip schedule: {resume_clip:.4f} → {final_clip} over remaining steps")
            print(f"  batch_size: {model.batch_size}, n_epochs: {model.n_epochs}")

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("  Starting fresh training instead...")
            resume = False
    
    if not resume:
        # Create fresh model
        print("\n" + "!"*60)
        print("WARNING: Training from scratch WITHOUT standing pretrain!")
        print("!"*60 + "\n")
        
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=lr_fn,
            n_steps=int(walking.get('n_steps', 2048)),
            batch_size=int(walking.get('batch_size', 2048)),
            n_epochs=int(walking.get('n_epochs', 6)),
            gamma=float(walking.get('gamma', 0.995)),
            gae_lambda=float(walking.get('gae_lambda', 0.95)),
            clip_range=clip_fn,
            ent_coef=initial_ent,
            vf_coef=float(walking.get('vf_coef', 0.5)),
            max_grad_norm=float(walking.get('max_grad_norm', 0.5)),
            target_kl=float(walking['target_kl']) if 'target_kl' in walking else None,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=int(walking.get('verbose', 1)),
            device=device,
        )

    callback_list = [
        # FIX: Pin command stats EVERY step to prevent variance drift
        CommandStatsProtectorCallback(body_dim=1484, pin_freq=1, verbose=0),
        # log_std_max=0.0 gives std=1.0: enough exploration without making KL explode
        # 0.5 (std=1.65) caused KL>100 even at LR=5e-5, preventing any learning
        LogStdClampCallback(log_std_min=-2.0, log_std_max=0.0, clamp_freq=500, verbose=1),
        EntropyScheduleCallback(initial_ent, final_ent, learn_timesteps, verbose=1),
        WalkingMetricsCallback(log_freq=int(walking.get('wandb_log_freq', 10000)), verbose=1),
        SaveWithModelManagerCallback(
            model_manager=model_manager,
            freq=int(walking.get('save_freq', 250_000))
        )
    ]

    # Policy scaling: with 17 action dims, full-speed PPO updates overshoot the
    # clip boundary at ANY learning rate. Permanent scaling keeps applied updates
    # conservative while letting Adam compute correct gradients.
    max_scale = float(walking.get('policy_max_scale', 1.0))
    extra_vf_epochs = int(walking.get('extra_vf_epochs', 7))

    if args.from_standing and args.model:
        # Transfer from standing: VF warmup + ramp + permanent scaling
        warmup_steps = int(walking.get('vf_warmup_steps', 250_000))
        rampup_steps = int(walking.get('vf_rampup_steps', 500_000))
        callback_list.insert(0, ValueFunctionWarmupCallback(
            warmup_steps=warmup_steps, rampup_steps=rampup_steps,
            max_scale=max_scale, extra_vf_epochs=extra_vf_epochs, verbose=1
        ))
    elif resume and max_scale < 1.0:
        # Resume: standalone permanent scaling (no warmup/ramp needed)
        # Use separate resume_policy_scale if available (walking models need less damping)
        resume_scale = float(walking.get('resume_policy_scale', max_scale))
        callback_list.insert(0, PermanentPolicyScalingCallback(
            max_scale=resume_scale, extra_vf_epochs=extra_vf_epochs, verbose=1
        ))
    
    
    # Add WandB callbacks if enabled
    if use_wandb:
        callback_list.extend([
            PPOMetricsWandBCallback(verbose=1),
            VelocityTrackingWandBCallback(
                log_freq=int(walking.get('wandb_log_freq', 5000)),
                project_name=walking.get('wandb_project', 'humanoid_walking'),
                config=walking
            ),
            CurriculumWandBCallback(log_freq=5000),
            RewardBreakdownWandBCallback(
                log_freq=int(walking.get('wandb_log_freq', 5000)),
                buffer_size=1000
            )
        ])
    
    callbacks = CallbackList(callback_list)

    # ========== PRE-TRAINING VERIFICATION (FIX: diagnose issues early) ==========
    print(f"\n{'='*60}")
    print("PRE-TRAINING VERIFICATION")
    print(f"{'='*60}")

    # Check policy log_std
    if hasattr(model.policy, 'log_std'):
        log_std_val = model.policy.log_std.mean().item()
        std_val = np.exp(log_std_val)
        print(f"Policy log_std: {log_std_val:.3f} (std = {std_val:.3f})")
        if log_std_val > 0:
            print("  WARNING: log_std > 0, will be clamped by callback")

    # Check VecNormalize stats
    print("\nVecNormalize observation stats:")
    print(f"  obs_rms.var min: {env.obs_rms.var.min():.4f}")
    print(f"  obs_rms.var max: {env.obs_rms.var.max():.4f}")
    print(f"  ret_rms.var: {env.ret_rms.var:.4f}")

    # Check command dims specifically (last 9 dims)
    cmd_start = 1484
    print(f"\nCommand block stats (dims {cmd_start}:{cmd_start+9}):")
    print(f"  mean: {env.obs_rms.mean[cmd_start:]}")
    print(f"  var: {env.obs_rms.var[cmd_start:]}")
    if env.obs_rms.var[cmd_start:].min() < 0.5:
        print("  WARNING: Command variance < 0.5, may need re-pinning")

    # PPO config verification
    print("\nPPO Configuration:")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Entropy coef: {model.ent_coef}")
    print(f"  Batch size: {model.batch_size}")
    print(f"  Target KL: {model.target_kl}")

    print(f"{'='*60}")

    # Train
    print(f"\n{'='*60}")
    print("Starting WALKING training:")
    print(f"  Mode: {'RESUME' if resume else 'FRESH START'}")
    print(f"  Training steps: {learn_timesteps:,}")
    print(f"  Target total: {total_timesteps:,}")
    print(f"  Environments: {n_envs}")
    print(f"  Device: {device}")
    print(f"  VecNormalize: {'LOADED' if vecnorm_loaded else 'NEW'}")
    speed_stages = walking.get('curriculum_max_speed_stages', [0.15, 0.4, 0.8])
    print(f"  Curriculum stages: 0-{walking.get('curriculum_max_stage', 2)}")
    print(f"  Max speed stages: {' → '.join(f'{s} m/s' for s in speed_stages)}")
    print(f"  Reward tracking weight: {walking.get('reward_tracking_weight', 2.5)}")
    print(f"  Action smoothing tau: {walking.get('action_smoothing_tau', 0.2)}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=learn_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps
    )

    # Fix: reset_num_timesteps=True makes num_timesteps equal to learn_timesteps
    # (the remaining steps), not the true total. Restore before saving.
    model.num_timesteps = total_timesteps

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
    print("\nModel locations:")
    print(f"  Final:  {model_manager.final_dir / 'model.zip'}")
    print(f"  Best:   {model_manager.best_dir / 'model.zip'} (vel_error: {best_info['metric']:.4f})")
    print(f"  Latest: {model_manager.latest_dir / 'model.zip'}")
    print("\nTo record demo videos, run:")
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

