"""
Standing agent for humanoid using PPO
Parallel VecEnvs + VecNormalize, Colab-friendly with WandB logging
Optimized for stability and balance tasks
FIXED: Gym API compatibility issues

standing_agent.py
"""

import os
from typing import Callable, Optional

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available. Install with: pip install wandb")


# Import the environment (adjust path for Colab)
try:
    from src.environments.standing_env import make_standing_env
except ImportError:
    import sys
    sys.path.append('/content')
    from src.environments.standing_env import make_standing_env


def safe_step(env, action):
    """Handle both old and new gym API step returns"""
    result = env.step(action)
    
    if len(result) == 4:
        # Old gym API: obs, reward, done, info
        obs, reward, done, info = result
        # Convert to new API format: obs, reward, terminated, truncated, info
        return obs, reward, done, False, info
    elif len(result) == 5:
        # New gymnasium API: obs, reward, terminated, truncated, info
        return result
    else:
        raise ValueError(f"Unexpected step() return format: {len(result)} values")
        
def record_evaluation_video(model, env, n_episodes=1, max_steps_per_episode=1000):
    """Record longer video for stability assessment"""
    frame = render_env.render()
    if frame is None:
        print("Render returned None - check render_mode")
        return []  # Early exit
    frames = []

    for episode in range(n_episodes):
        obs = env.reset()
        episode_frames = []
        
        for step in range(max_steps_per_episode):
            try:
                if hasattr(env, 'venv'):
                    render_env = env.venv.envs[0]
                elif hasattr(env, 'envs'):
                    render_env = env.envs[0]
                else:
                    render_env = env
                
                frame = render_env.render()
                if frame is not None and hasattr(frame, 'shape') and len(frame.shape) == 3:
                    # Record every 2nd frame for speed
                    if step % 2 == 0:
                        episode_frames.append(frame)
                    
            except Exception as e:
                if step == 0:
                    print(f"Rendering failed: {e}")
                break
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = safe_step(env, action)
            
            done = bool(terminated[0] if hasattr(terminated, '__len__') else terminated) or \
                bool(truncated[0] if hasattr(truncated, '__len__') else truncated)
            
            if done:
                break
        
        frames.extend(episode_frames[:200])  # Longer videos to show stability
        
        if len(frames) >= 300:
            break
    
    return frames


class StandingCallback(BaseCallback):
    """Unified callback for standing task: evaluation + checkpointing + WandB logging."""

    def __init__(
            self,
            config,
            eval_env_fn: Optional[Callable[[], DummyVecEnv]] = None,
            wandb_run=None,
        ):
            super().__init__(config.get('verbose', 1))

            self.config = config
            
            # Get all values from config
            self.eval_env_fn = eval_env_fn
            self.n_eval_episodes = config.get('n_eval_episodes')
            self.eval_freq = config.get('eval_freq')
            self.save_freq = config.get('save_freq')
            self.log_freq = config.get('wandb_log_freq')
            self.video_freq = config.get('video_freq')
            self.best_model_path = config.get('best_model_path')
            self.checkpoint_dir = config.get('checkpoint_dir')
            self.checkpoint_prefix = config.get('checkpoint_prefix')
            self.wandb_run = wandb_run
            self.target_height = config.get('target_height')
            self.success_threshold = config.get('target_reward_threshold')
            self.episode_print_freq = config.get('episode_print_freq',1000)
            self.vecnormalize_path = config.get('vecnormalize_path')
            
            # Initialize tracking variables
            self.best_mean_reward = -np.inf
            self.best_height_error = np.inf
            self.best_height_stability = np.inf
            self.height_data = []
            
            # Episode tracking (remove duplicates)
            self.episode_rewards = []
            self.episode_lengths = []
            self._current_episode_heights = []
            

            os.makedirs("models/saved_models", exist_ok=True)

            for path in [self.best_model_path, self.checkpoint_dir]:
                if path is not None:  # ✅ Check for None first!
                    parent_dir = os.path.dirname(path)
                    if parent_dir:
                        os.makedirs(parent_dir, exist_ok=True)

    def _collect_episode_data(self):
        infos = self.locals.get("infos", [])
        
        for info in infos:
            # Collect height data during episode (step-level)
            if "height" in info:
                if not hasattr(self, 'height_data'):
                    self.height_data = []
                self.height_data.append(info["height"])
                
                # Also track per-episode for detailed logging
                if not hasattr(self, '_current_episode_heights'):
                    self._current_episode_heights = []
                self._current_episode_heights.append(info["height"])
            
            # Track XY position for drift calculation
            if "x_position" in info and "y_position" in info:
                if not hasattr(self, '_current_episode_xy'):
                    self._current_episode_xy = []
                self._current_episode_xy.append([info["x_position"], info["y_position"]])
            
            # When episode ends, store summary data
            if "episode" in info:
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Print progress periodically
                if self.verbose and len(self.episode_rewards) % self.episode_print_freq == 0:
                    if self._current_episode_heights:
                        ep_height_mean = np.mean(self._current_episode_heights)
                        ep_height_std = np.std(self._current_episode_heights)
                        print(f"Episode {len(self.episode_rewards)}: "
                            f"len={episode_length}, height={ep_height_mean:.3f}±{ep_height_std:.3f}, "
                            f"reward={episode_reward:.1f}")
                
                # Reset per-episode tracking (but keep global height_data!)
                self._current_episode_heights = []
                self._current_episode_xy = []
            
    def _save_checkpoint(self):
        """Save model checkpoint AND VecNormalize"""
        path = os.path.join(self.checkpoint_dir, f"{self.checkpoint_prefix}_{self.num_timesteps}.zip")
        self.model.save(path)
        
        # ALSO save VecNormalize at checkpoints
        if isinstance(self.model.get_env(), VecNormalize):
            vecnorm_checkpoint = path.replace('.zip', '_vecnorm.pkl')
            self.model.get_env().save(vecnorm_checkpoint)
            if self.verbose:
                print(f"Checkpoint + VecNormalize saved: {path}")
        elif self.verbose:
            print(f"Checkpoint saved: {path}")
        
        # Save to WandB
        if WANDB_AVAILABLE and wandb.run:
            wandb.save(path)
            if isinstance(self.model.get_env(), VecNormalize):
                wandb.save(vecnorm_checkpoint)

    def _on_step(self) -> bool:
        t = self.num_timesteps

        # Collect episode data for episode-based logging
        self._collect_episode_data()

        # WandB training logging (every log_freq steps)
        if self.log_freq > 0 and (t % self.log_freq == 0) and WANDB_AVAILABLE and wandb.run:
            self._log_training_metrics()

        # Evaluation (every eval_freq steps)
        if self.eval_env_fn is not None and self.eval_freq > 0 and (t % self.eval_freq == 0):
            self._run_evaluation()

        # Video recording (every video_freq steps)
        if (self.video_freq > 0 and (t % self.video_freq == 0) and 
            WANDB_AVAILABLE and wandb.run and self.eval_env_fn is not None):
            self._record_and_log_video()

        # Checkpointing (every save_freq steps)
        if self.save_freq > 0 and (t % self.save_freq == 0):
            self._save_checkpoint()

        return True

    def _log_training_metrics(self):
        """Log training metrics every log_freq steps using recent episode data"""
        if not (WANDB_AVAILABLE and wandb.run):
            return
        
        # Calculate metrics from recent completed episodes
        if not self.episode_rewards or not self.episode_lengths:
            return  # No episodes completed yet
        
        # Use last N completed episodes for current metrics (e.g., last 10 episodes)
        n_recent = min(10, len(self.episode_rewards))
        recent_lengths = self.episode_lengths[-n_recent:]
        
        # Get height data from recent episodes
        if hasattr(self, 'height_data') and self.height_data:
            # Use last ~1000 height samples (approximate from recent episodes)
            recent_heights = self.height_data[-1000:]
            
            mean_height = np.mean(recent_heights)
            height_stability = np.std(recent_heights)
            height_error = abs(mean_height - self.target_height)
            
            # Calculate max consecutive in range
            in_range = np.abs(np.array(recent_heights) - self.target_height) < 0.05
            max_consec = 0
            current_consec = 0
            for in_target in in_range:
                if in_target:
                    current_consec += 1
                    max_consec = max(max_consec, current_consec)
                else:
                    current_consec = 0
            
            # NEW: Calculate XY drift from tracked positions
            xy_drift = 0.0
            if hasattr(self, '_current_episode_xy') and self._current_episode_xy:
                xy_array = np.array(self._current_episode_xy[-1000:])  # Last 1000 positions
                distances = np.sqrt(xy_array[:, 0]**2 + xy_array[:, 1]**2)
                xy_drift = float(np.mean(distances))
            
            # Get action data
            actions = self.locals.get("actions", np.array([[0]]))
            action_mag_mean = float(np.abs(actions).mean())
            action_mag_max = float(np.abs(actions).max())
            
            # Log the 8 core metrics
            wandb.log({
                "global_step": self.num_timesteps,
                "train/episode_length": float(np.mean(recent_lengths)),
                "train/mean_height": float(mean_height),
                "train/height_stability": float(height_stability),
                "train/max_consecutive_in_range": float(max_consec),
                "train/xy_drift": xy_drift,  # FIXED: Now calculated
                "train/action_magnitude_mean": action_mag_mean,  # FIXED: Already calculated
                "train/action_magnitude_max": action_mag_max,  # FIXED: Already calculated
                "train/height_error": float(height_error),
            }, step=self.num_timesteps)

    def _record_and_log_video(self):
        if self.verbose:
            print(f"Recording video at step {self.num_timesteps:,}")
        
        try:
            # FIXED: Create environment with proper render mode for headless
            render_env = make_standing_env(render_mode="rgb_array", config=self.config)
            
            frames = []
            obs, _ = render_env.reset()
            
            # Record longer episodes for better video
            for step in range(400):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = render_env.step(action)
                
                # FIXED: Get frame with error handling
                try:
                    frame = render_env.render()
                    if frame is not None and hasattr(frame, 'shape'):
                        # Convert to uint8 if needed
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                        frames.append(frame)
                except Exception as e:
                    if step == 0:  # Only print on first frame failure
                        print(f"Frame capture failed at step {step}: {e}")
                    break
                
                if terminated or truncated:
                    break
            
            render_env.close()
            
            # FIXED: Better video logging with validation
            if len(frames) > 10 and WANDB_AVAILABLE and wandb.run:  # Need at least 10 frames
                try:
                    # Performance metrics from last info
                    height = info.get('height', 0) if isinstance(info, dict) else 0
                    height_error = abs(height - self.target_height)
                    
                    caption = f"Step {self.num_timesteps//1000}k | Height:{height:.2f} | Error:{height_error:.3f}"
                    
                    # Convert frames to proper format
                    video_array = np.array(frames)
                    
                    # FIXED: Ensure proper video format
                    if len(video_array.shape) == 4 and video_array.shape[-1] == 3:  # (frames, height, width, channels)
                        wandb.log({
                            "video/training_progress": wandb.Video(
                                video_array, 
                                fps=30,  # Increased FPS for smoother video
                                format="mp4",
                                caption=caption
                            )
                        }, step=self.num_timesteps)
                        
                        print(f"Video logged successfully: {len(frames)} frames, shape {video_array.shape}")
                    else:
                        print(f"Invalid video array shape: {video_array.shape}")
                        
                except Exception as e:
                    print(f"Video upload failed: {e}")
            else:
                print(f"Insufficient frames for video: {len(frames)} frames captured")
            
        except Exception as e:
            print(f"Video recording failed: {e}")

    def _run_evaluation(self):
        # SAVE VecNormalize BEFORE evaluation
        if isinstance(self.model.get_env(), VecNormalize):
            try:
                from pathlib import Path
                vecnorm_path = Path(self.vecnormalize_path)
                vecnorm_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.get_env().save(str(vecnorm_path))
                print(f"✓ VecNormalize stats saved to {vecnorm_path}")
            except Exception as e:
                print(f"✗ WARNING: VecNormalize save failed: {e}")
        
        eval_env = self.eval_env_fn()
        
        heights = []
        rewards = []
        episode_lengths = []
        xy_positions = []  # NEW: Track XY drift
        actions_taken = []  # NEW: Track actions
        
        for ep in range(self.n_eval_episodes):
            obs = eval_env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            ep_heights = []
            ep_xy_positions = []  # NEW
            ep_actions = []  # NEW
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = safe_step(eval_env, action)
                done = bool(terminated[0] if hasattr(terminated, '__len__') else terminated) or \
                    bool(truncated[0] if hasattr(truncated, '__len__') else truncated)
                
                ep_reward += reward[0] if hasattr(reward, '__len__') else reward
                ep_length += 1
                
                # Extract height
                if hasattr(info, '__len__') and len(info) > 0 and 'height' in info[0]:
                    ep_heights.append(info[0]['height'])
                    # NEW: Extract XY position
                    if 'x_position' in info[0] and 'y_position' in info[0]:
                        ep_xy_positions.append([info[0]['x_position'], info[0]['y_position']])
                elif hasattr(obs, '__len__') and len(obs) > 0:
                    ep_heights.append(obs[0][0] if hasattr(obs[0], '__len__') else obs[0])
                
                # NEW: Track actions
                if hasattr(action, '__len__'):
                    ep_actions.append(action[0] if len(action.shape) > 1 else action)
                else:
                    ep_actions.append(action)
            
            rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            if ep_heights:
                heights.extend(ep_heights)
            if ep_xy_positions:
                xy_positions.extend(ep_xy_positions)
            if ep_actions:
                actions_taken.extend(ep_actions)
        
        # Calculate metrics
        mean_rew = np.mean(rewards)
        std_rew = np.std(rewards)

        if mean_rew > self.best_mean_reward:
            self.best_mean_reward = mean_rew
            
            # FIXED: Proper file saving
            from pathlib import Path
            
            # Get the path WITHOUT .zip
            model_path_str = self.best_model_path
            # Remove .zip if someone added it in config
            if model_path_str.endswith('.zip'):
                model_path_str = model_path_str[:-4]
            
            model_path = Path(model_path_str)
            
            # Create parent directory
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # CRITICAL: Save without .zip, SB3 adds it
            self.model.save(str(model_path))
            
            # Verify the file was created correctly
            zip_file = Path(str(model_path) + ".zip")
            if zip_file.is_file():
                print(f"✓ Best model saved: {zip_file}")
            else:
                print(f"✗ WARNING: Expected file not created: {zip_file}")
            
            # Save VecNormalize
            if isinstance(self.model.get_env(), VecNormalize):
                vecnorm_path = Path(self.vecnormalize_path)
                vecnorm_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.get_env().save(str(vecnorm_path))
                print(f"✓ VecNormalize saved: {vecnorm_path}")
            
            print(f"  New best reward: {mean_rew:.2f}")
        
        # Log evaluation metrics to WandB
        if WANDB_AVAILABLE and wandb.run and heights:
            mean_height = float(np.mean(heights))
            height_stability = float(np.std(heights))
            height_error = float(abs(mean_height - self.target_height))
            mean_length = float(np.mean(episode_lengths))
            
            # NEW: Calculate XY drift (distance from origin)
            if xy_positions:
                xy_array = np.array(xy_positions)
                distances_from_origin = np.sqrt(xy_array[:, 0]**2 + xy_array[:, 1]**2)
                xy_drift = float(np.mean(distances_from_origin))
            else:
                xy_drift = 0.0
            
            # NEW: Calculate action magnitudes
            if actions_taken:
                actions_array = np.array(actions_taken)
                action_mag_mean = float(np.mean(np.abs(actions_array)))
                action_mag_max = float(np.max(np.abs(actions_array)))
            else:
                action_mag_mean = 0.0
                action_mag_max = 0.0
            
            # Calculate max consecutive in range
            in_range = np.abs(np.array(heights) - self.target_height) < 0.05
            max_consec = 0
            curr_consec = 0
            for in_target in in_range:
                if in_target:
                    curr_consec += 1
                    max_consec = max(max_consec, curr_consec)
                else:
                    curr_consec = 0
            
            wandb.log({
                "global_step": self.num_timesteps,
                "eval/episode_length": mean_length,
                "eval/mean_height": mean_height,
                "eval/height_stability": height_stability,
                "eval/max_consecutive_in_range": float(max_consec),
                "eval/xy_drift": xy_drift,  # FIXED: Now calculated
                "eval/action_magnitude_mean": action_mag_mean,  # FIXED: Now calculated
                "eval/action_magnitude_max": action_mag_max,  # FIXED: Now calculated
                "eval/height_error": height_error,
                "eval/mean_reward": float(mean_rew),
            }, step=self.num_timesteps)
            
            # Update best values in summary
            if height_error < wandb.run.summary.get("best_height_error", float('inf')):
                wandb.run.summary["best_height_error"] = height_error
            if height_stability < wandb.run.summary.get("best_height_stability", float('inf')):
                wandb.run.summary["best_height_stability"] = height_stability
            if mean_length > wandb.run.summary.get("best_episode_length", 0):
                wandb.run.summary["best_episode_length"] = mean_length
            if max_consec > wandb.run.summary.get("best_max_consecutive", 0):
                wandb.run.summary["best_max_consecutive"] = max_consec

        eval_env.close()


class StandingAgent:
    """PPO agent for humanoid standing with parallel envs, normalization, and WandB logging."""

    def __init__(self, config):
        self.config = config
        self.model: Optional[PPO] = None
        self.env = None
        self.vecnormalize_path = self.config.get("vecnormalize_path")
        self.wandb_run = config.get('wandb_run', None)
        
        # Standing-specific parameters
        self.target_height = config.get('target_height')
        self.success_threshold = config.get('target_reward_threshold')
        self.n_envs = config.get('n_envs')


    # ---------- Env construction ----------

    def _make_single_env(self, seed: int, rank: int, render_mode="rgb_array") -> Callable:
        """Factory that returns a thunk creating one monitored env with seeding."""
        def _init():
            os.environ.setdefault("MUJOCO_GL", "egl")
            env = make_standing_env(render_mode=render_mode, config=self.config)
            log_dir = self.config.get("log_dir", "data/logs")
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, filename=os.path.join(log_dir, f"train_env_{rank}.csv"))
            # Seeding
            if hasattr(env, "reset"):
                env.reset(seed=seed + rank)
            try:
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
            except Exception:
                pass
            return env
        return _init

    def create_environment(self, render_mode=None):
        """Create vectorized training env, wrap with VecNormalize if requested."""
        n_envs = int(self.config.get("n_envs", 8))
        seed = int(self.config.get("seed", 42))

        if n_envs > 1:
            env_fns = [self._make_single_env(seed, i, render_mode=None) for i in range(n_envs)]
            vec = SubprocVecEnv(env_fns)
        else:
            vec = DummyVecEnv([self._make_single_env(seed, 0, render_mode=None)])

        if self.config.get("normalize", True):
            self.env = VecNormalize(
                vec,
                norm_obs=True,
                norm_reward=False,  # CRITICAL FIX: Don't normalize rewards
                clip_obs=10.0,
                gamma=self.config.get("gamma"),
            )
        else:
            self.env = vec
        return self.env

    # ---------- Model construction ----------

    def create_model(self):
        """Create PPO model optimized for standing task."""
        if self.env is None:
            self.create_environment()

            
        # Standing-optimized parameters
        model_params = {
            "learning_rate": self.config.get("learning_rate"),
            "n_steps": self.config.get("n_steps"),
            "batch_size": self.config.get("batch_size"),
            "n_epochs": self.config.get("n_epochs"),
            "gamma": self.config.get("gamma"),
            "gae_lambda": self.config.get("gae_lambda"),
            "clip_range": self.config.get("clip_range"),
            "ent_coef": self.config.get("ent_coef"),
            "vf_coef": self.config.get("vf_coef"),
            "max_grad_norm": self.config.get("max_grad_norm"),
            "verbose": self.config.get("verbose"),
            "seed": self.config.get("seed"),
            "device": self.config.get("device"),
        }

        # Standing-optimized architecture (deeper for fine control)
        policy_kwargs = self.config.get("policy_kwargs")

        # Optional: map string activation names to torch.nn modules
        import torch.nn as nn
        activation_map = {
            "relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoid", "elu": "ELU",
            "gelu": "GELU", "leaky_relu": "LeakyReLU", "leakyrelu": "LeakyReLU",
            "silu": "SiLU", "mish": "Mish", "hardswish": "Hardswish",
        }
        if "activation_fn" in policy_kwargs and isinstance(policy_kwargs["activation_fn"], str):
            act = policy_kwargs["activation_fn"].lower().replace("_", "")
            policy_kwargs["activation_fn"] = getattr(nn, activation_map.get(act, policy_kwargs["activation_fn"]))

        tensorboard_log = self.config.get("log_dir", "data/logs") if self.config.get("use_tensorboard", True) else None

        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            **model_params,
        )
        
        # Log model architecture to WandB
        if WANDB_AVAILABLE and wandb.run:
            wandb.config.update({
                "model/policy_kwargs": str(policy_kwargs),
                "model/n_params": sum(p.numel() for p in self.model.policy.parameters()),
                "task/target_height": self.target_height,
                "task/success_threshold": self.success_threshold,
            })
        
        return self.model

    # ---------- Training / Evaluation ----------

    def _build_eval_env(self):
        """Create a single-env VecEnv for evaluation, restoring normalization stats if available."""
        eval_vec = DummyVecEnv([self._make_single_env(self.config.get("seed", 42), rank=10_000)])
        if self.config.get("normalize", True) and os.path.exists(self.vecnormalize_path):
            eval_vec = VecNormalize.load(self.vecnormalize_path, eval_vec)
            eval_vec.training = False
            eval_vec.norm_reward = False
        return eval_vec

    def train(self, total_timesteps=None, callback=None):
        if self.model is None:
            self.create_model()
        if total_timesteps is None:
            total_timesteps = int(self.config.get("total_timesteps"))

        def eval_env_fn():
            # Create base eval environment
            base_eval_env = DummyVecEnv([lambda: make_standing_env(render_mode="rgb_array", config=self.config)])
            
            # Wrap with VecNormalize to match training env structure
            if isinstance(self.env, VecNormalize):
                # Load existing normalization stats if available
                if os.path.exists(self.vecnormalize_path):
                    eval_env = VecNormalize.load(self.vecnormalize_path, base_eval_env)
                else:
                    # Create new VecNormalize with same settings as training
                    eval_env = VecNormalize(
                        base_eval_env,
                        norm_obs=True,
                        norm_reward=False,
                        clip_obs=10.0,
                        gamma=self.config.get("gamma", 0.995),
                    )
                
                # Set to evaluation mode
                eval_env.training = False
                eval_env.norm_reward = False
                
                return eval_env
            else:
                return base_eval_env
        
        callbacks = [
            StandingCallback(
                config=self.config,
                eval_env_fn=eval_env_fn,
                wandb_run=self.wandb_run,
            ),
            EarlyStoppingCallback(
                check_freq=50000,
                target_reward=self.config['target_reward_threshold'],
                target_height_error=self.config['height_error_threshold'],
                target_stability=self.config['height_stability_threshold'],
                min_episode_length=900,
                patience=self.config.get('early_stop', {}).get('patience')
            )
        ]
        
        # Use custom callback if provided
        if callback is not None:
            callbacks.append(callback)

        os.makedirs("data/checkpoints", exist_ok=True)
        os.makedirs("models/saved_models", exist_ok=True)

        n_envs = getattr(self.env, "num_envs", 4)
        print(f"Starting standing training for {total_timesteps:,} timesteps (n_envs={n_envs})...")
        
        # Log training start to WandB
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({
                "train/started": 1, 
                "train/total_timesteps_target": total_timesteps,
                "train/task_type": "standing"
            })
        
        # Train with all callbacks
        from stable_baselines3.common.callbacks import CallbackList
        combined_callback = CallbackList(callbacks)
        self.model.learn(total_timesteps=total_timesteps, callback=combined_callback)

        # Save final model + normalization stats
        from pathlib import Path

        # Use config path
        final_model_path = self.config.get('final_model_path', 'models/standing_model/final_standing_model')
        model_path = Path(final_model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save WITHOUT .zip extension (SB3 adds it automatically)
        self.model.save(str(model_path))
        print(f"✓ Final model saved: {model_path}.zip")

        # Save VecNormalize
        if isinstance(self.env, VecNormalize):
            vecnorm_path = Path(self.vecnormalize_path)
            vecnorm_path.parent.mkdir(parents=True, exist_ok=True)
            self.env.save(str(vecnorm_path))
            print(f"✓ VecNormalize saved: {vecnorm_path}")
            
            if WANDB_AVAILABLE and wandb.run:
                wandb.save(str(vecnorm_path))

        # Save final model to WandB (add .zip for WandB)
        if WANDB_AVAILABLE and wandb.run:
            wandb.save(str(model_path) + ".zip")
            wandb.log({"train/completed": 1})

        print(f"Training completed!")
        return self.model

    def load_model(self, model_path):
        """Load a trained model and restore normalization stats."""
        # Recreate env first
        self.create_environment()
        self.model = PPO.load(model_path, env=self.env, device=self.config.get("device", "cuda"))
        # Load VecNormalize stats if present
        if isinstance(self.env, VecNormalize) and os.path.exists(self.vecnormalize_path):
            loaded = VecNormalize.load(self.vecnormalize_path, self.env.venv)
            loaded.training = False
            loaded.norm_reward = False
            self.env = loaded
            self.model.set_env(self.env)
        print(f"Standing model loaded from {model_path}")
        return self.model

    def evaluate(self, n_episodes=5, render=False):
        """Evaluate the trained standing model with enhanced metrics."""
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")

        eval_env = self._build_eval_env()
        
        # Enhanced evaluation for standing task
        episode_rewards = []
        episode_lengths = []
        all_heights = []
        episode_heights = []
        
        for ep in range(n_episodes):
            obs = eval_env.reset()
            done = False
            ep_len = 0
            ep_rew = 0
            heights_this_ep = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Use safe_step to handle API differences
                obs, reward, terminated, truncated, info = safe_step(eval_env, action)
                done = bool(terminated[0] if hasattr(terminated, '__len__') else terminated) or \
                       bool(truncated[0] if hasattr(truncated, '__len__') else truncated)
                
                ep_len += 1
                ep_rew += reward[0] if isinstance(reward, np.ndarray) else reward
                
                # Extract height from observation or info
                if hasattr(info, '__len__') and len(info) > 0 and 'height' in info[0]:
                    height = info[0]['height']
                elif hasattr(obs, '__len__') and len(obs) > 0:
                    height = obs[0][0] if hasattr(obs[0], '__len__') else obs[0]
                else:
                    height = None
                
                if height is not None:
                    heights_this_ep.append(height)
                    all_heights.append(height)
            
            episode_rewards.append(ep_rew)
            episode_lengths.append(ep_len)
            episode_heights.append(heights_this_ep)

        eval_env.close()

        # Calculate overall metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        results = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "mean_length": float(mean_length),
            "std_length": float(std_length),
            "episodes": episode_rewards,
        }
        
        # Add standing-specific metrics
        if all_heights:
            mean_height = np.mean(all_heights)
            std_height = np.std(all_heights)
            height_error = abs(mean_height - self.target_height)
            
            # Calculate height stability (std across all episodes)
            height_stability = std_height
            
            results.update({
                "mean_height": float(mean_height),
                "std_height": float(std_height),
                "height_error": float(height_error),
                "height_stability": float(height_stability),
                "target_height": float(self.target_height),
            })

        print(f"\nStanding Evaluation Results:")
        print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Mean Length: {mean_length:.2f} ± {std_length:.2f}")
        
        if all_heights:
            print(f"   Mean Height: {mean_height:.3f} ± {std_height:.3f} (target: {self.target_height:.3f})")
            print(f"   Height Error: {height_error:.3f}")
            print(f"   Height Stability: {height_stability:.3f}")
            
            # Standing success assessment
            success_reward = mean_reward > self.success_threshold
            success_height = height_error < self.config.get('height_error_threshold', 0.1)
            success_stability = height_stability < self.config.get('height_stability_threshold', 0.2)
            overall_success = success_reward and success_height and success_stability
            
            print(f"\nStanding Assessment:")
            print(f"   Reward Success: {'✓' if success_reward else '✗'} "
                  f"({mean_reward:.2f} > {self.success_threshold})")
            print(f"   Height Success: {'✓' if success_height else '✗'} "
                  f"(error {height_error:.3f} < {self.config.get('height_error_threshold', 0.1)})")
            print(f"   Stability Success: {'✓' if success_stability else '✗'} "
                  f"(std {height_stability:.3f} < {self.config.get('height_stability_threshold', 0.2)})")
            print(f"   Overall Success: {'✓ STANDING MASTERED' if overall_success else '✗ NEEDS MORE TRAINING'}")
            
            results["standing_success"] = overall_success

        return results

    def predict(self, observation, deterministic=True):
        """Predict action for a given observation (supports raw or vec env)."""
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        return self.model.predict(observation, deterministic=deterministic)

    def close(self):
        if self.env is not None:
            self.env.close()

    def analyze_standing_performance(self, n_episodes=10):
        """Detailed analysis of standing performance for debugging."""
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        
        print("Analyzing standing performance...")
        eval_env = self._build_eval_env()
        
        all_data = []
        
        for ep in range(n_episodes):
            obs = eval_env.reset()
            episode_data = {
                'heights': [],
                'rewards': [],
                'actions': [],
                'terminated_early': False,
                'termination_step': None,
            }
            
            done = False
            step = 0
            
            while not done and step < 1000:
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Use safe_step to handle API differences
                obs, reward, terminated, truncated, info = safe_step(eval_env, action)
                
                done = bool(terminated[0] if hasattr(terminated, '__len__') else terminated) or \
                       bool(truncated[0] if hasattr(truncated, '__len__') else truncated)
                
                # Extract height
                if hasattr(info, '__len__') and len(info) > 0 and 'height' in info[0]:
                    height = info[0]['height']
                elif hasattr(obs, '__len__') and len(obs) > 0:
                    height = obs[0][0] if hasattr(obs[0], '__len__') else obs[0]
                else:
                    height = None
                
                if height is not None:
                    episode_data['heights'].append(height)
                episode_data['rewards'].append(reward[0] if isinstance(reward, np.ndarray) else reward)
                episode_data['actions'].append(action)
                
                step += 1
                
                if done and step < 1000:
                    episode_data['terminated_early'] = True
                    episode_data['termination_step'] = step
            
            all_data.append(episode_data)
            
            if episode_data['heights']:
                ep_mean_height = np.mean(episode_data['heights'])
                ep_height_std = np.std(episode_data['heights'])
                ep_total_reward = sum(episode_data['rewards'])
                
                print(f"Episode {ep}: height={ep_mean_height:.3f}±{ep_height_std:.3f}, "
                      f"reward={ep_total_reward:.2f}, steps={step}, "
                      f"early_term={'Yes' if episode_data['terminated_early'] else 'No'}")
        
        eval_env.close()
        
        # Overall analysis
        all_heights = [h for ep_data in all_data for h in ep_data['heights']]
        all_rewards = [r for ep_data in all_data for r in ep_data['rewards']]
        
        if all_heights:
            analysis = {
                "mean_height": np.mean(all_heights),
                "height_stability": np.std(all_heights),
                "height_error": abs(np.mean(all_heights) - self.target_height),
                "mean_reward": np.mean(all_rewards),
                "episodes_completed": len([ep for ep in all_data if not ep['terminated_early']]),
                "early_termination_rate": len([ep for ep in all_data if ep['terminated_early']]) / n_episodes,
            }
            
            print(f"\nOverall Standing Analysis:")
            print(f"   Mean Height: {analysis['mean_height']:.3f} ± {analysis['height_stability']:.3f}")
            print(f"   Height Error: {analysis['height_error']:.3f}")
            print(f"   Mean Reward: {analysis['mean_reward']:.3f}")
            print(f"   Episodes Completed: {analysis['episodes_completed']}/{n_episodes}")
            print(f"   Early Termination Rate: {analysis['early_termination_rate']:.1%}")
            
            # Log detailed analysis to WandB
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    "analysis/mean_height": analysis['mean_height'],
                    "analysis/height_stability": analysis['height_stability'],
                    "analysis/height_error": analysis['height_error'],
                    "analysis/mean_reward": analysis['mean_reward'],
                    "analysis/completion_rate": 1 - analysis['early_termination_rate'],
                })
            
            return analysis
        
        return {"error": "No height data collected"}


# ======================================================
# CONVENIENCE FUNCTIONS FOR COLAB
# ======================================================

def create_standing_agent(config_path='config/training_config.yaml', device='auto', use_wandb=True):
    """Quick setup function that uses YAML config."""
    import yaml
    
    # Load config from YAML
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    config = full_config['standing'].copy()
    
    # Only override device if specified
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['device'] = device
    
    # Only override wandb if specified
    if not use_wandb:
        config['use_wandb'] = False
    
    return StandingAgent(config)


class EarlyStoppingCallback(BaseCallback):
    """Stop training when standing is mastered"""
    
    def __init__(self, check_freq=50000, target_reward=5000, target_height_error=0.15, 
                 target_stability=0.1, min_episode_length=500, patience=5):
        super().__init__()
        self.check_freq = check_freq
        self.target_reward = target_reward  # Much higher threshold
        self.target_height_error = target_height_error
        self.target_stability = target_stability
        self.min_episode_length = min_episode_length  # Require long episodes
        self.patience = patience
        self.success_count = 0
        
    def _on_step(self):
        if self.num_timesteps % self.check_freq == 0:
            # Proper evaluation with full episodes
            eval_env = self.training_env
            
            episode_rewards = []
            episode_heights = []
            episode_lengths = []
            
            for _ in range(5):  # Test 5 episodes
                obs = eval_env.reset()
                ep_reward = 0
                ep_heights = []
                ep_length = 0
                
                for step in range(2000):  # Allow longer episodes
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    
                    ep_reward += reward[0] if hasattr(reward, '__len__') else reward
                    ep_length += 1
                    
                    if hasattr(info, '__len__') and len(info) > 0 and 'height' in info[0]:
                        ep_heights.append(info[0]['height'])
                    
                    if any(done) if hasattr(done, '__len__') else done:
                        break
                
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                if ep_heights:
                    episode_heights.extend(ep_heights)
            
            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            if episode_heights:
                mean_height = np.mean(episode_heights)
                height_error = abs(mean_height - 1.4)  # Correct target height
                height_stability = np.std(episode_heights)
                
                # Stricter success criteria
                success = (mean_reward > self.target_reward and 
                          height_error < self.target_height_error and
                          height_stability < self.target_stability and
                          mean_length > self.min_episode_length)  # Must survive long
                
                if success:
                    self.success_count += 1
                    print(f"✓ Standing success {self.success_count}/{self.patience} | "
                          f"reward={mean_reward:.0f}, error={height_error:.3f}, "
                          f"stability={height_stability:.3f}, length={mean_length:.0f}")
                    
                    if self.success_count >= self.patience:
                        print("Standing truly mastered! Stopping training.")
                        return False
                else:
                    self.success_count = 0
                    print(f"✗ Not standing well | reward={mean_reward:.0f}, "
                          f"error={height_error:.3f}, stability={height_stability:.3f}, "
                          f"length={mean_length:.0f}")
        
        return True