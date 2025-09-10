"""
Standing agent for humanoid using PPO
Parallel VecEnvs + VecNormalize, Colab-friendly with WandB logging
Optimized for stability and balance tasks
FIXED: Gym API compatibility issues
"""

import os
from typing import Callable, Optional

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
    from src.environments.humanoid_env import make_humanoid_env
except ImportError:
    import sys
    sys.path.append('/content')
    from src.environments.humanoid_env import make_humanoid_env


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


class StandingCallback(BaseCallback):
    """Unified callback for standing task: evaluation + checkpointing + WandB logging."""

    def __init__(
        self,
        eval_env_fn: Optional[Callable[[], DummyVecEnv]] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 40_000,
        save_freq: int = 200_000,
        log_freq: int = 1000,  # NEW: WandB logging frequency
        verbose: int = 1,
        best_model_path: str = "models/saved_models/best_standing_model.zip",
        checkpoint_dir: str = "data/checkpoints",
        checkpoint_prefix: str = "standing_model",
        wandb_run=None,
        target_height: float = 1.3,
        success_threshold: float = 150.0,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.log_freq = log_freq  # NEW
        self.best_model_path = best_model_path
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.best_mean_reward = -np.inf
        self.wandb_run = wandb_run
        self.target_height = target_height
        self.success_threshold = success_threshold
        
        # Standing-specific tracking
        self.best_height_error = np.inf
        self.best_height_stability = np.inf
        
        # NEW: WandB training data tracking (moved from StandingWandBCallback)
        self.episode_rewards = []
        self.episode_lengths = []
        self.height_data = []
        self.stability_data = []
        
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        t = self.num_timesteps

        # NEW: WandB training logging (every log_freq steps)
        if self.log_freq > 0 and (t % self.log_freq == 0) and WANDB_AVAILABLE and wandb.run:
            self._log_training_metrics()

        # Evaluation (every eval_freq steps)
        if self.eval_env_fn is not None and self.eval_freq > 0 and (t % self.eval_freq == 0):
            self._run_evaluation()

        # Checkpointing (every save_freq steps)
        if self.save_freq > 0 and (t % self.save_freq == 0):
            self._save_checkpoint()

        return True

    def _log_training_metrics(self):
        """Log training metrics to WandB (merged from StandingWandBCallback)"""
        # Get info from all environments
        infos = self.locals.get("infos", [])
        
        # Collect episode stats and standing-specific metrics
        current_heights = []
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
            
            # Collect height data for standing analysis
            if "height" in info:
                current_heights.append(info["height"])
                self.height_data.append(info["height"])
        
        # Build log dictionary
        log_dict = {
            "train/timesteps": self.num_timesteps,
            "train/n_updates": self.model.n_updates if hasattr(self.model, 'n_updates') else 0,
        }
        
        # Add episode stats if we have them
        if self.episode_rewards:
            log_dict.update({
                "train/episode_reward_mean": np.mean(self.episode_rewards[-100:]),
                "train/episode_reward_std": np.std(self.episode_rewards[-100:]),
                "train/episode_length_mean": np.mean(self.episode_lengths[-100:]),
            })
        
        # Add standing-specific metrics
        if current_heights:
            log_dict.update({
                "train/height_mean": np.mean(current_heights),
                "train/height_std": np.std(current_heights),
                "train/height_error": abs(np.mean(current_heights) - self.target_height),
            })
        
        if len(self.height_data) >= 50:  # Need some history for stability
            recent_heights = self.height_data[-50:]
            log_dict["train/height_stability"] = np.std(recent_heights)
        
        # Get learning rate
        if hasattr(self.model, 'learning_rate'):
            if callable(self.model.learning_rate):
                lr = self.model.learning_rate(self.model._current_progress_remaining)
            else:
                lr = self.model.learning_rate
            log_dict["train/learning_rate"] = lr
        
        # Get other training stats from logger if available
        if hasattr(self.model, "logger") and self.model.logger:
            if hasattr(self.model.logger, "name_to_value"):
                for key, value in self.model.logger.name_to_value.items():
                    if value is not None:
                        if "loss" in key.lower():
                            log_dict[f"train/{key}"] = value
                        elif "entropy" in key.lower():
                            log_dict[f"train/{key}"] = value
                        elif "clip" in key.lower():
                            log_dict[f"train/{key}"] = value
        
        wandb.log(log_dict, step=self.num_timesteps)

    def _run_evaluation(self):
        """Run evaluation and track best models (existing code with minor updates)"""
        eval_env = self.eval_env_fn()
        
        # Enhanced evaluation for standing task
        heights = []
        rewards = []
        episode_lengths = []
        
        for ep in range(self.n_eval_episodes):
            obs = eval_env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            ep_heights = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Use safe_step to handle API differences
                obs, reward, terminated, truncated, info = safe_step(eval_env, action)
                done = bool(terminated[0] if hasattr(terminated, '__len__') else terminated) or \
                       bool(truncated[0] if hasattr(truncated, '__len__') else truncated)
                
                ep_reward += reward[0] if hasattr(reward, '__len__') else reward
                ep_length += 1
                
                # Extract height from observation or info
                if hasattr(info, '__len__') and len(info) > 0 and 'height' in info[0]:
                    ep_heights.append(info[0]['height'])
                elif hasattr(obs, '__len__') and len(obs) > 0:
                    ep_heights.append(obs[0][0] if hasattr(obs[0], '__len__') else obs[0])
            
            rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            if ep_heights:
                heights.extend(ep_heights)
        
        mean_rew = np.mean(rewards)
        std_rew = np.std(rewards)
        
        # Calculate standing-specific metrics
        standing_metrics = {}
        if heights:
            mean_height = np.mean(heights)
            height_error = abs(mean_height - self.target_height)
            height_stability = np.std(heights)
            
            standing_metrics = {
                "mean_height": mean_height,
                "height_error": height_error,
                "height_stability": height_stability,
                "target_height": self.target_height,
            }
            
            if self.verbose:
                print(f"[eval] t={self.num_timesteps:,} reward={mean_rew:.2f}±{std_rew:.2f} "
                      f"height={mean_height:.3f}±{height_stability:.3f} "
                      f"error={height_error:.3f}")
        else:
            if self.verbose:
                print(f"[eval] t={self.num_timesteps:,} reward={mean_rew:.2f}±{std_rew:.2f}")
        
        # Track best model (considering both reward and standing metrics)
        is_new_best = False
        if mean_rew > self.best_mean_reward:
            if not standing_metrics or standing_metrics["height_error"] < 0.2:  # Reasonable height error
                self.best_mean_reward = mean_rew
                if standing_metrics:
                    self.best_height_error = standing_metrics["height_error"]
                    self.best_height_stability = standing_metrics["height_stability"]
                is_new_best = True
        
        # Also save model if significantly better standing performance
        elif (standing_metrics and 
              standing_metrics["height_error"] < self.best_height_error * 0.9 and
              mean_rew > self.success_threshold):
            self.best_mean_reward = mean_rew
            self.best_height_error = standing_metrics["height_error"]
            self.best_height_stability = standing_metrics["height_stability"]
            is_new_best = True
        
        if is_new_best:
            self.model.save(self.best_model_path)
            if self.verbose:
                print(f"New best standing model saved (reward={mean_rew:.2f}, "
                      f"height_error={standing_metrics.get('height_error', 'N/A'):.3f})")
            
            # Log to WandB
            if WANDB_AVAILABLE and wandb.run:
                wandb.save(self.best_model_path)
                wandb.run.summary["best_mean_reward"] = mean_rew
                if standing_metrics:
                    wandb.run.summary["best_height_error"] = standing_metrics["height_error"]
                    wandb.run.summary["best_height_stability"] = standing_metrics["height_stability"]
        
        # Log to tensorboard
        if hasattr(self.model, "logger") and self.model.logger is not None:
            self.model.logger.record("eval/mean_reward", float(mean_rew))
            self.model.logger.record("eval/std_reward", float(std_rew))
            self.model.logger.record("eval/best_mean_reward", float(self.best_mean_reward))
            
            if standing_metrics:
                for key, value in standing_metrics.items():
                    self.model.logger.record(f"eval/{key}", float(value))
        
        # Log to WandB
        if WANDB_AVAILABLE and wandb.run:
            log_data = {
                "eval/mean_reward": float(mean_rew),
                "eval/std_reward": float(std_rew),
                "eval/best_mean_reward": float(self.best_mean_reward),
                "eval/mean_length": float(np.mean(episode_lengths)),
                "eval/timesteps": self.num_timesteps,
            }
            
            if standing_metrics:
                for key, value in standing_metrics.items():
                    log_data[f"eval/{key}"] = float(value)
            
            wandb.log(log_data, step=self.num_timesteps)
        
        eval_env.close()

    def _save_checkpoint(self):
        """Save model checkpoint"""
        path = os.path.join(self.checkpoint_dir, f"{self.checkpoint_prefix}_{self.num_timesteps}.zip")
        self.model.save(path)
        if self.verbose:
            print(f"Checkpoint saved: {path}")
        # Save checkpoint to WandB
        if WANDB_AVAILABLE and wandb.run:
            wandb.save(path)


class StandingAgent:
    """PPO agent for humanoid standing with parallel envs, normalization, and WandB logging."""

    def __init__(self, config):
        self.config = config
        self.model: Optional[PPO] = None
        self.env = None  # VecEnv (possibly VecNormalize)
        self.vecnormalize_path = self.config.get(
            "vecnormalize_path", "models/saved_models/vecnorm_standing.pkl"
        )
        self.wandb_run = config.get('wandb_run', None)
        
        # Standing-specific parameters
        self.target_height = config.get('target_height', 1.3)
        self.success_threshold = config.get('target_reward_threshold', 150.0)

    # ---------- Env construction ----------

    def _make_single_env(self, seed: int, rank: int, render_mode=None) -> Callable:
        """Factory that returns a thunk creating one monitored env with seeding."""
        def _init():
            os.environ.setdefault("MUJOCO_GL", "egl")
            env = make_humanoid_env("standing", render_mode=render_mode)
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
        n_envs = int(self.config.get("n_envs", 1))
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
                gamma=self.config.get("gamma", 0.995),
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
            "learning_rate": self.config.get("learning_rate", 5e-5),
            "n_steps": self.config.get("n_steps", 2048),
            "batch_size": self.config.get("batch_size", 128),
            "n_epochs": self.config.get("n_epochs", 6),
            "gamma": self.config.get("gamma", 0.995),
            "gae_lambda": self.config.get("gae_lambda", 0.98),
            "clip_range": self.config.get("clip_range", 0.1),
            "ent_coef": self.config.get("ent_coef", 0.01),
            "vf_coef": self.config.get("vf_coef", 0.8),
            "max_grad_norm": self.config.get("max_grad_norm", 0.3),
            "verbose": self.config.get("verbose", 1),
            "seed": self.config.get("seed", 42),
            "device": self.config.get("device", "cuda"),
        }

        # Standing-optimized architecture (deeper for fine control)
        policy_kwargs = self.config.get("policy_kwargs", {
            "net_arch": {"pi": [512, 256, 128], "vf": [512, 256, 128]},
            "activation_fn": "tanh"  # Better for control tasks
        })

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
        """Train the standing agent with periodic eval/checkpoints and save VecNormalize stats."""
        if self.model is None:
            self.create_model()
        if total_timesteps is None:
            total_timesteps = int(self.config.get("total_timesteps", 1_500_000))  # Less training needed for standing

        # Prepare eval env factory that uses current normalization stats
        def eval_env_fn():
            # Save current stats so eval can load the latest normalization
            if isinstance(self.env, VecNormalize):
                os.makedirs(os.path.dirname(self.vecnormalize_path), exist_ok=True)
                self.env.save(self.vecnormalize_path)
            return self._build_eval_env()

        # Create callbacks list
        callbacks = [
            StandingCallback(
                eval_env_fn=eval_env_fn,
                n_eval_episodes=self.config.get("n_eval_episodes", 5),
                eval_freq=self.config.get("eval_freq", 40_000),
                save_freq=self.config.get("save_freq", 200_000),
                log_freq=self.config.get("wandb_log_freq", 1000),  # NEW PARAMETER
                verbose=self.config.get("verbose", 1),
                wandb_run=self.wandb_run,  # Pass WandB run
                best_model_path=self.config.get("best_model_path", "models/saved_models/best_standing_model.zip"),
                checkpoint_dir=self.config.get("checkpoint_dir", "data/checkpoints"),
                checkpoint_prefix=self.config.get("checkpoint_prefix", "standing_model"),
                target_height=self.target_height,
                success_threshold=self.success_threshold,
            )
        ]
        
        # Use custom callback if provided
        if callback is not None:
            callbacks.append(callback)

        os.makedirs("data/checkpoints", exist_ok=True)
        os.makedirs("models/saved_models", exist_ok=True)

        n_envs = getattr(self.env, "num_envs", 1)
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
        final_model_path = "models/saved_models/final_standing_model.zip"
        self.model.save(final_model_path)
        if isinstance(self.env, VecNormalize):
            self.env.save(self.vecnormalize_path)
            if WANDB_AVAILABLE and wandb.run:
                wandb.save(self.vecnormalize_path)
        
        # Save final model to WandB
        if WANDB_AVAILABLE and wandb.run:
            wandb.save(final_model_path)
            wandb.log({"train/completed": 1})
        
        print(f"Training completed! Final model: {final_model_path}")
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
            
            # Log individual episode to WandB
            if WANDB_AVAILABLE and wandb.run:
                ep_metrics = {
                    f"eval_episode/reward_ep_{ep}": ep_rew,
                    f"eval_episode/length_ep_{ep}": ep_len,
                }
                
                if heights_this_ep:
                    ep_metrics.update({
                        f"eval_episode/mean_height_ep_{ep}": np.mean(heights_this_ep),
                        f"eval_episode/height_stability_ep_{ep}": np.std(heights_this_ep),
                        f"eval_episode/height_error_ep_{ep}": abs(np.mean(heights_this_ep) - self.target_height),
                    })
                
                wandb.log(ep_metrics)

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

def create_standing_agent(device='auto', use_wandb=True, n_envs=2):
    """Quick setup function for Colab notebooks."""
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {
        'device': device,
        'n_envs': n_envs,
        'normalize': True,
        'learning_rate': 1e-4,
        'batch_size': 128,
        'n_steps': 2048,
        'n_epochs': 8,
        'gamma': 0.995,
        'target_height': 1.3,
        'target_reward_threshold': 150.0,
        'height_error_threshold': 0.1,
        'height_stability_threshold': 0.2,
        'verbose': 1,
        'seed': 42,
        'policy_kwargs': {
            'net_arch': {'pi': [512, 256, 128], 'vf': [512, 256, 128]},
            'activation_fn': 'tanh',
        },
    }
    
    if use_wandb:
        config['use_wandb'] = True
        config['wandb_project'] = 'humanoid-standing'
    
    return StandingAgent(config)
