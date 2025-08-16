"""
Walking agent for humanoid using PPO
Parallel VecEnvs + VecNormalize, Colab-friendly with WandB logging
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

# Ensure headless Mujoco works in workers
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Import the environment (adjust path for Colab)
try:
    from src.environments.humanoid_env import make_humanoid_env
except ImportError:
    import sys
    sys.path.append('/content')
    from src.environments.humanoid_env import make_humanoid_env


class WandBCallback(BaseCallback):
    """Custom callback for logging to WandB during training."""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log every log_freq steps
        if self.n_calls % self.log_freq == 0 and WANDB_AVAILABLE and wandb.run:
            # Get info from all environments
            infos = self.locals.get("infos", [])
            
            # Collect episode stats if available
            for info in infos:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
            
            # Log training metrics
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
            
            # Get learning rate
            if hasattr(self.model, 'learning_rate'):
                if callable(self.model.learning_rate):
                    lr = self.model.learning_rate(self.model._current_progress_remaining)
                else:
                    lr = self.model.learning_rate
                log_dict["train/learning_rate"] = lr
            
            # Get other training stats from logger if available
            if hasattr(self.model, "logger") and self.model.logger:
                # These values might be available in the logger
                if hasattr(self.model.logger, "name_to_value"):
                    for key, value in self.model.logger.name_to_value.items():
                        if value is not None:
                            # Map SB3 keys to more readable names
                            if "loss" in key.lower():
                                log_dict[f"train/{key}"] = value
                            elif "entropy" in key.lower():
                                log_dict[f"train/{key}"] = value
                            elif "clip" in key.lower():
                                log_dict[f"train/{key}"] = value
            
            wandb.log(log_dict, step=self.num_timesteps)
        
        return True


class WalkingCallback(BaseCallback):
    """Periodic evaluation + checkpointing, with best-model saving and WandB logging."""

    def __init__(
        self,
        eval_env_fn: Optional[Callable[[], DummyVecEnv]] = None,
        n_eval_episodes: int = 3,
        eval_freq: int = 50_000,
        save_freq: int = 250_000,
        verbose: int = 1,
        best_model_path: str = "models/saved_models/best_walking_model.zip",
        checkpoint_dir: str = "data/checkpoints",
        checkpoint_prefix: str = "walking_model",
        wandb_run=None,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.best_model_path = best_model_path
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.best_mean_reward = -np.inf
        self.wandb_run = wandb_run
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        t = self.num_timesteps

        if self.eval_env_fn is not None and self.eval_freq > 0 and (t % self.eval_freq == 0):
            eval_env = self.eval_env_fn()
            mean_rew, std_rew = evaluate_policy(
                self.model, eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True, render=False
            )
            if self.verbose:
                print(f"[eval] t={t:,} mean={mean_rew:.2f} ± {std_rew:.2f}")
            
            # Track best model
            if mean_rew > self.best_mean_reward:
                self.best_mean_reward = mean_rew
                self.model.save(self.best_model_path)
                if self.verbose:
                    print(f"New best model saved to {self.best_model_path} (mean {mean_rew:.2f})")
                # Log to WandB
                if WANDB_AVAILABLE and wandb.run:
                    wandb.save(self.best_model_path)
                    wandb.run.summary["best_mean_reward"] = mean_rew
            
            # Log to tensorboard
            if hasattr(self.model, "logger") and self.model.logger is not None:
                self.model.logger.record("eval/mean_reward", float(mean_rew))
                self.model.logger.record("eval/std_reward", float(std_rew))
                self.model.logger.record("eval/best_mean_reward", float(self.best_mean_reward))
            
            # Log to WandB
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    "eval/mean_reward": float(mean_rew),
                    "eval/std_reward": float(std_rew),
                    "eval/best_mean_reward": float(self.best_mean_reward),
                    "eval/timesteps": t,
                }, step=t)
            
            eval_env.close()

        if self.save_freq > 0 and (t % self.save_freq == 0):
            path = os.path.join(self.checkpoint_dir, f"{self.checkpoint_prefix}_{t}.zip")
            self.model.save(path)
            if self.verbose:
                print(f"Checkpoint saved: {path}")
            # Save checkpoint to WandB
            if WANDB_AVAILABLE and wandb.run:
                wandb.save(path)

        return True


class WalkingAgent:
    """PPO agent for humanoid walking with parallel envs, normalization, and WandB logging."""

    def __init__(self, config):
        self.config = config
        self.model: Optional[PPO] = None
        self.env = None  # VecEnv (possibly VecNormalize)
        self.vecnormalize_path = self.config.get(
            "vecnormalize_path", "models/saved_models/vecnorm_walking.pkl"
        )
        self.wandb_run = config.get('wandb_run', None)

    # ---------- Env construction ----------

    def _make_single_env(self, seed: int, rank: int, render_mode=None) -> Callable:
        """Factory that returns a thunk creating one monitored env with seeding."""
        def _init():
            os.environ.setdefault("MUJOCO_GL", "egl")
            env = make_humanoid_env("walking", render_mode=render_mode)
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
                norm_reward=True,
                clip_obs=10.0,
                gamma=self.config.get("gamma", 0.99),
            )
        else:
            self.env = vec
        return self.env

    # ---------- Model construction ----------

    def create_model(self):
        """Create PPO model on the requested device."""
        if self.env is None:
            self.create_environment()

        model_params = {
            "learning_rate": self.config.get("learning_rate", 3e-4),
            "n_steps": self.config.get("n_steps", 2048),      # per-env
            "batch_size": self.config.get("batch_size", 256),
            "n_epochs": self.config.get("n_epochs", 10),
            "gamma": self.config.get("gamma", 0.99),
            "gae_lambda": self.config.get("gae_lambda", 0.95),
            "clip_range": self.config.get("clip_range", 0.1),
            "ent_coef": self.config.get("ent_coef", 0.001),
            "vf_coef": self.config.get("vf_coef", 0.5),
            "max_grad_norm": self.config.get("max_grad_norm", 0.5),
            "verbose": self.config.get("verbose", 1),
            "seed": self.config.get("seed", 42),
            "device": self.config.get("device", "cuda"),
        }

        policy_kwargs = self.config.get("policy_kwargs", {"net_arch": dict(pi=[400, 300], vf=[400, 300])})

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
        """Train the agent with periodic eval/checkpoints and save VecNormalize stats."""
        if self.model is None:
            self.create_model()
        if total_timesteps is None:
            total_timesteps = int(self.config.get("total_timesteps", 2_000_000))

        # Prepare eval env factory that uses current normalization stats
        def eval_env_fn():
            # Save current stats so eval can load the latest normalization
            if isinstance(self.env, VecNormalize):
                os.makedirs(os.path.dirname(self.vecnormalize_path), exist_ok=True)
                self.env.save(self.vecnormalize_path)
            return self._build_eval_env()

        # Create callbacks list
        callbacks = []
        
        # Add WandB callback
        if WANDB_AVAILABLE and wandb.run:
            wandb_callback = WandBCallback(
                log_freq=self.config.get("wandb_log_freq", 1000),
                verbose=self.config.get("verbose", 1)
            )
            callbacks.append(wandb_callback)
        
        # Add evaluation callback
        eval_callback = WalkingCallback(
            eval_env_fn=eval_env_fn,
            n_eval_episodes=self.config.get("n_eval_episodes", 3),
            eval_freq=self.config.get("eval_freq", 50_000),
            save_freq=self.config.get("save_freq", 250_000),
            verbose=self.config.get("verbose", 1),
            best_model_path=self.config.get("best_model_path", "models/saved_models/best_walking_model.zip"),
            checkpoint_dir=self.config.get("checkpoint_dir", "data/checkpoints"),
            checkpoint_prefix=self.config.get("checkpoint_prefix", "walking_model"),
            wandb_run=self.wandb_run,
        )
        callbacks.append(eval_callback)
        
        # Use custom callback if provided
        if callback is not None:
            callbacks.append(callback)

        os.makedirs("data/checkpoints", exist_ok=True)
        os.makedirs("models/saved_models", exist_ok=True)

        n_envs = getattr(self.env, "num_envs", 1)
        print(f"Starting training for {total_timesteps:,} timesteps (n_envs={n_envs})...")
        
        # Log training start to WandB
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({"train/started": 1, "train/total_timesteps_target": total_timesteps})
        
        # Train with all callbacks
        from stable_baselines3.common.callbacks import CallbackList
        combined_callback = CallbackList(callbacks)
        self.model.learn(total_timesteps=total_timesteps, callback=combined_callback)

        # Save final model + normalization stats
        final_model_path = "models/saved_models/final_walking_model.zip"
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
        print(f"Model loaded from {model_path}")
        return self.model

    def evaluate(self, n_episodes=5, render=False):
        """Evaluate the trained model with consistent normalization."""
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")

        eval_env = self._build_eval_env()
        mean_rew, std_rew = evaluate_policy(
            self.model, eval_env, n_eval_episodes=n_episodes, deterministic=True, render=render
        )

        # Manual rollout stats per-episode length if desired
        lengths = []
        rewards = []
        for ep in range(n_episodes):
            obs = eval_env.reset()
            done = False
            ep_len = 0
            ep_rew = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = bool(terminated | truncated)
                ep_len += 1
                ep_rew += reward[0] if isinstance(reward, np.ndarray) else reward
            lengths.append(ep_len)
            rewards.append(ep_rew)
            
            # Log individual episode to WandB
            if WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    f"eval_episode/reward_ep_{ep}": ep_rew,
                    f"eval_episode/length_ep_{ep}": ep_len,
                })

        eval_env.close()

        print(f"\nEvaluation Results:")
        print(f"   Mean Reward: {mean_rew:.2f} ± {std_rew:.2f}")
        print(f"   Mean Length: {float(np.mean(lengths)):.2f} ± {float(np.std(lengths)):.2f}")

        return {
            "mean_reward": float(mean_rew),
            "std_reward": float(std_rew),
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "episodes": rewards,
        }

    def predict(self, observation, deterministic=True):
        """Predict action for a given observation (supports raw or vec env)."""
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        return self.model.predict(observation, deterministic=deterministic)

    def close(self):
        if self.env is not None:
            self.env.close()