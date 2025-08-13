"""
Walking agent for humanoid using PPO
Designed for Google Colab training
"""

import os
import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Import the environment (adjust path for Colab)
try:
    from src.environments.humanoid_env import make_humanoid_env
except ImportError:
    # Fallback for Colab
    import sys
    sys.path.append('/content')
    from src.environments.humanoid_env import make_humanoid_env


class WalkingCallback(BaseCallback):
    """Custom callback for training progress and model saving"""
    
    def __init__(self, eval_freq=10000, save_freq=25000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.best_mean_reward = -np.inf
        
    def _on_step(self):
        # Evaluate model periodically
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate_model()
        
        # Save model periodically
        if self.num_timesteps % self.save_freq == 0:
            model_path = f"data/checkpoints/walking_model_{self.num_timesteps}.zip"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved: {model_path}")
        
        return True
    
    def _evaluate_model(self, n_episodes=3):
        """Evaluate current model performance"""
        env = make_humanoid_env("walking")
        
        episode_rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        
        # Save best model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            best_model_path = "models/saved_models/best_walking_model.zip"
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            self.model.save(best_model_path)
            if self.verbose > 0:
                print(f"🏆 New best model! Reward: {mean_reward:.2f}")
        
        env.close()
        
        # Log to tensorboard
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            self.model.logger.record('eval/mean_reward', mean_reward)
            self.model.logger.record('eval/best_mean_reward', self.best_mean_reward)


class WalkingAgent:
    """PPO agent for humanoid walking"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.env = None
        
    def create_environment(self, render_mode=None):
        """Create the walking environment"""
        env = make_humanoid_env("walking", render_mode=render_mode)
        
        # Add monitoring for logging
        log_dir = self.config.get('log_dir', 'data/logs')
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, 'training.csv'))
        
        # Vectorize for stable-baselines3
        self.env = DummyVecEnv([lambda: env])
        return self.env
    
    def create_model(self):
        """Create PPO model"""
        if self.env is None:
            self.create_environment()
        
        # PPO parameters from config
        model_params = {
            'learning_rate': self.config.get('learning_rate', 0.0003),
            'n_steps': self.config.get('n_steps', 2048),
            'batch_size': self.config.get('batch_size', 64),
            'n_epochs': self.config.get('n_epochs', 10),
            'gamma': self.config.get('gamma', 0.99),
            'gae_lambda': self.config.get('gae_lambda', 0.95),
            'clip_range': self.config.get('clip_range', 0.2),
            'ent_coef': self.config.get('ent_coef', 0.0),
            'vf_coef': self.config.get('vf_coef', 0.5),
            'max_grad_norm': self.config.get('max_grad_norm', 0.5),
            'verbose': self.config.get('verbose', 1),
            'seed': self.config.get('seed', 42),
            'device': self.config.get('device', 'auto'),
        }
        
        # Network architecture
        policy_kwargs = self.config.get('policy_kwargs', {
            'net_arch': dict(pi=[400, 300], vf=[400, 300])
        })
        import torch.nn as nn
        
        if 'activation_fn' in policy_kwargs:
            act_fn_str = policy_kwargs['activation_fn']
            if isinstance(act_fn_str, str):
                try:
                    policy_kwargs['activation_fn'] = getattr(nn, act_fn_str)
                except AttributeError:
                    raise ValueError(f"Invalid activation function '{act_fn_str}'. Use a valid torch.nn module name like 'ReLU' or 'Tanh'.")
        
        # Tensorboard logging
        tensorboard_log = self.config.get('log_dir', 'data/logs') if self.config.get('use_tensorboard', True) else None
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            **model_params
        )
        
        return self.model
    
    def train(self, total_timesteps=None, callback=None):
        """Train the agent"""
        if self.model is None:
            self.create_model()
        
        if total_timesteps is None:
            total_timesteps = self.config.get('total_timesteps', 500000)
        
        # Setup callback
        if callback is None:
            callback = WalkingCallback(
                eval_freq=self.config.get('eval_freq', 10000),
                save_freq=self.config.get('save_freq', 25000),
                verbose=self.config.get('verbose', 1)
            )
        
        # Create directories
        os.makedirs('data/checkpoints', exist_ok=True)
        os.makedirs('models/saved_models', exist_ok=True)
        
        print(f"Starting training for {total_timesteps:,} timesteps...")
        
        # Train
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Save final model
        final_model_path = "models/saved_models/final_walking_model.zip"
        self.model.save(final_model_path)
        print(f"Training completed! Final model: {final_model_path}")
        
        return self.model
    
    def load_model(self, model_path):
        """Load a trained model"""
        if self.env is None:
            self.create_environment()
        
        self.model = PPO.load(model_path, env=self.env)
        print(f"Model loaded from {model_path}")
        return self.model
    
    def evaluate(self, n_episodes=5, render=False):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        
        render_mode = "human" if render else None
        eval_env = make_humanoid_env("walking", render_mode=render_mode)
        
        episode_rewards = []
        episode_lengths = []
        
        print(f"Evaluating model for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                if render:
                    eval_env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        eval_env.close()
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        print(f"\nEvaluation Results:")
        print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Mean Length: {mean_length:.2f} ± {std_length:.2f}")
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'std_length': std_length,
            'episodes': episode_rewards
        }
    
    def predict(self, observation, deterministic=True):
        """Predict action for given observation"""
        if self.model is None:
            raise ValueError("No model available. Train or load a model first.")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def close(self):
        """Clean up resources"""
        if self.env is not None:
            self.env.close()