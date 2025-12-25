# standing_env.py
"""
Standing environment for humanoid balance training.
First stage of progressive training: Standing -> Walking -> Navigation.

Reward focuses on:
- Maintaining upright posture
- Stable height around 1.4m
- Minimal movement/drift
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium.spaces import Box


class StandingEnv(gym.Wrapper):
    """
    Standing environment that trains humanoid to maintain balance.
    
    This is the first stage of progressive training before walking.
    """
    
    def __init__(self, render_mode: Optional[str] = None, config=None):
        env_id = "Humanoid-v5"
        print(f"Using {env_id} for standing task")
        
        # Create base environment
        env = gym.make(
            env_id, 
            render_mode=render_mode,
            exclude_current_positions_from_observation=False
        )
        super().__init__(env)
        
        # Configuration
        self.cfg = config or {}
        self.base_target_height = float(self.cfg.get('target_height', 1.4))
        self.max_episode_steps = int(self.cfg.get('max_episode_steps', 10000))
        self.current_step = 0
        
        # Domain randomization
        self.domain_rand = self.cfg.get('domain_rand', False)
        self.rand_mass_range = self.cfg.get('rand_mass_range', [0.95, 1.05])
        self.rand_friction_range = self.cfg.get('rand_friction_range', [0.95, 1.05])
        
        # Random height initialization for recovery training
        self.random_height_init = self.cfg.get('random_height_init', True)
        self.random_height_prob = self.cfg.get('random_height_prob', 0.3)
        self.random_height_range = self.cfg.get('random_height_range', [-0.3, 0.1])
        
        # Reward caps
        reward_caps = self.cfg.get('reward_caps', {})
        self.recovery_bonus_scale = reward_caps.get('recovery_bonus_scale', 50.0)
        self.termination_penalty = reward_caps.get('termination_penalty_constant', 50.0)
        
        self.reward_history = {
            'height': [], 'upright': [], 'stability': [], 'control': []
        }
        
        # Action smoothing
        self.enable_action_smoothing = bool(self.cfg.get('action_smoothing', False))
        self.action_smoothing_tau = float(self.cfg.get('action_smoothing_tau', 0.5))
        self.prev_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        
        # Observation processing
        self.enable_history = int(self.cfg.get('obs_history', 0)) > 0
        self.history_len = int(self.cfg.get('obs_history', 0))
        self.obs_history = []
        self.include_com = bool(self.cfg.get('obs_include_com', False))
        self.feature_norm = bool(self.cfg.get('obs_feature_norm', False))
        
        # Calculate observation dimension
        base_obs_from_space = int(env.observation_space.shape[0])
        base_obs_dim = base_obs_from_space + 15  # Actual observations
        
        extra_dim = 6 if self.include_com else 0
        feature_dim = base_obs_dim + extra_dim
        
        if self.enable_history:
            total_dim = feature_dim * self.history_len
        else:
            total_dim = feature_dim
        
        self.frozen_obs_dim = total_dim
        
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.frozen_obs_dim,), 
            dtype=np.float32
        )
        
        print(f"Standing environment observation dimension: {self.frozen_obs_dim}")
    
    def reset(self, seed: Optional[int] = None):
        observation, info = self.env.reset(seed=seed)
        
        default_height = self.env.unwrapped.data.qpos[2]
        
        self.current_step = 0
        self.prev_height = default_height
        self.prev_action[:] = 0.0
        self.obs_history = []
        
        for key in self.reward_history:
            self.reward_history[key] = []
        
        # Domain randomization
        if self.domain_rand:
            self.env.unwrapped.model.body_mass *= np.random.uniform(
                self.rand_mass_range[0], self.rand_mass_range[1],
                size=self.env.unwrapped.model.body_mass.shape
            )
            self.env.unwrapped.model.geom_friction[:, 0] *= np.random.uniform(
                self.rand_friction_range[0], self.rand_friction_range[1],
                size=self.env.unwrapped.model.geom_friction.shape[0]
            )
        
        # Random height initialization
        if self.random_height_init and np.random.random() < self.random_height_prob:
            perturb = np.random.uniform(self.random_height_range[0], self.random_height_range[1])
            new_height = np.clip(default_height + perturb, 0.6, 1.6)
            self.env.unwrapped.data.qpos[2] = new_height
            self.prev_height = new_height
            observation = self.env.unwrapped._get_obs()
        
        observation = self._process_observation(observation)
        return observation, info
    
    def step(self, action):
        proc_action = self._process_action(np.asarray(action, dtype=np.float32))
        
        observation, base_reward, terminated, truncated, info = self.env.step(proc_action)
        self.current_step += 1
        
        reward, terminated = self._compute_standing_reward(observation, proc_action)
        
        truncated = self.current_step >= self.max_episode_steps
        
        info.update(self._get_task_info())
        observation = self._process_observation(observation)
        
        return observation, reward, terminated, truncated, info
    
    def _compute_standing_reward(self, obs, action):
        """Compute standing reward - focus on balance and stability."""
        height = self.env.unwrapped.data.qpos[2]
        quat = self.env.unwrapped.data.qpos[3:7]
        linear_vel = self.env.unwrapped.data.qvel[0:3]
        angular_vel = self.env.unwrapped.data.qvel[3:6]
        
        # Height reward (Gaussian around target)
        height_error = abs(height - self.base_target_height)
        height_reward = 10.0 * np.exp(-5.0 * height_error**2)
        
        # Upright reward (quaternion w close to 1)
        upright_error = 1.0 - abs(quat[0])
        upright_reward = 10.0 * np.exp(-8.0 * upright_error**2) if height >= 1.0 else 0.0
        
        # Stability reward (minimize velocities)
        linear_speed = np.linalg.norm(linear_vel)
        angular_speed = np.linalg.norm(angular_vel)
        stability_reward = 5.0 * np.exp(-2.0 * linear_speed) + 3.0 * np.exp(-1.5 * angular_speed)
        
        # Control cost
        control_cost = -0.01 * np.sum(action ** 2)
        
        # Total reward
        total_reward = height_reward + upright_reward + stability_reward + control_cost
        
        # Recovery bonus
        height_velocity = height - self.prev_height
        if height < 1.0 and height_velocity > 0.01:
            recovery_scale = (1.0 - height) / 0.4
            total_reward += self.recovery_bonus_scale * height_velocity * recovery_scale
        
        # Termination
        terminate = (height < 0.75 or height > 2.0 or abs(quat[0]) < 0.5)
        if terminate:
            total_reward -= self.termination_penalty
        
        self.prev_height = height
        
        # Track components
        self.reward_history['height'].append(height_reward)
        self.reward_history['upright'].append(upright_reward)
        self.reward_history['stability'].append(stability_reward)
        self.reward_history['control'].append(control_cost)
        
        return total_reward, terminate
    
    def _process_action(self, action: np.ndarray) -> np.ndarray:
        if self.enable_action_smoothing:
            tau = np.clip(self.action_smoothing_tau, 0.0, 1.0)
            action = (1.0 - tau) * self.prev_action + tau * action
        
        low, high = self.env.action_space.low, self.env.action_space.high
        action = np.clip(action, low, high)
        self.prev_action = action.copy()
        return action
    
    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        features = [obs]
        
        if self.include_com:
            try:
                com_pos = self.env.unwrapped.data.subtree_com[0]
                com_vel = self.env.unwrapped.data.qvel[:3]
                features.append(np.asarray(com_pos, dtype=np.float32))
                features.append(np.asarray(com_vel, dtype=np.float32))
            except Exception:
                pass
        
        feat_vec = np.concatenate([np.atleast_1d(f).ravel() for f in features]).astype(np.float32)
        
        if self.feature_norm:
            feat_vec = np.tanh(feat_vec * 0.1)
        
        if self.enable_history:
            self.obs_history.append(feat_vec)
            if len(self.obs_history) > self.history_len:
                self.obs_history = self.obs_history[-self.history_len:]
            
            if len(self.obs_history) < self.history_len:
                pad_count = self.history_len - len(self.obs_history)
                padded = [np.zeros_like(feat_vec) for _ in range(pad_count)] + self.obs_history
            else:
                padded = self.obs_history
            
            feat_vec = np.concatenate(padded, axis=0)
        
        # Handle dimension mismatch
        current_dim = feat_vec.shape[0]
        if current_dim != self.frozen_obs_dim:
            if current_dim > self.frozen_obs_dim:
                feat_vec = feat_vec[:self.frozen_obs_dim]
            else:
                pad = np.zeros((self.frozen_obs_dim - current_dim,), dtype=np.float32)
                feat_vec = np.concatenate([feat_vec, pad], axis=0)
        
        return feat_vec
    
    def _get_task_info(self):
        height = self.env.unwrapped.data.qpos[2]
        quat = self.env.unwrapped.data.qpos[3:7]
        linear_vel = self.env.unwrapped.data.qvel[0:3]
        
        return {
            'height': height,
            'quaternion_w': quat[0],
            'x_velocity': linear_vel[0],
            'y_velocity': linear_vel[1],
            'xy_drift': np.sqrt(self.env.unwrapped.data.qpos[0]**2 + self.env.unwrapped.data.qpos[1]**2),
        }


def make_standing_env(render_mode=None, config=None):
    """Create standing environment with given config."""
    return StandingEnv(render_mode=render_mode, config=config)


if __name__ == "__main__":
    print("Testing Standing Environment")
    print("=" * 60)
    
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.5,
        'random_height_init': False,
    }
    
    env = make_standing_env(render_mode=None, config=config)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Expected: {env.frozen_obs_dim}")
    
    print("\nRunning 200 steps...")
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 50 == 0:
            print(f"Step {step}: height={info['height']:.3f}, reward={reward:.2f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    env.close()
    print("Test completed!")

