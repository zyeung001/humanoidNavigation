"""
Humanoid environment wrapper for MuJoCo Humanoid-v4
Simple wrapper for walking, standing, and navigation tasks
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class HumanoidEnv(gym.Wrapper):  # Inherit from Wrapper
    """Wrapper for MuJoCo Humanoid environment with task-specific modifications"""
    
    def __init__(self, task_type: str = "walking", render_mode: Optional[str] = None):
        env = gym.make("Humanoid-v5", render_mode=render_mode)  # Create the base env first
        super().__init__(env)  # Pass it to super() for proper wrapping
        self.task_type = task_type
        
        # Task parameters
        self.target_position = None
        self.max_episode_steps = 1000
        self.current_step = 0
        
    # No need for explicit @property overrides; they are inherited from Wrapper
    
    def reset(self, seed: Optional[int] = None):
        observation, info = self.env.reset(seed=seed)
        self.current_step = 0
        
        if self.task_type == "navigation":
            self._set_random_target()
            
        return observation, info
    
    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Modify reward based on task
        reward = self._compute_task_reward(observation, base_reward, info)
        
        # Check task-specific termination
        task_terminated = self._check_task_termination(observation, info)
        terminated = terminated or task_terminated
        
        # Episode length limit
        truncated = truncated or (self.current_step >= self.max_episode_steps)
        
        # Add task info
        info.update(self._get_task_info(observation))
        
        return observation, reward, terminated, truncated, info
    
    def _compute_task_reward(self, obs, base_reward, info):
        """Compute task-specific reward"""
        if self.task_type == "walking":
            height = obs[0]
            # Target upright height for torso
            upright_bonus = 2.0 if height > 1.2 else -2.0

            # Penalize torso tilt (if orientation quaternions/angles are in obs[7:10])
            tilt_penalty = 0.0
            if len(obs) > 10:
                tilt_penalty = -np.sum(np.abs(obs[7:10])) * 0.5

            # Encourage survival at every step
            alive_bonus = 5.0

            # You can also directly reward forward velocity from obs if you want tighter control
            forward_vel = np.clip(obs[22], 0, 5.0)
            forward_bonus = forward_vel * 1.0


            return base_reward + upright_bonus + tilt_penalty + alive_bonus + forward_bonus

        elif self.task_type == "standing":
            # Reward staying still and upright
            height = obs[0]
            x_velocity = obs[22] if len(obs) > 22 else 0
            
            balance_reward = 1.0 if height > 1.2 else 0.0
            stillness_reward = max(0, 1.0 - abs(x_velocity))
            return balance_reward + stillness_reward
            
        elif self.task_type == "navigation":
            if self.target_position is None:
                return base_reward
            
            current_pos = obs[1:3]  # x, y position
            distance = np.linalg.norm(current_pos - self.target_position)
            distance_reward = -distance * 0.1
            
            # Bonus for reaching target
            if distance < 0.5:
                distance_reward += 10.0
                
            return base_reward + distance_reward
        
        return base_reward
    
    def _check_task_termination(self, obs, info):
        """Check if task is complete"""
        if self.task_type == "navigation" and self.target_position is not None:
            current_pos = obs[1:3]
            distance = np.linalg.norm(current_pos - self.target_position)
            return distance < 0.3
        return False
    
    def _get_task_info(self, obs):
        """Get task-specific information"""
        info = {
            'task_type': self.task_type,
            'step': self.current_step,
            'height': obs[0],
        }
        
        if self.task_type == "navigation" and self.target_position is not None:
            current_pos = obs[1:3]
            info['distance_to_target'] = np.linalg.norm(current_pos - self.target_position)
            info['target_position'] = self.target_position.copy()
        
        return info
    
    def _set_random_target(self):
        """Set random target for navigation"""
        self.target_position = np.random.uniform(low=[-5, -5], high=[5, 5])
    
    # No need for explicit render/close; inherited from Wrapper


def make_humanoid_env(task_type="walking", render_mode=None):
    """Factory function to create humanoid environment"""
    return HumanoidEnv(task_type=task_type, render_mode=render_mode)