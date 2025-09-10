"""
Fixed Humanoid environment wrapper for MuJoCo Humanoid-v5
Improved reward structure for better learning of walking, standing, and navigation tasks
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class HumanoidEnv(gym.Wrapper):
    """Fixed wrapper for MuJoCo Humanoid environment with improved reward design"""
    
    def __init__(self, task_type: str = "walking", render_mode: Optional[str] = None):
        env = gym.make("Humanoid-v5", render_mode=render_mode)
        super().__init__(env)
        self.task_type = task_type
        
        # Task parameters - INCREASED for longer episodes
        self.target_position = None
        self.max_episode_steps = 5000  # Much longer episodes for true standing
        self.current_step = 0
        
        # Get observation info to verify indices
        obs_space = env.observation_space
        print(f"Observation space shape: {obs_space.shape}")
    
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
        if self.task_type == "standing":
            height = obs[2]  # Corrected: z-position is height
            target_height = 1.25  # Adjusted to typical initial standing height; verify with reset
            
            # Softer quadratic penalty for height error
            height_error = abs(height - target_height)
            if height_error < 0.05:
                height_reward = 15.0  # Strong reward for precise height
            elif height_error < 0.1:
                height_reward = 5.0
            else:
                height_reward = - (height_error ** 2) * 10.0  # Quadratic for smoother gradients
            
            # Penalize movement more heavily for true standing
            movement_penalty = 0.0
            velocities = obs[24:27]  # Corrected: torso linear velocities (x/y/z)
            movement_penalty = -np.sum(np.abs(velocities)) * 0.5
            
            # Add penalty for x/y drift (promote centered standing)
            xy_drift = np.sum(np.abs(obs[0:2]))  # Root x/y positions
            drift_penalty = -xy_drift * 0.1
            
            # Strong survival bonus that increases over time
            survival_reward = 2.0 if height > 1.0 else -10.0
            
            # Time-based bonus for longer standing
            time_bonus = min(self.current_step * 0.01, 5.0)  # Up to +5 reward for long episodes
            
            total_reward = height_reward + movement_penalty + drift_penalty + survival_reward + time_bonus
            return total_reward
            
        return base_reward
    
    def _check_task_termination(self, obs, info):
        """Check if episode should terminate based on task"""
        height = obs[2]  # Corrected
        
        if self.task_type == "standing":
            # Only terminate if truly fallen (very lenient)
            if height < 1.0:  # Stricter than before for better learning
                return True
        else:
            # More aggressive termination for other tasks
            if height < 0.9:
                return True
        
        return False
    
    def _get_task_info(self, obs):
        """Get task-specific information"""
        info = {
            'task_type': self.task_type,
            'step': self.current_step,
            'height': obs[2],  # Corrected
            'x_position': obs[0],  # Corrected
            'y_position': obs[1],  # Corrected
        }
        
        # Add velocities if available
        info['x_velocity'] = obs[24]
        info['y_velocity'] = obs[25]
        info['z_velocity'] = obs[26]
        
        if self.task_type == "navigation" and self.target_position is not None:
            current_pos = obs[0:2]  # Corrected to x/y
            info['distance_to_target'] = np.linalg.norm(current_pos - self.target_position)
            info['target_position'] = self.target_position.copy()
        
        return info
    
    def _set_random_target(self):
        """Set random target for navigation"""
        self.target_position = np.random.uniform(low=[-3, -3], high=[3, 3])
    
    def get_observation_info(self):
        """Helper method to understand observation space"""
        obs, _ = self.env.reset()
        print("\nObservation Analysis:")
        print(f"Total observation size: {len(obs)}")
        print(f"obs[0:3] (x,y,z/height pos): {obs[0:3]}")
        print(f"obs[3:7] (root quaternion): {obs[3:7]}")
        print(f"obs[24:27] (linear vel x/y/z): {obs[24:27]}")
        print(f"obs[27:30] (angular vel): {obs[27:30]}")
        print("...")
        return obs


def make_humanoid_env(task_type="walking", render_mode=None):
    """Factory function to create humanoid environment"""
    return HumanoidEnv(task_type=task_type, render_mode=render_mode)


def test_environment():
    """Test the fixed environment"""
    print("Testing Fixed Humanoid Environment")
    print("=" * 40)
    
    # Test with random policy
    env = make_humanoid_env(task_type="standing", render_mode=None)
    
    # Show observation info
    env.get_observation_info()
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(200):  # Test longer episodes
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 50 == 0:
            print(f"Step {step}: height={info['height']:.3f}, "
                  f"reward={reward:.3f}, total_reward={total_reward:.2f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step} ({'terminated' if terminated else 'truncated'})")
            break
    
    env.close()
    print(f"Final total reward: {total_reward:.2f}")


if __name__ == "__main__":
    test_environment()