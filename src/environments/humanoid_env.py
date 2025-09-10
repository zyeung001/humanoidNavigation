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
        
        # Task parameters
        self.target_position = None
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # Get observation info to verify indices
        obs_space = env.observation_space
        print(f"Observation space shape: {obs_space.shape}")
        # For Humanoid-v5: obs[0] = z-position (height), obs[1:3] = x,y position
        # obs[3:7] = quaternion orientation, obs[8:] = velocities and joint info
    
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
        
        # Check task-specific termination (less aggressive)
        task_terminated = self._check_task_termination(observation, info)
        terminated = terminated or task_terminated
        
        # Episode length limit
        truncated = truncated or (self.current_step >= self.max_episode_steps)
        
        # Add task info
        info.update(self._get_task_info(observation))
        
        return observation, reward, terminated, truncated, info
    
    def _compute_task_reward(self, obs, base_reward, info):
        if self.task_type == "standing":
            # Only focus on height maintenance
            height = obs[0]  # Verify this is correct
            target_height = 1.3
            
            # Simple distance-based reward
            height_error = abs(height - target_height)
            
            # Strong reward for being near target height
            if height_error < 0.1:
                height_reward = 10.0  # Strong positive reward
            elif height_error < 0.2:
                height_reward = 5.0
            else:
                height_reward = -height_error * 10.0  # Strong penalty
            
            # Small penalties for movement (if velocity data available)
            movement_penalty = 0.0
            if len(obs) > 8:
                velocities = obs[8:11] if len(obs) > 11 else obs[8:10]
                movement_penalty = -np.sum(np.abs(velocities)) * 0.1
            
            # Base survival reward only if upright
            survival_reward = 1.0 if height > 1.0 else -5.0
            
            total_reward = height_reward + movement_penalty + survival_reward
            
            return total_reward
        
        return base_reward
    
    def _check_task_termination(self, obs, info):
        """More aggressive termination for faster learning"""
        height = obs[0]
        
        # Terminate if clearly fallen
        if height < 0.9:  # More aggressive than 0.7
            return True
            
        return False
    
    def _get_task_info(self, obs):
        """Get task-specific information"""
        info = {
            'task_type': self.task_type,
            'step': self.current_step,
            'height': obs[0],
            'x_position': obs[1] if len(obs) > 1 else 0,
            'y_position': obs[2] if len(obs) > 2 else 0,
        }
        
        # Add velocities if available
        if len(obs) > 8:
            info['x_velocity'] = obs[8]
            info['y_velocity'] = obs[9] if len(obs) > 9 else 0
        
        if self.task_type == "navigation" and self.target_position is not None:
            current_pos = obs[1:3]
            info['distance_to_target'] = np.linalg.norm(current_pos - self.target_position)
            info['target_position'] = self.target_position.copy()
        
        return info
    
    def _set_random_target(self):
        """Set random target for navigation"""
        # More reasonable target range
        self.target_position = np.random.uniform(low=[-3, -3], high=[3, 3])
    
    def get_observation_info(self):
        """Helper method to understand observation space"""
        obs, _ = self.env.reset()
        print("\nObservation Analysis:")
        print(f"Total observation size: {len(obs)}")
        print(f"obs[0] (height): {obs[0]:.3f}")
        print(f"obs[1:3] (x,y pos): {obs[1:3]}")
        if len(obs) > 6:
            print(f"obs[3:7] (quaternion): {obs[3:7]}")
        if len(obs) > 8:
            print(f"obs[8:11] (linear vel): {obs[8:11]}")
        if len(obs) > 11:
            print(f"obs[11:14] (angular vel): {obs[11:14]}")
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
    env = make_humanoid_env(task_type="walking", render_mode=None)
    
    # Show observation info
    env.get_observation_info()
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"Step {step}: height={info['height']:.3f}, "
                  f"reward={reward:.3f}, total_reward={total_reward:.2f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    env.close()
    print(f"Final total reward: {total_reward:.2f}")


if __name__ == "__main__":
    test_environment()