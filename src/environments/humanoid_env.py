"""
Fixed Humanoid environment wrapper for MuJoCo Humanoid-v4
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
        """Compute task-specific reward with improved design"""
        if self.task_type == "walking":
            # Height reward (gradual, not binary)
            height = obs[0]  # z-position
            target_height = 1.3  # Target standing height
            height_reward = -abs(height - target_height) * 0.5
            
            # Encourage upright posture (using quaternion if available)
            stability_reward = 0.0
            if len(obs) > 6:
                # Quaternion orientation is in obs[3:7], w component should be close to 1
                quat_w = obs[6]  # w component of quaternion
                stability_reward = (abs(quat_w) - 0.8) * 2.0 if abs(quat_w) > 0.8 else -1.0
            
            # Forward velocity reward (encourage movement)
            x_vel = obs[8] if len(obs) > 8 else 0  # x-velocity
            forward_reward = np.clip(x_vel, 0, 2.0) * 0.5
            
            # Small survival bonus (much smaller than before)
            alive_bonus = 0.1
            
            # Penalty for excessive angular velocity (encourage stability)
            angular_vel_penalty = 0.0
            if len(obs) > 11:
                angular_vel = np.sum(np.abs(obs[9:12]))  # Angular velocities
                angular_vel_penalty = -np.clip(angular_vel, 0, 5.0) * 0.1
            
            return (base_reward + height_reward + stability_reward + 
                   forward_reward + alive_bonus + angular_vel_penalty)

        elif self.task_type == "standing":
            # Focus on staying upright and still
            height = obs[0]
            target_height = 1.3
            height_reward = -abs(height - target_height) * 1.0
            
            # Penalize movement for standing task
            x_vel = obs[8] if len(obs) > 8 else 0
            y_vel = obs[9] if len(obs) > 9 else 0
            stillness_reward = -0.5 * (abs(x_vel) + abs(y_vel))
            
            # Encourage upright orientation
            stability_bonus = 0.0
            if len(obs) > 6:
                quat_w = obs[6]
                stability_bonus = max(0, abs(quat_w) - 0.9) * 5.0
            
            # Small survival bonus
            alive_bonus = 0.2
            
            return base_reward + height_reward + stillness_reward + stability_bonus + alive_bonus
            
        elif self.task_type == "navigation":
            if self.target_position is None:
                return base_reward
            
            # Height maintenance
            height = obs[0]
            height_reward = -abs(height - 1.3) * 0.3
            
            # Distance to target
            current_pos = obs[1:3]  # x, y position
            distance = np.linalg.norm(current_pos - self.target_position)
            distance_reward = -distance * 0.1
            
            # Bonus for reaching target
            if distance < 0.5:
                distance_reward += 5.0
            
            # Direction reward (encourage moving toward target)
            direction_to_target = self.target_position - current_pos
            direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-8)
            
            x_vel = obs[8] if len(obs) > 8 else 0
            y_vel = obs[9] if len(obs) > 9 else 0
            velocity_vec = np.array([x_vel, y_vel])
            
            direction_reward = np.dot(velocity_vec, direction_to_target) * 0.3
            
            # Small survival bonus
            alive_bonus = 0.1
            
            return (base_reward + height_reward + distance_reward + 
                   direction_reward + alive_bonus)
        
        return base_reward
    
    def _check_task_termination(self, obs, info):
        """Check if task is complete (less aggressive termination)"""
        height = obs[0]
        
        # Only terminate if really fallen (not just wobbling)
        if height < 0.7:  # More lenient than before
            return True
            
        # Task-specific termination
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