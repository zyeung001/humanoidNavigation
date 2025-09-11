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
        # Critical fix: Include x and y positions in observations for consistent indexing
        env = gym.make(
            "Humanoid-v5", 
            render_mode=render_mode,
            exclude_current_positions_from_observation=False  # Include x,y for obs[0:2]=x,y, obs[2]=z
        )
        super().__init__(env)
        self.task_type = task_type
        
        # Task parameters
        self.target_position = None
        self.max_episode_steps = 1000  # Reduced for faster learning cycles
        self.current_step = 0
        
        # Verify observation space
        obs_space = env.observation_space
        print(f"Observation space shape: {obs_space.shape}")  # Should be (378,) now with x,y included
    
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
        
        # Add task info with raw values
        info.update(self._get_task_info(observation))
        
        return observation, reward, terminated, truncated, info
    
    def _compute_task_reward(self, obs, base_reward, info):
        """Compute task-specific reward"""
        if self.task_type == "standing":
            height = obs[2]  # z-position
            target_height = 1.4  # Match default initial height from env docs
            
            # Height reward: Stronger positive for precision, linear penalty
            height_error = abs(height - target_height)
            if height_error < 0.05:
                height_reward = 30.0  # Boosted for near-perfect
            elif height_error < 0.1:
                height_reward = 15.0
            else:
                height_reward = -height_error * 3.0  # Milder to avoid over-penalizing
            
            # Zero out forward reward to focus on standing (override base if needed)
            forward_reward = 0.0  # Explicitly disable movement incentive
            
            # Stronger movement penalty
            velocities = obs[24:27]  # Linear vel x,y,z
            movement_penalty = -np.sum(np.abs(velocities)) * 0.5  # Increased weight
            
            # Drift penalty
            xy_drift = np.linalg.norm(obs[0:2])
            drift_penalty = -xy_drift * 0.1
            
            # Survival reward scaled by height quality
            survival_reward = 10.0 * (height / target_height) if height > 1.0 else -10.0
            
            # Time bonus ramps slower for long-term incentive
            time_bonus = min(self.current_step * 0.005, 20.0)
            
            # Enhanced upright bonus + penalize angular velocity
            quat = obs[3:7]
            upright_bonus = 5.0 * quat[0]  # Boosted
            angular_vel = obs[27:30]  # Angular vel x,y,z
            angular_penalty = -np.sum(np.abs(angular_vel)) * 0.2  # New: Reduce wobbling
            
            # Control cost to smooth actions (add if not in base)
            ctrl_cost = -0.01 * np.sum(np.square(info.get('ctrl', 0.0)))  # Mild, from base if available
            
            total_reward = (
                height_reward + forward_reward + movement_penalty + drift_penalty +
                survival_reward + time_bonus + upright_bonus + angular_penalty + ctrl_cost
            )
            return total_reward
        
        return base_reward
    
    def _check_task_termination(self, obs, info):
        """Check if episode should terminate based on task"""
        height = obs[2]
        
        if self.task_type == "standing":
            # Lenient termination: only if truly fallen (allows recovery learning)
            if height < 0.9:  # Lowered threshold to permit more exploration
                return True
        else:
            if height < 0.9:
                return True
        
        return False
    
    def _get_task_info(self, obs):
        """Get task-specific information (using raw obs values)"""
        info = {
            'task_type': self.task_type,
            'step': self.current_step,
            'height': obs[2],
            'x_position': obs[0],
            'y_position': obs[1],
        }
        
        # Add velocities
        info['x_velocity'] = obs[24]
        info['y_velocity'] = obs[25]
        info['z_velocity'] = obs[26]
        
        if self.task_type == "navigation" and self.target_position is not None:
            current_pos = obs[0:2]
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
        print(f"obs[3:7] (root quaternion w,x,y,z): {obs[3:7]}")
        print(f"obs[24:27] (linear vel x,y,z): {obs[24:27]}")
        print(f"obs[27:30] (angular vel x,y,z): {obs[27:30]}")
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
    
    for step in range(200):
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