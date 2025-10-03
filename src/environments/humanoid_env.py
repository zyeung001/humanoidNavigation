"""
Fixed Humanoid environment wrapper for MuJoCo Humanoid-v5 and Standup-v5
Improved reward structure for better learning of walking, standing, and navigation tasks

humanoid_env.py
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class HumanoidEnv(gym.Wrapper):
    """Fixed wrapper for MuJoCo Humanoid environment with improved reward design"""
    
    def __init__(self, task_type: str = "walking", render_mode: Optional[str] = None):

        env_id = "Humanoid-v5"

        print(f"Using {env_id} for {task_type} task")
        
        env = gym.make(
            env_id, 
            render_mode=render_mode,
            exclude_current_positions_from_observation=True  # Default for Humanoid-v5; excludes x/y
        )
        super().__init__(env)
        self.task_type = task_type
        
        # Task parameters
        self.target_position = None
        self.max_episode_steps = 1000  # Matches Standup default for fair comparison
        self.current_step = 0
        
        # Verify observation space (should be 348 for Humanoid-v5)
        obs_space = env.observation_space
        print(f"Observation space shape for {task_type}: {obs_space.shape}")
    
    def reset(self, seed: Optional[int] = None): 
        observation, info = self.env.reset(seed=seed)
        self.current_step = 0
        
        if self.task_type == "navigation":
            self._set_random_target()
            
        # For standing reward tracking
        if self.task_type == "standing":
            self.prev_height = self.env.unwrapped.data.qpos[2]  # Initial height for upward vel

        
        return observation, info
    
    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Modify reward based on task (pass action for ctrl_cost)
        reward = self._compute_task_reward(observation, base_reward, info, action)
        
        # Check task-specific termination (removed for standing to allow longer episodes)
        task_terminated = self._check_task_termination(observation, info)
        terminated = terminated or task_terminated
        
        # Episode length limit
        truncated = truncated or (self.current_step >= self.max_episode_steps)
        
        # Add task info with raw values
        info.update(self._get_task_info(observation))
        
        return observation, reward, terminated, truncated, info
    
    def _compute_task_reward(self, obs, base_reward, info, action):
        if self.task_type == "standing":
            height = self.env.unwrapped.data.qpos[2]
            target_height = 1.3
            
            # Softer height reward: Gaussian for smooth gradients (peaks at 50 at target, decays fast)
            height_error = abs(height - target_height)
            height_reward = 50.0 * np.exp(-10 * height_error**2)  # e^{-10*e^2}: full at 0 error, ~0 at 0.2m
            
            # STRONG penalty for movement (stay still!)
            linear_vel = self.env.unwrapped.data.qvel[0:3]
            velocity_penalty = -2.0 * np.sum(np.square(linear_vel))
            
            # Control penalty (smooth movements)
            ctrl_penalty = -0.02 * np.sum(np.square(action))
            
            # Reduce survival bonus (make falling hurt more)
            survival_bonus = 2.0
            
            # Uprightness (must be very upright) + tilt penalty
            quat_w = abs(self.env.unwrapped.data.qpos[3])  # Abs for robustness
            upright_bonus = 5.0 * max(0, quat_w - 0.9)
            tilt_penalty = -1.0 * (1.0 - quat_w)  # Extra cost for any tilt
            
            # NEW: Position penalty - prevent drift from origin
            root_x, root_y = self.env.unwrapped.data.qpos[0:2]
            pos_penalty = -1.0 * (root_x**2 + root_y**2)  # Quadratic cost on distance
            
            # NO upward velocity or height delta bonuses - pure stability
            
            total_reward = (height_reward + velocity_penalty + ctrl_penalty + 
                        survival_bonus + upright_bonus + tilt_penalty + pos_penalty)
            
            # Debug every 200 steps (as in your old code)
            if self.current_step % 200 == 0:
                print(f"Step {self.current_step}: height={height:.3f} (target={target_height}), "
                    f"reward={total_reward:.2f} (h={height_reward:.1f}, pos={pos_penalty:.2f}, v={velocity_penalty:.2f})")
            
            return total_reward
        
        return base_reward  # Fallback for other tasks
    
    def _check_task_termination(self, obs, info):
        """Check if episode should terminate based on task"""
        if self.task_type == "standing":
            # Removed custom termination for standing - rely on env's (height <1.0)
            # This allows longer episodes to learn balance
            return False
        
        return False  # For other tasks, add if needed
    
    def _get_task_info(self, obs):
        """Get task-specific information using mjData"""
        height = self.env.unwrapped.data.qpos[2]
        info = {
            'task_type': self.task_type,
            'step': self.current_step,
            'height': height,
            'x_position': self.env.unwrapped.data.qpos[0],
            'y_position': self.env.unwrapped.data.qpos[1],
            'x_velocity': self.env.unwrapped.data.qvel[0],
            'y_velocity': self.env.unwrapped.data.qvel[1],
            'z_velocity': self.env.unwrapped.data.qvel[2],
        }
        return info
        
    def _set_random_target(self):
        """Set random target for navigation"""
        self.target_position = np.random.uniform(low=[-3, -3], high=[3, 3])
    
    def get_observation_info(self):
        """Helper method to understand observation space"""
        obs, _ = self.env.reset()
        print("\nObservation Analysis:")
        print(f"Total observation size: {len(obs)}")
        print(f"Actual height (qpos[2]): {self.env.unwrapped.data.qpos[2]}")
        print(f"Root quaternion w (qpos[3]): {self.env.unwrapped.data.qpos[3]}")
        print(f"Linear vel x,y,z (qvel[0:3]): {self.env.unwrapped.data.qvel[0:3]}")
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