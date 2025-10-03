"""
Standing environment wrapper for MuJoCo Humanoid-v5
Optimized reward for indefinite standing balance

standing_env.py
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class StandingEnv(gym.Wrapper):
    """Wrapper for MuJoCo Humanoid environment optimized for standing task"""
    
    def __init__(self, render_mode: Optional[str] = None, config=None):
        env_id = "Humanoid-v5"
        print(f"Using {env_id} for standing task")
        
        env = gym.make(
            env_id, 
            render_mode=render_mode,
            exclude_current_positions_from_observation=True  # Default for Humanoid-v5; excludes x/y
        )
        super().__init__(env)
        
        # Use config for max_episode_steps if provided
        self.max_episode_steps = config.get('max_episode_steps', 5000) if config else 5000
        self.current_step = 0
        
        # Verify observation space (should be 348 for Humanoid-v5)
        obs_space = env.observation_space
        print(f"Observation space shape for standing: {obs_space.shape}")
    
    def reset(self, seed: Optional[int] = None): 
        observation, info = self.env.reset(seed=seed)
        self.current_step = 0
        
        # For standing reward tracking
        self.prev_height = self.env.unwrapped.data.qpos[2]  # Initial height
        
        return observation, info
    
    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Modify reward for standing
        reward = self._compute_task_reward(observation, base_reward, info, action)
        
        # Override termination for standing to allow indefinite episodes
        terminated = False  # Prevent early termination based on height
        truncated = self.current_step >= self.max_episode_steps
        
        # Add task info
        info.update(self._get_task_info())
        
        return observation, reward, terminated, truncated, info
    
    def _compute_task_reward(self, obs, base_reward, info, action):
        height = self.env.unwrapped.data.qpos[2]
        target_height = 1.3
        
        # Softer Gaussian height with tiered base for strong signal
        height_error = abs(height - target_height)
        height_reward = 50.0 * np.exp(-15 * height_error**2)  # Milder decay for broader basin
        
        # STRONG velocity penalty
        linear_vel = self.env.unwrapped.data.qvel[0:3]
        velocity_penalty = -3.0 * np.sum(np.square(linear_vel))  # Increased from -2.0
        
        # Control penalty
        ctrl_penalty = -0.02 * np.sum(np.square(action))
        
        # Survival bonus
        survival_bonus = 2.0
        
        # Uprightness
        quat_w = abs(self.env.unwrapped.data.qpos[3])
        upright_bonus = 5.0 * max(0, quat_w - 0.95)  # Stricter threshold (0.95 vs 0.9)
        
        # Tilt penalty
        tilt_penalty = -1.0 * (1.0 - quat_w)  # Stronger from -0.5
        
        # Stronger position penalty
        root_x, root_y = self.env.unwrapped.data.qpos[0:2]
        pos_penalty = -1.5 * (root_x**2 + root_y**2)  # Increased from -0.5
        
        # Angular velocity penalty
        angular_vel = self.env.unwrapped.data.qvel[3:6]
        ang_vel_penalty = -1.0 * np.sum(np.square(angular_vel))  # Increased from -0.5
        
        total_reward = (height_reward + velocity_penalty + ctrl_penalty + 
                        survival_bonus + upright_bonus + tilt_penalty + 
                        pos_penalty + ang_vel_penalty)
        
        # Debug every 200 steps
        if self.current_step % 200 == 0:
            dist = np.sqrt(root_x**2 + root_y**2)
            print(f"Step {self.current_step}: height={height:.3f} (error={height_error:.3f}), "
                  f"dist={dist:.3f}, quat_w={quat_w:.3f}, reward={total_reward:.2f} "
                  f"(h={height_reward:.1f}, pos={pos_penalty:.2f}, v={velocity_penalty:.2f}, ang_v={ang_vel_penalty:.2f})")
        
        return total_reward
    
    def _get_task_info(self):
        """Get task-specific information"""
        root_x, root_y = self.env.unwrapped.data.qpos[0:2]
        dist = np.sqrt(root_x**2 + root_y**2)
        return {
            'height': self.env.unwrapped.data.qpos[2],
            'distance_from_origin': dist,
            'x_position': root_x,
            'y_position': root_y,
            'x_velocity': self.env.unwrapped.data.qvel[0],
            'y_velocity': self.env.unwrapped.data.qvel[1],
            'z_velocity': self.env.unwrapped.data.qvel[2],
        }
    
    def get_observation_info(self):
        """Helper method to understand observation space"""
        obs, _ = self.env.reset()
        print("\nObservation Analysis:")
        print(f"Total observation size: {len(obs)}")
        print(f"Actual height (qpos[2]): {self.env.unwrapped.data.qpos[2]}")
        print(f"Root quaternion w (qpos[3]): {self.env.unwrapped.data.qpos[3]}")
        print(f"Linear vel x,y,z (qvel[0:3]): {self.env.unwrapped.data.qvel[0:3]}")
        return obs


def make_standing_env(render_mode=None, config=None):
    """Factory function to create standing environment"""
    return StandingEnv(render_mode=render_mode, config=config)


def test_environment():
    """Test the fixed environment"""
    print("Testing Fixed Humanoid Environment")
    print("=" * 40)
    
    # Test with random policy
    env = make_standing_env(render_mode=None)
    
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