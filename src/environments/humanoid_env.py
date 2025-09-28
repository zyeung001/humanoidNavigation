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
        # Use Standup for standing task; regular Humanoid for others
        env_id = "Humanoid-v5"
        print(f"Creating Humanoid-v5 for {task_type} task")
        
        env = gym.make(
            env_id, 
            render_mode=render_mode,
            exclude_current_positions_from_observation=False  # Include x,y for consistency in non-standup
        )
        super().__init__(env)
        self.task_type = task_type
        
        # Task parameters
        self.target_position = None
        self.max_episode_steps = 1000  # Matches Standup default for fair comparison
        self.current_step = 0
        
        # Verify observation space (348 for Standup, 378 for Humanoid with x/y)
        obs_space = env.observation_space
        print(f"Observation space shape for {task_type}: {obs_space.shape}")
    
    def reset(self, seed: Optional[int] = None): 
        observation, info = self.env.reset(seed=seed)
        self.current_step = 0
        
        if self.task_type == "navigation":
            self._set_random_target()
            
        # For standing reward tracking
        if self.task_type == "standing":
            self.prev_height = observation[0]  # Initial height for upward vel
        
        return observation, info
    
    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Modify reward based on task (pass action for ctrl_cost)
        reward = self._compute_task_reward(observation, base_reward, info, action)
        
        # Check task-specific termination (lenient for standing)
        task_terminated = self._check_task_termination(observation, info)
        terminated = terminated or task_terminated
        
        # Episode length limit
        truncated = truncated or (self.current_step >= self.max_episode_steps)
        
        # Add task info with raw values
        info.update(self._get_task_info(observation))
        
        return observation, reward, terminated, truncated, info
    
    def _compute_task_reward(self, obs, base_reward, info, action):
        if self.task_type == "standing":
            # For HumanoidStandup-v5, obs[0] is the z-coordinate
            # The initial spawn height is around 1.05m, obs[0] starts near 0
            height = 1.05 + obs[0]  # More accurate base height
            target_height = 1.3
            
            # Height reward (main component)
            height_error = abs(height - target_height)
            if height_error < 0.1:
                height_reward = 50.0
            else:
                height_reward = max(0, 50.0 * (1.0 - height_error / 0.5))
            
            # Stability rewards (keep the humanoid stable)
            # Use center-of-mass velocities (indices 22-24 for HumanoidStandup-v5)
            if len(obs) >= 25:
                linear_vel = obs[22:25]  # x, y, z velocities
                velocity_penalty = -0.1 * np.sum(np.square(linear_vel))
            else:
                velocity_penalty = 0
            
            # Control effort
            ctrl_penalty = -0.01 * np.sum(np.square(action))
            
            # Survival bonus
            survival_bonus = 5.0
            
            # Uprightness reward (quaternion w-component at index 1)
            if len(obs) > 1:
                quat_w = abs(obs[1])  # w-component of root quaternion
                upright_bonus = 5.0 * quat_w  # Reward being upright
            else:
                upright_bonus = 0
            
            total_reward = height_reward + velocity_penalty + ctrl_penalty + survival_bonus + upright_bonus
            
            # Debug less frequently
            if self.current_step % 200 == 0:
                print(f"Step {self.current_step}: height={height:.3f} (target={target_height}), "
                    f"reward={total_reward:.2f} (h={height_reward:.1f}, v={velocity_penalty:.1f})")
            
            return total_reward
        
        return base_reward
    
    def _check_task_termination(self, obs, info):
        """Check if episode should terminate based on task"""
        if self.task_type == "standing":
            height = 1.05 + obs[0]
            
            # Only terminate if completely fallen (very low threshold)
            if height < 0.2:  # Much lower threshold
                return True
            
            # Also check if humanoid is completely upside down
            if len(obs) > 1:
                quat_w = abs(obs[1])  # w-component of quaternion
                if quat_w < 0.1:  # Completely inverted
                    return True
        
        return False
    
    def _get_task_info(self, obs):
        """Get task-specific information (using raw obs values)"""
        if self.task_type == "standing":
            # FIXED: Correct indices for HumanoidStandup-v5
            height_idx = 0       # Height at index 0 for HumanoidStandup-v5
            vel_start = 22       # Linear velocities start at index 22
            
            info = {
                'task_type': self.task_type,
                'step': self.current_step,
                'height': 1.05 + obs[height_idx],
                'x_position': 0.0,   # Not available in HumanoidStandup-v5
                'y_position': 0.0,   # Not available in HumanoidStandup-v5
            }
            
            # Add velocities with bounds checking
            if len(obs) > vel_start + 2:
                info['x_velocity'] = obs[vel_start]
                info['y_velocity'] = obs[vel_start + 1]
                info['z_velocity'] = obs[vel_start + 2]
            else:
                info['x_velocity'] = 0.0
                info['y_velocity'] = 0.0
                info['z_velocity'] = 0.0
                
        else:
            # For regular Humanoid-v5
            height_idx = 2
            x_idx = 0
            y_idx = 1
            vel_start = 24
            
            info = {
                'task_type': self.task_type,
                'step': self.current_step,
                'height': obs[height_idx],
                'x_position': obs[x_idx],
                'y_position': obs[y_idx],
            }
            
            if len(obs) > vel_start + 2:
                info['x_velocity'] = obs[vel_start]
                info['y_velocity'] = obs[vel_start + 1]
                info['z_velocity'] = obs[vel_start + 2]
            else:
                info['x_velocity'] = 0.0
                info['y_velocity'] = 0.0
                info['z_velocity'] = 0.0
            
        return info
        
    def _set_random_target(self):
        """Set random target for navigation"""
        self.target_position = np.random.uniform(low=[-3, -3], high=[3, 3])
    
    def get_observation_info(self):
        """Helper method to understand observation space"""
        obs, _ = self.env.reset()
        print("\nObservation Analysis:")
        print(f"Total observation size: {len(obs)}")
        height_idx = 0 if self.task_type == "standing" else 2
        quat_start = 1 if self.task_type == "standing" else 3
        vel_start = 22 if self.task_type == "standing" else 24
        ang_start = 25 if self.task_type == "standing" else 27
        print(f"obs[{height_idx}] (z/height pos): {obs[height_idx]}")
        print(f"obs[{quat_start}:{quat_start+4}] (root quaternion w,x,y,z): {obs[quat_start:quat_start+4]}")
        print(f"obs[{vel_start}:{vel_start+3}] (linear vel x,y,z): {obs[vel_start:vel_start+3]}")
        print(f"obs[{ang_start}:{ang_start+3}] (angular vel x,y,z): {obs[ang_start:ang_start+3]}")
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
