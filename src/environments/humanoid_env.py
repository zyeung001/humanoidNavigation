"""
Fixed Humanoid environment wrapper for MuJoCo Humanoid-v5 and Standup-v5
Improved reward structure for better learning of walking, standing, and navigation tasks
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class HumanoidEnv(gym.Wrapper):
    """Fixed wrapper for MuJoCo Humanoid environment with improved reward design"""
    
    def __init__(self, task_type: str = "walking", render_mode: Optional[str] = None):
        # Use Standup for standing task; regular Humanoid for others
        if task_type == "standing":
            env_id = "HumanoidStandup-v5"
            print(f"Using HumanoidStandup-v5 for standing task")
        else:
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
            # CRITICAL DEBUG: Print first 20 values every 100 steps
            if self.current_step % 100 == 0:
                print(f"\n=== STEP {self.current_step} DEBUG ===")
                print(f"obs.shape: {obs.shape}")
                print(f"First 20 values:")
                for i in range(min(20, len(obs))):
                    print(f"  obs[{i:2d}]: {obs[i]:8.4f}")
                print("Looking for height values (should be ~0.8-1.5)...")
            
            # Don't use abs() - we need to see negative values
            height_candidates = [0, 1, 2, 3, 4, 5]  # Check more indices
            height = obs[0]  # Default
            selected_idx = 0
            
            # Find reasonable height without abs()
            for idx in height_candidates:
                if idx < len(obs) and 0.7 <= obs[idx] <= 2.0:  # Positive values only
                    height = obs[idx]
                    selected_idx = idx
                    break
            
            if self.current_step % 100 == 0:
                print(f"Selected height from obs[{selected_idx}]: {height:.4f}")
            
            target_height = 1.3
            
            # Rest of reward calculation...
            height_ratio = min(height / target_height, 1.0)  
            height_reward = 50.0 * height_ratio  
            
            # Simplified reward for debugging
            velocity_penalty = -1.0 * np.sum(np.abs(obs[-10:-7])) if len(obs) > 10 else 0
            ctrl_penalty = -0.1 * np.sum(np.square(action))
            survival_bonus = 10.0 

            total_reward = height_reward + velocity_penalty + ctrl_penalty + survival_bonus
            
            if self.current_step % 100 == 0:
                print(f"Rewards: height={height_reward:.2f}, vel={velocity_penalty:.2f}, ctrl={ctrl_penalty:.2f}, total={total_reward:.2f}")
                print("=== END DEBUG ===\n")
            
            return total_reward
        
        return base_reward
    
    def _check_task_termination(self, obs, info):
        """Check if episode should terminate based on task"""
        if self.task_type == "standing":
            # Use same height detection logic as reward function
            height_candidates = [0, 1, 2]
            height = 0.0
            
            for idx in height_candidates:
                if idx < len(obs) and 0.5 <= abs(obs[idx]) <= 2.0:
                    height = abs(obs[idx])
                    break
            
            # More lenient termination - only terminate if really fallen
            if height < 0.5:  # Very low threshold
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
                'height': obs[height_idx],
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
