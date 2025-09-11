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
        else:
            env_id = "Humanoid-v5"
        
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
            
        return observation, info
    
    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Modify reward based on task
        reward = self._compute_task_reward(observation, base_reward, info)
        
        # Check task-specific termination (lenient for standing)
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
            # For Standup env: obs[0] is z-height (no x/y)
            height = obs[0]  # Adjusted for Standup obs
            target_height = 1.4  # Matches standing torso z
            
            # Blend with Standup defaults: upward vel + alive - costs
            # Upward cost (from Standup): encourage rising
            upward_vel = (height - self.prev_height if hasattr(self, 'prev_height') else 0) / self.env.unwrapped.dt
            self.prev_height = height  # Track for next step
            uph_reward = upward_vel  # Default w=1
            
            # Height bonus for precision and stability
            height_error = abs(height - target_height)
            if height_error < 0.05:
                height_reward = 50.0  # Strong for perfect stand
            elif height_error < 0.1:
                height_reward = 25.0
            else:
                height_reward = -height_error * 2.0  # Mild penalty
            
            # Alive bonus (constant for survival)
            alive_bonus = 1.0
            
            # Penalties (boosted for stability)
            ctrl_cost = -0.1 * np.sum(np.square(action))  # Default from Standup
            impact_cost = -5e-7 * np.sum(np.square(info.get('cfrc_ext', np.zeros(1))))  # External forces
            velocities = obs[22:25]  # Linear vel x,y,z (adjusted index for Standup)
            movement_penalty = -np.sum(np.abs(velocities)) * 0.5  # Heavier for stillness
            angular_vel = obs[25:28]  # Angular vel
            angular_penalty = -np.sum(np.abs(angular_vel)) * 0.3  # Reduce wobble
            
            # Upright bonus
            quat = obs[1:5]  # Quaternion w,x,y,z
            upright_bonus = 5.0 * quat[0]  # w~1 when upright
            
            # Time bonus for long standing
            time_bonus = min(self.current_step * 0.01, 30.0)
            
            total_reward = (
                uph_reward + height_reward + alive_bonus + ctrl_cost + 
                impact_cost + movement_penalty + angular_penalty + upright_bonus + time_bonus
            )
            return total_reward
            
        # For other tasks, use base reward with possible adjustments
        return base_reward
    
    def _check_task_termination(self, obs, info):
        """Check if episode should terminate based on task"""
        height = obs[0] if self.task_type == "standing" else obs[2]  # Adjust for Standup
        
        if self.task_type == "standing":
            # No termination for falls (inherit from Standup)
            return False
        else:
            if height < 0.9:
                return True
        
        return False
    
    def _get_task_info(self, obs):
        """Get task-specific information (using raw obs values)"""
        height_idx = 0 if self.task_type == "standing" else 2
        x_idx = -1 if self.task_type == "standing" else 0  # No x/y in Standup
        y_idx = -1 if self.task_type == "standing" else 1
        vel_start = 22 if self.task_type == "standing" else 24  # Adjust vel indices
        
        info = {
            'task_type': self.task_type,
            'step': self.current_step,
            'height': obs[height_idx],
            'x_position': obs[x_idx] if x_idx >= 0 else 0.0,
            'y_position': obs[y_idx] if y_idx >= 0 else 0.0,
        }
        
        # Add velocities
        info['x_velocity'] = obs[vel_start]
        info['y_velocity'] = obs[vel_start + 1]
        info['z_velocity'] = obs[vel_start + 2]
        
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