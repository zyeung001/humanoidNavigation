"""
Standing environment wrapper for MuJoCo Humanoid-v5
Optimized reward for indefinite standing balance

standing_env.py
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from mujoco import mj_name2id, mjtObj

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
        self.base_target_height = 1.3
        self.curriculum_progress = 0.0

        # In reset():
        # Gradually increase target height from 1.2 to 1.3
        self.target_height = 1.2 + 0.1 * min(1.0, self.curriculum_progress)
        self.curriculum_progress += 1.0 / 100000


        self.max_episode_steps = config.get('max_episode_steps', 5000) if config else 5000
        self.current_step = 0

        self.domain_rand = config.get('domain_rand')
        self.rand_mass_range = config.get('rand_mass_range')
        self.rand_friction_range = config.get('rand_friction_range')
        
        # Verify observation space (should be 348 for Humanoid-v5)
        obs_space = env.observation_space
        print(f"Observation space shape for standing: {obs_space.shape}")
    
    def reset(self, seed: Optional[int] = None): 
        observation, info = self.env.reset(seed=seed)
        self.current_step = 0
        self.prev_height = self.env.unwrapped.data.qpos[2]
        
        if self.domain_rand:
            # Randomize body masses
            original_masses = self.env.unwrapped.model.body_mass.copy()  # Backup if needed
            self.env.unwrapped.model.body_mass *= np.random.uniform(
                self.rand_mass_range[0], self.rand_mass_range[1],
                size=self.env.unwrapped.model.body_mass.shape
            )
            
            # Randomize geom friction (for feet/contact surfaces)
            original_friction = self.env.unwrapped.model.geom_friction.copy()
            self.env.unwrapped.model.geom_friction[:, 0] *= np.random.uniform(  # Lateral friction
                self.rand_friction_range[0], self.rand_friction_range[1],
                size=self.env.unwrapped.model.geom_friction.shape[0]
            )
        
        return observation, info
    
    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Modify reward for standing
        reward, terminated = self._compute_task_reward(observation, base_reward, info, action)
        
        # Override termination for standing to allow indefinite episodes
        terminated = terminated  # Prevent early termination based on height
        truncated = self.current_step >= self.max_episode_steps
        
        # Add task info 
        info.update(self._get_task_info())
        
        return observation, reward, terminated, truncated, info
    
    def _compute_task_reward(self, obs, base_reward, info, action):
        """Enhanced reward function for more stable standing convergence"""
        
        # === State extraction ===
        height = self.env.unwrapped.data.qpos[2]
        root_x, root_y = self.env.unwrapped.data.qpos[0:2]
        quat_w = self.env.unwrapped.data.qpos[3]
        
        linear_vel = self.env.unwrapped.data.qvel[0:3]
        angular_vel = self.env.unwrapped.data.qvel[3:6]

        # Get torso and feet positions
        torso_id = mj_name2id(self.env.unwrapped.model, mjtObj.mjOBJ_BODY, 'torso')
        com_pos = self.env.unwrapped.data.subtree_com[torso_id]

        left_foot_id = mj_name2id(self.env.unwrapped.model, mjtObj.mjOBJ_BODY, 'left_foot')
        right_foot_id = mj_name2id(self.env.unwrapped.model, mjtObj.mjOBJ_BODY, 'right_foot')
        left_foot_pos = self.env.unwrapped.data.xpos[left_foot_id]
        right_foot_pos = self.env.unwrapped.data.xpos[right_foot_id]
        
        # Support polygon center
        support_center = (left_foot_pos[:2] + right_foot_pos[:2]) / 2
        com_error = np.linalg.norm(com_pos[:2] - support_center)
        
        # === PRIMARY: Height Reward (MORE AGGRESSIVE) ===
        target_height = 1.3
        height_error = abs(height - target_height)
        
        # Stronger exponential with higher base reward
        height_error = abs(height - target_height)

        # Even stronger gradient toward target
        if height_error < 0.02:  # Within 2cm
            height_reward = 700.0  # Massive reward
        elif height_error < 0.05:  # Within 5cm
            height_reward = 400.0 + 200.0 * (0.05 - height_error) / 0.03
        elif height_error < 0.10:  # Within 10cm
            height_reward = 200.0 + 200.0 * (0.10 - height_error) / 0.05
        else:
            # Exponential decay for larger errors
            height_reward = 200.0 * np.exp(-20.0 * (height_error - 0.10)**2)

        # Additional penalty for being too short (your robot's current problem)
        if height < target_height - 0.05:  # More than 5cm too short
            height_reward *= 0.5  # Cut reward in half
        
        # Stronger penalty for being too tall
        if height > target_height + 0.15:
            height_reward *= 0.3  # Harsh penalty
        
        # === CRITICAL: Height velocity penalty (prevent oscillations) ===
        height_vel = self.env.unwrapped.data.qvel[2]
        height_vel_penalty = -50.0 * abs(height_vel) if abs(height_vel) > 0.1 else 0
        
        # === Uprightness (stronger) ===
        upright_reward = 50.0 * (abs(quat_w) ** 3)  # Cubic for stronger gradient near 1.0
        
        # === Position stability (softer early, stricter later) ===
        position_error = np.sqrt(root_x**2 + root_y**2)
        # Progressive penalty based on training progress
        position_penalty = -5.0 * min(position_error**2, 2.0)
        
        # === Velocity penalties (reduced to not interfere with height corrections) ===
        linear_vel_penalty = -0.5 * np.clip(np.sum(np.square(linear_vel[:2])), 0, 3)  # Only x,y
        angular_vel_penalty = -0.5 * np.clip(np.sum(np.square(angular_vel)), 0, 3)
        
        # === COM balance (critical for stability) ===
        com_penalty = -30.0 * min(com_error**2, 1.0)
        
        # === Control (very light to allow corrections) ===
        control_penalty = -0.005 * np.sum(np.square(action))
        
        # === Survival bonus (height-dependent) ===
        if 1.25 < height < 1.35:  # Near target zone
            survival_bonus = 20.0
        elif 1.20 < height < 1.40:  # Acceptable zone
            survival_bonus = 10.0
        else:
            survival_bonus = 2.0
        
        # === Perfect standing bonus ===
        if (1.29 < height < 1.31 and  # Very close to target
            abs(quat_w) > 0.98 and  # Very upright
            position_error < 0.1 and  # Not drifting
            com_error < 0.05):  # Well balanced
            perfect_bonus = 100.0
        else:
            perfect_bonus = 0.0
        
        # === Foot contact bonus (both feet on ground) ===
        left_contact = left_foot_pos[2] < 0.05
        right_contact = right_foot_pos[2] < 0.05
        contact_bonus = 10.0 if (left_contact and right_contact) else -20.0
        
        # === Total Reward ===
        total_reward = (
            height_reward +
            height_vel_penalty +
            upright_reward +
            position_penalty +
            linear_vel_penalty +
            angular_vel_penalty +
            com_penalty +
            control_penalty +
            survival_bonus +
            perfect_bonus +
            contact_bonus
        )
        
        # === Termination with grace period ===
        terminate = False
        if height < 0.7:  # Lower threshold for falls
            terminate = True
            total_reward -= 1000.0  # Severe penalty
        elif height < 0.9 and self.current_step > 100:  # Grace period for startup
            terminate = True
            total_reward -= 500.0
        elif abs(quat_w) < 0.6:  # Very tilted
            terminate = True
            total_reward -= 500.0
        
        # === Debug (less frequent) ===
        if self.current_step % 200 == 0:
            print(f"Step {self.current_step:4d} | "
                f"R:{total_reward:6.1f} | "
                f"H:{height:.2f}({height_reward:5.0f}) | "
                f"Hv:{height_vel:.2f}({height_vel_penalty:4.0f}) | "
                f"Up:{quat_w:.2f}({upright_reward:3.0f}) | "
                f"COM:{com_error:.2f}({com_penalty:3.0f})")
        
        # Store for smoothness calculation
        self.prev_height = height
        
        return total_reward, terminate
    
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
    env = make_standing_env(render_mode=None, config=self.config)
    
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