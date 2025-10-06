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
        self.max_episode_steps = config.get('max_episode_steps', 5000) if config else 5000
        self.current_step = 0

        self.domain_rand = config.get('domain_rand')
        self.rand_mass_range = config.get('rand_mass_range')
        self.rand_friction_range = config.get('rand_friction_range', [0.7, 1.3])
        
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
        """Reward function for stable, indefinite standing at target height."""
        
        # === State extraction ===
        height = self.env.unwrapped.data.qpos[2]
        root_x, root_y = self.env.unwrapped.data.qpos[0:2]
        quat_w = self.env.unwrapped.data.qpos[3]
        
        linear_vel = self.env.unwrapped.data.qvel[0:3]
        angular_vel = self.env.unwrapped.data.qvel[3:6]

        # Get torso ID (root body for Humanoid)
        torso_id = mj_name2id(self.env.unwrapped.model, mjtObj.mjOBJ_BODY, 'torso')
        com_pos = self.env.unwrapped.data.subtree_com[torso_id]  # [x, y, z]

        # COM and feet for balance (use body_xpos as fixed)
        left_foot_pos = self.env.unwrapped.data.body_xpos[mj_name2id(self.env.unwrapped.model, mjtObj.mjOBJ_BODY, 'left_foot')]
        right_foot_pos = self.env.unwrapped.data.body_xpos[mj_name2id(self.env.unwrapped.model, mjtObj.mjOBJ_BODY, 'right_foot')]
        support_center = (left_foot_pos[:2] + right_foot_pos[:2]) / 2
        com_error = np.linalg.norm(com_pos[:2] - support_center)
        
        # === Primary: Height (smooth Gaussian, high scale) ===
        target_height = 1.3
        height_error = abs(height - target_height)
        height_reward = 100.0 * np.exp(-10.0 * height_error**2)  # Peak 100 at target, drops smoothly
        
        # === Uprightness ===
        upright_reward = 30.0 * np.exp(-5.0 * (1.0 - abs(quat_w))**2)  # Peak 30 when vertical
        
        # === Stability (stronger penalties to minimize motion/drift) ===
        position_error = np.sqrt(root_x**2 + root_y**2)
        position_penalty = -5.0 * np.clip(position_error**2, 0, 10)  # Clipped max -50
        
        linear_vel_penalty = -3.0 * np.clip(np.sum(np.square(linear_vel)), 0, 10)  # Clipped max -30
        angular_vel_penalty = -2.0 * np.clip(np.sum(np.square(angular_vel)), 0, 10)  # Clipped max -20
        
        # COM balance
        com_penalty = -10.0 * np.clip(com_error**2, 0, 5)  # Clipped max -50, stronger than before
        
        # === Efficiency ===
        control_penalty = -0.01 * np.sum(np.square(action))  # Minimal
        
        # === Survival (constant dense bonus for longevity) ===
        survival_bonus = 5.0 + 0.001 * self.current_step  # Ramp slightly for longer episodes

        healthy_bonus = 15.0 if height > 1.0 and abs(quat_w) > 0.8 else 0.0
        
        # === Total ===
        total_reward = (
            height_reward +
            upright_reward +
            position_penalty +
            linear_vel_penalty +
            angular_vel_penalty +
            com_penalty +
            control_penalty +
            survival_bonus +
            healthy_bonus
        )
        
        # === Termination (tighter for bad tilts/low height) ===
        terminate = False
        if height < 0.8 or abs(quat_w) < 0.7:  # Stricter than 0.5/0.2 to avoid prolonged bad states
            terminate = True
            total_reward -= 500.0  # Harsh fall penalty
        
        # === Debug ===
        if self.current_step % 1000 == 0 and os.getpid() == os.getppid():
            print(f"Step {self.current_step:4d} | "
                f"Total: {total_reward:6.1f} | "
                f"H: {height:5.2f} ({height_reward:5.1f}) | "
                f"Up: {quat_w:4.2f} ({upright_reward:5.1f}) | "
                f"COM err: {com_error:5.2f} ({com_penalty:6.1f})")
        
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