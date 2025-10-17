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
        self.base_target_height = 1.4

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
        
        default_height = self.env.unwrapped.data.qpos[2]
        
        self.current_step = 0

        # Add small noise to joints for robustness
        qpos = self.env.unwrapped.data.qpos.copy()
        qpos[7:] += np.random.uniform(-0.01, 0.01, size=len(qpos[7:]))  # Joints after root
        self.env.unwrapped.data.qpos[:] = qpos

        self.prev_height = default_height
        self.target_height = self.base_target_height
        
        if self.domain_rand:
            # Randomize body masses
            original_masses = self.env.unwrapped.model.body_mass.copy()
            self.env.unwrapped.model.body_mass *= np.random.uniform(
                self.rand_mass_range[0], self.rand_mass_range[1],
                size=self.env.unwrapped.model.body_mass.shape
            )
            
            # Randomize geom friction (for feet/contact surfaces)
            original_friction = self.env.unwrapped.model.geom_friction.copy()
            self.env.unwrapped.model.geom_friction[:, 0] *= np.random.uniform(
                self.rand_friction_range[0], self.rand_friction_range[1],
                size=self.env.unwrapped.model.geom_friction.shape[0]
            )
        
        return observation, info
    
    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # Modify reward for standing
        reward, terminate = self._compute_task_reward(observation, base_reward, info, action)
        
        # Override termination for standing to allow indefinite episodes
        terminated = terminate  # Only true if fallen hard
        truncated = self.current_step >= self.max_episode_steps
        
        # Add task info 
        info.update(self._get_task_info())
        
        return observation, reward, terminated, truncated, info
    
    def _pose_reward(self):
        """Reward for joints close to standing pose (0 for most)."""
        qpos = self.env.unwrapped.data.qpos[7:]  # Joint positions after root
        # Assume standing is all 0, adjust if needed
        return np.exp(-0.2 * np.sum(np.square(qpos)))
    
    def _tol(self, x, bounds=(0,0), margin=2.0):
        """Tolerance function: high reward in [lower, upper], drops off to 0 outside with margin."""
        lower, upper = bounds
        if lower == upper:
            diff = abs(x - lower)
            return np.exp(- (diff / margin)**2)  # Smooth version instead of binary
        else:
            # For range, but here for still it's (0,0)
            return 1.0 - np.clip(abs(x - (lower + upper)/2) / margin, 0, 1)

    def _upright(self):
        """Upright torso reward [0,1]."""
        # Get torso quaternion
        torso_quat = self.env.unwrapped.data.body("torso").xquat  # [w, x, y, z]
        # World up vector [0,0,1]
        # Rotate local up [0,0,1] by conjugate quat to get world alignment
        # Simplified dot product with up
        # For upright, the z-component of rotated xmat should be high
        torso_xmat = self.env.unwrapped.data.body("torso").xmat.reshape(3,3)
        up_alignment = torso_xmat[2, 2]  # z-z component, cosine of angle to up
        return np.clip((up_alignment + 1) / 2, 0, 1)  # Map [-1,1] to [0,1]

    def _height_reward(self):
        """Height reward [0,1]."""
        torso_id = mj_name2id(self.env.unwrapped.model, mjtObj.mjOBJ_BODY, "torso")
        height = self.env.unwrapped.data.xpos[torso_id][2]
        target = 1.4  # Adjusted to match initial standing
        return self._tol(height, (target, target), margin=0.03)

    def _effort(self, action):
        """Low effort reward [0,1]."""
        return np.exp(-0.1 * np.sum(np.square(action)))  # Scale to [0,1], adjust coef

    def _still(self):
        """Low velocity in x,y."""
        vel = self.env.unwrapped.data.qvel[0:2]  # x,y only, ignore z
        still_x = self._tol(vel[0], (0,0), margin=0.1)
        still_y = self._tol(vel[1], (0,0), margin=0.1)
        return (still_x + still_y) / 2.0
        
    def _compute_task_reward(self, obs, base_reward, info, action):
        """Balanced reward for learning to stand still"""
        height = self.env.unwrapped.data.qpos[2]
        quat = self.env.unwrapped.data.qpos[3:7]  # Though not directly used, for completeness
        vel = self.env.unwrapped.data.qvel[0:3]
        angular_vel = self.env.unwrapped.data.qvel[3:6]
        root_x, root_y = self.env.unwrapped.data.qpos[0:2]
        
        # Components (each [0,1] or similar)
        upright = self._upright()
        height_r = self._height_reward()
        effort = self._effort(action)
        still = self._still()
        pose_r = self._pose_reward()

        # Additive weighted reward
        total_reward = (
            10.0 * height_r +   # Increase priority on height
            5.0 * upright +     # Increase on posture
            0.5 * effort +      # Reduce effort weight to allow more control if needed
            2.0 * still +       # Keep minimal movement
            2.0 * pose_r  # Encourage correct pose
        )

        # Survival bonus per step
        total_reward += 1.0

        # Small penalties
        angular_penalty = -0.2 * np.sum(np.square(angular_vel))  
        position_penalty = -0.1 * (root_x**2 + root_y**2)
        total_reward += angular_penalty + position_penalty

        ctrl_cost = -0.5 * np.square(action).sum()  # Already in effort, but enhance
        impact_cost = -5e-7 * np.square(self.env.unwrapped.data.cfrc_ext).sum()  # Penalize external contacts
        total_reward += impact_cost

        z_vel_penalty = -0.1 * abs(vel[2])  # Minimize vertical bobbing
        total_reward += z_vel_penalty

        # Impact penalty to avoid harsh contacts
        impact = -5e-7 * np.square(self.env.unwrapped.data.cfrc_ext).sum()
        total_reward += impact

        # Termination only if truly fallen
        terminate = height < 0.5 or upright < 0.3

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
    env = make_standing_env(render_mode=None, config=None)
    
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