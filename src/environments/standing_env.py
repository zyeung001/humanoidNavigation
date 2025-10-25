"""
Standing environment wrapper for MuJoCo Humanoid-v5
OPTIMIZED reward for indefinite standing balance

FIXES:
- Simplified reward function (height first, then stability)
- More lenient termination conditions
- Better reward shaping with clearer objectives
- Enhanced debugging output

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
            exclude_current_positions_from_observation=True
        )
        super().__init__(env)
        
        
        self.base_target_height = 1.3
        
        self.max_episode_steps = config.get('max_episode_steps', 5000) if config else 5000
        self.current_step = 0
        
        self.domain_rand = config.get('domain_rand', False) if config else False
        self.rand_mass_range = config.get('rand_mass_range', [0.95, 1.05]) if config else [0.95, 1.05]
        self.rand_friction_range = config.get('rand_friction_range', [0.95, 1.05]) if config else [0.95, 1.05]
        
        self.reward_history = {
            'height': [], 'upright': [], 'velocity': [], 
            'angular': [], 'position': [], 'control': []
        }
        
        obs_space = env.observation_space
        print(f"Observation space shape for standing: {obs_space.shape}")
    
    def reset(self, seed: Optional[int] = None): 
        observation, info = self.env.reset(seed=seed)
        
        default_height = self.env.unwrapped.data.qpos[2]
        
        self.current_step = 0
        self.prev_height = default_height
        self.target_height = self.base_target_height
        
        # Clear reward history
        for key in self.reward_history:
            self.reward_history[key] = []
        
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
        reward, terminated = self._compute_task_reward(observation, base_reward, info, action)
        
        # Override termination for standing to allow indefinite episodes
        truncated = self.current_step >= self.max_episode_steps
        
        # Add task info 
        info.update(self._get_task_info())
        
        return observation, reward, terminated, truncated, info
        
    def _compute_task_reward(self, obs, base_reward, info, action):
        """SIMPLIFIED: Only 3 components"""
        # State extraction
        height = self.env.unwrapped.data.qpos[2]
        quat = self.env.unwrapped.data.qpos[3:7]
        
        target_height = 1.3
        height_error = abs(height - target_height)
        
        # === ONLY 3 REWARD COMPONENTS ===
        height_reward = 30.0 * np.exp(-10.0 * height_error**2)
        upright_reward = 10.0 * max(0, quat[0] - 0.5) / 0.5
        control_cost = -0.5 * np.sum(np.square(action))
        
        total_reward = height_reward + upright_reward + control_cost
        
        # Lenient termination
        terminate = height < 0.6 or height > 2.0 or quat[0] < 0.3
        
        # Debug logging
        if self.current_step % 100 == 0:
            print(f"Step {self.current_step}: h={height:.3f}, quat={quat[0]:.3f}, "
                f"r={total_reward:.1f} (h={height_reward:.1f}, u={upright_reward:.1f}, "
                f"c={control_cost:.1f})")
        
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
            'quaternion_w': self.env.unwrapped.data.qpos[3],
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
    
    def get_reward_analysis(self):
        """Analyze reward components over episode"""
        if not any(self.reward_history.values()):
            return None
            
        analysis = {}
        for component, values in self.reward_history.items():
            if values:
                analysis[component] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'total': np.sum(values)
                }
        return analysis


def make_standing_env(render_mode=None, config=None):
    """Factory function to create standing environment"""
    return StandingEnv(render_mode=render_mode, config=config)


def test_environment():
    """Test the fixed environment"""
    print("Testing OPTIMIZED Humanoid Standing Environment")
    print("=" * 60)
    
    # Test with random policy
    env = make_standing_env(render_mode=None, config=None)
    
    # Show observation info
    env.get_observation_info()
    
    obs, info = env.reset()
    total_reward = 0
    
    print("\nRunning 200 steps with random policy...")
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 50 == 0:
            print(f"\nStep {step}:")
            print(f"  Height: {info['height']:.3f} (target: 1.4)")
            print(f"  Quaternion w: {info['quaternion_w']:.3f}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Total reward: {total_reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step} ({'terminated' if terminated else 'truncated'})")
            break
    
    # Show reward analysis
    print("\n" + "=" * 60)
    print("Reward Component Analysis:")
    print("=" * 60)
    analysis = env.get_reward_analysis()
    if analysis:
        for component, stats in analysis.items():
            print(f"\n{component.upper()}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Total contribution: {stats['total']:.2f}")
            print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    
    env.close()
    print(f"\nFinal total reward: {total_reward:.2f}")


if __name__ == "__main__":
    test_environment()