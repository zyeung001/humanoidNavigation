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
            exclude_current_positions_from_observation=True  # Default for Humanoid-v5; excludes x/y
        )
        super().__init__(env)
        
        # Use config for max_episode_steps if provided
        self.base_target_height = 1.4  # Correct target height for full standing

        self.max_episode_steps = config.get('max_episode_steps', 5000) if config else 5000
        self.current_step = 0

        self.domain_rand = config.get('domain_rand', False) if config else False
        self.rand_mass_range = config.get('rand_mass_range', [0.95, 1.05]) if config else [0.95, 1.05]
        self.rand_friction_range = config.get('rand_friction_range', [0.95, 1.05]) if config else [0.95, 1.05]
        
        # Track reward components for debugging
        self.reward_history = {
            'height': [], 'upright': [], 'velocity': [], 
            'angular': [], 'position': [], 'control': []
        }
        
        # Verify observation space (should be 348 for Humanoid-v5)
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
        
        # State extraction
        height = self.env.unwrapped.data.qpos[2]
        vel = self.env.unwrapped.data.qvel[0:3]
        quat = self.env.unwrapped.data.qpos[3:7]
        angular_vel = self.env.unwrapped.data.qvel[3:6]
        root_x, root_y = self.env.unwrapped.data.qpos[0:2]
        
        # Calculate key metrics
        height_error = abs(height - self.target_height)
        total_vel = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        xy_dist = np.sqrt(root_x**2 + root_y**2)
        
        # IMPROVED REWARD FUNCTION: Progressive height rewards with stability bonuses
        
        # 1. HEIGHT REWARD (Primary objective) - More generous scaling
        if height_error < 0.02:  # Excellent (within 2cm)
            height_reward = 100.0
        elif height_error < 0.05:  # Very good (within 5cm)
            height_reward = 80.0
        elif height_error < 0.10:  # Good (within 10cm)
            height_reward = 60.0
        elif height_error < 0.20:  # Acceptable (within 20cm)
            height_reward = 40.0
        elif height_error < 0.30:  # Poor but not terrible (within 30cm)
            height_reward = 20.0
        else:  # Too far - but don't punish too harshly
            height_reward = max(0, 10.0 - height_error * 10.0)
        
        # 2. UPRIGHT ORIENTATION (Critical for standing)
        # quat[0] is the w component, should be close to 1.0 when upright
        upright_bonus = 30.0 * max(0, quat[0] - 0.3)  # Reward being upright, more generous
        
        # 3. STABILITY REWARDS (Encourage smooth control)
        # Reward low velocity (being still)
        velocity_bonus = max(0, 20.0 - total_vel * 5.0)  # Bonus for being still
        
        # Reward low angular velocity (not spinning)
        angular_bonus = max(0, 15.0 - np.sum(np.square(angular_vel)) * 2.0)
        
        # 4. POSITION STABILITY (Stay in place)
        position_bonus = max(0, 10.0 - xy_dist * 5.0)  # Bonus for staying centered
        
        # 5. CONTROL EFFICIENCY (Encourage smooth actions)
        action_magnitude = np.sum(np.square(action))
        control_bonus = max(0, 5.0 - action_magnitude * 0.1)  # Small bonus for efficient control
        
        # 6. SURVIVAL BONUS (Base reward for not falling)
        survival_bonus = 10.0
        
        # 7. PROGRESSIVE BONUS (Reward for maintaining good state)
        if height_error < 0.10 and quat[0] > 0.7:  # Good standing state
            progressive_bonus = 25.0
        elif height_error < 0.20 and quat[0] > 0.5:  # Decent standing state
            progressive_bonus = 15.0
        else:
            progressive_bonus = 0.0
        
        # Total reward (all positive components)
        total_reward = (
            height_reward + 
            upright_bonus +
            velocity_bonus +
            angular_bonus +
            position_bonus +
            control_bonus +
            survival_bonus +
            progressive_bonus
        )
        
        # Track reward components for debugging
        self.reward_history['height'].append(height_reward)
        self.reward_history['upright'].append(upright_bonus)
        self.reward_history['velocity'].append(velocity_bonus)
        self.reward_history['angular'].append(angular_bonus)
        self.reward_history['position'].append(position_bonus)
        self.reward_history['control'].append(control_bonus)
        
        # MUCH MORE LENIENT TERMINATION - only if completely fallen
        # Only terminate if really fallen over or crashed
        terminate = height < 0.2 or quat[0] < 0.05 or height > 3.0

        # Detailed logging every 100 steps
        if self.current_step % 100 == 0:
            print(f"Step {self.current_step}: h={height:.2f}, err={height_error:.3f}, "
                  f"vel={total_vel:.3f}, quat[0]={quat[0]:.3f}, r={total_reward:.1f}")
            print(f"  Components: height={height_reward:.1f}, upright={upright_bonus:.1f}, "
                  f"vel={velocity_bonus:.1f}, ang={angular_bonus:.1f}, "
                  f"pos={position_bonus:.1f}, ctrl={control_bonus:.1f}, prog={progressive_bonus:.1f}")
            
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