# humanoid_velocity_env.py
"""
Humanoid Velocity Tracking Environment with Integrated Command Generation.

This module demonstrates the integration of VelocityCommandGenerator
into an OpenAI Gym-style environment with the complete reward function
for velocity tracking training.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium.spaces import Box

# Import the velocity command generator from core
from src.core.command_generator import VelocityCommandGenerator


class HumanoidVelocityEnv(gym.Wrapper):
    """
    Humanoid environment with velocity command generation and tracking reward.
    
    Implements the standard recipe for humanoid locomotion training:
    - VelocityCommandGenerator for target command generation
    - Gaussian kernel reward for velocity tracking
    - Survival reward for staying upright
    - Effort penalty for action magnitude
    
    Reward Function:
        R_total = R_tracking + R_upright + R_effort
        
        R_tracking = exp(-β * ||v_target - v_agent||²)  [Gaussian kernel]
        R_upright = +10.0 if upright else 0.0           [Survival reward]
        R_effort = -0.01 * ||action||²                   [Effort penalty]
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the humanoid velocity tracking environment.
        
        Args:
            render_mode: 'human', 'rgb_array', or None
            config: Configuration dictionary with optional overrides
        """
        # Create base Humanoid environment
        env = gym.make(
            "Humanoid-v5",
            render_mode=render_mode,
            exclude_current_positions_from_observation=False
        )
        super().__init__(env)
        
        # Configuration
        self.cfg = config or {}
        self.dt = self.cfg.get('dt', 0.01)  # Simulation timestep
        self.max_episode_steps = self.cfg.get('max_episode_steps', 5000)
        self.current_step = 0
        
        # ========== REWARD PARAMETERS ==========
        # β parameter for Gaussian kernel (higher = stricter tracking)
        self.beta = self.cfg.get('beta', 5.0)
        
        # Upright survival reward
        self.upright_reward_value = self.cfg.get('upright_reward', 10.0)
        
        # Effort penalty coefficient
        self.effort_penalty_coef = self.cfg.get('effort_penalty_coef', 0.01)
        
        # ========== COMMAND GENERATOR ==========
        vx_range = self.cfg.get('vx_range', (-0.5, 1.5))
        vy_range = self.cfg.get('vy_range', (-0.5, 0.5))
        yaw_rate_range = self.cfg.get('yaw_rate_range', (-1.0, 1.0))
        switch_interval = self.cfg.get('switch_interval_range', (2.0, 5.0))
        stop_probability = self.cfg.get('stop_probability', 0.15)
        
        self.command_generator = VelocityCommandGenerator(
            vx_range=vx_range,
            vy_range=vy_range,
            yaw_rate_range=yaw_rate_range,
            switch_interval_range=switch_interval,
            stop_probability=stop_probability
        )
        
        # Current target command
        self._target_command = np.zeros(3, dtype=np.float32)
        
        # ========== OBSERVATION SPACE ==========
        # Base observation + commanded velocity (3 components)
        base_obs_dim = env.observation_space.shape[0] + 15  # Humanoid-v5 adjustment
        command_dim = 3  # [vx, vy, yaw_rate]
        total_obs_dim = base_obs_dim + command_dim
        
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
        # ========== REWARD TRACKING ==========
        self.reward_components = {
            'tracking': [],
            'upright': [],
            'effort': []
        }
        
        print(f"HumanoidVelocityEnv initialized:")
        print(f"  Observation dim: {total_obs_dim}")
        print(f"  Beta (tracking strictness): {self.beta}")
        print(f"  Upright reward: {self.upright_reward_value}")
        print(f"  Effort penalty coefficient: {self.effort_penalty_coef}")
    
    @property
    def agent_velocity(self) -> np.ndarray:
        """
        Get current agent velocity in world frame.
        
        Returns:
            np.ndarray: [actual_vx, actual_vy, actual_yaw_rate]
        """
        # Linear velocity (world frame)
        linear_vel = self.env.unwrapped.data.qvel[0:3]
        
        # Angular velocity around z-axis (yaw rate)
        angular_vel = self.env.unwrapped.data.qvel[3:6]
        yaw_rate = angular_vel[2]  # z-component
        
        return np.array([linear_vel[0], linear_vel[1], yaw_rate], dtype=np.float32)
    
    def is_upright(self) -> bool:
        """
        Check if the humanoid is in an upright position.
        
        Returns:
            bool: True if the humanoid is upright and at a reasonable height
        """
        # Get height
        height = self.env.unwrapped.data.qpos[2]
        
        # Get orientation quaternion [w, x, y, z]
        quat = self.env.unwrapped.data.qpos[3:7]
        
        # Check if upright:
        # - Height above threshold (not fallen)
        # - Quaternion w component close to 1 (upright orientation)
        height_ok = height > 1.0
        orientation_ok = abs(quat[0]) > 0.7  # w > 0.7 means roughly upright
        
        return height_ok and orientation_ok
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment.
        
        Returns:
            observation: Initial observation with commanded velocity
            info: Dictionary with additional information
        """
        observation, info = self.env.reset(seed=seed, options=options)
        
        self.current_step = 0
        
        # Reset command generator and get initial command
        self._target_command = self.command_generator.reset(sample_new=True)
        
        # Clear reward history
        for key in self.reward_components:
            self.reward_components[key] = []
        
        # Augment observation with commanded velocity
        observation = self._augment_observation(observation)
        
        # Add command info
        info['target_command'] = self._target_command.copy()
        info['agent_velocity'] = self.agent_velocity
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """
        Execute one environment step.
        
        This implements the core training loop:
        1. Update target command from generator
        2. Step physics simulation
        3. Compute reward using Gaussian kernel tracking
        4. Check termination conditions
        
        Args:
            action: Agent's action
            
        Returns:
            observation: Next observation
            reward: Total reward (R_tracking + R_upright + R_effort)
            terminated: Whether episode ended due to failure
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # ========== UPDATE TARGET COMMAND ==========
        self._target_command = self.command_generator.get_command(self.dt)
        
        # ========== STEP PHYSICS ==========
        observation, _, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        
        # ========== COMPUTE REWARD ==========
        reward, terminated = self._compute_reward(action, terminated)
        
        # ========== CHECK TRUNCATION ==========
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        # ========== AUGMENT OBSERVATION ==========
        observation = self._augment_observation(observation)
        
        # ========== UPDATE INFO ==========
        info['target_command'] = self._target_command.copy()
        info['agent_velocity'] = self.agent_velocity
        info['is_upright'] = self.is_upright()
        info['velocity_error'] = np.linalg.norm(
            self._target_command[:2] - self.agent_velocity[:2]
        )
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward(
        self,
        action: np.ndarray,
        base_terminated: bool
    ) -> Tuple[float, bool]:
        """
        Compute the total reward using the standard recipe.
        
        R_total = R_tracking + R_upright + R_effort
        
        Args:
            action: The action taken
            base_terminated: Whether the base env signaled termination
            
        Returns:
            total_reward: Combined reward value
            terminated: Updated termination flag
        """
        # ========== R_TRACKING: GAUSSIAN KERNEL VELOCITY TRACKING ==========
        # R_tracking = exp(-β * ||v_target - v_agent||²)
        v_target = self._target_command
        v_agent = self.agent_velocity
        
        velocity_error_squared = np.sum((v_target - v_agent) ** 2)
        R_tracking = np.exp(-self.beta * velocity_error_squared)
        
        # ========== R_UPRIGHT: SURVIVAL REWARD ==========
        # Binary reward for staying upright
        R_upright = self.upright_reward_value if self.is_upright() else 0.0
        
        # ========== R_EFFORT: ACTION PENALTY ==========
        # Penalize large actions to encourage smooth control
        R_effort = -self.effort_penalty_coef * np.sum(action ** 2)
        
        # ========== TOTAL REWARD ==========
        R_total = R_tracking + R_upright + R_effort
        
        # ========== TERMINATION CHECK ==========
        height = self.env.unwrapped.data.qpos[2]
        quat = self.env.unwrapped.data.qpos[3:7]
        
        terminated = (
            base_terminated or
            height < 0.8 or           # Fallen too low
            height > 2.0 or           # Jumped too high
            abs(quat[0]) < 0.5        # Too tilted
        )
        
        # ========== TRACK REWARD COMPONENTS ==========
        self.reward_components['tracking'].append(R_tracking)
        self.reward_components['upright'].append(R_upright)
        self.reward_components['effort'].append(R_effort)
        
        # ========== DEBUG LOGGING ==========
        if self.current_step % 500 == 0:
            print(f"Step {self.current_step:4d}: "
                  f"cmd=({v_target[0]:.2f}, {v_target[1]:.2f}, {v_target[2]:.2f}), "
                  f"agent=({v_agent[0]:.2f}, {v_agent[1]:.2f}, {v_agent[2]:.2f}), "
                  f"R_track={R_tracking:.3f}, R_up={R_upright:.1f}, R_eff={R_effort:.3f}, "
                  f"R_tot={R_total:.2f}")
        
        return R_total, terminated
    
    def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Augment observation with commanded velocity.
        
        Args:
            obs: Base observation from environment
            
        Returns:
            Augmented observation including target command
        """
        return np.concatenate([
            obs,
            self._target_command
        ]).astype(np.float32)
    
    def update_curriculum(self, new_ranges: list) -> None:
        """
        Update command generator ranges for curriculum learning.
        
        Args:
            new_ranges: List of 3 tuples for [vx, vy, yaw_rate] ranges
        """
        self.command_generator.update_curriculum_ranges(new_ranges)
    
    def get_reward_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze reward components over the episode.
        
        Returns:
            Dictionary with statistics for each reward component
        """
        analysis = {}
        for component, values in self.reward_components.items():
            if values:
                analysis[component] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'total': np.sum(values)
                }
        return analysis
    
    def get_current_command(self) -> np.ndarray:
        """Get the current target command."""
        return self._target_command.copy()


def make_humanoid_velocity_env(
    render_mode: Optional[str] = None,
    config: Optional[Dict] = None
) -> HumanoidVelocityEnv:
    """Factory function to create the environment."""
    return HumanoidVelocityEnv(render_mode=render_mode, config=config)


# ============================================================================
# EXAMPLE: Step Function Implementation (as requested in Prompt 3)
# ============================================================================

def example_step_function():
    """
    Demonstrates the step function implementation with all components.
    
    This is a standalone example showing how the step function works
    with the command generator and reward computation.
    """
    
    print("=" * 70)
    print("Example: Humanoid Step Function with Velocity Command Generator")
    print("=" * 70)
    
    # Configuration
    config = {
        'beta': 5.0,                    # Gaussian kernel parameter
        'upright_reward': 10.0,         # Survival reward
        'effort_penalty_coef': 0.01,    # Action penalty
        'vx_range': (-0.5, 1.5),
        'vy_range': (-0.5, 0.5),
        'yaw_rate_range': (-1.0, 1.0),
        'stop_probability': 0.15,
        'max_episode_steps': 1000
    }
    
    # Create environment
    env = make_humanoid_velocity_env(render_mode=None, config=config)
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial target command: {info['target_command']}")
    
    # Run a few steps
    total_reward = 0.0
    print("\nRunning 500 steps with random actions...")
    
    for step in range(500):
        # Random action (replace with your policy)
        action = env.action_space.sample()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Log periodically
        if step % 100 == 0:
            print(f"  Step {step}: reward={reward:.2f}, "
                  f"vel_error={info['velocity_error']:.3f}, "
                  f"upright={info['is_upright']}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
    
    # Analyze rewards
    print("\n" + "=" * 50)
    print("Reward Analysis:")
    analysis = env.get_reward_analysis()
    for component, stats in analysis.items():
        print(f"  {component}:")
        print(f"    mean={stats['mean']:.3f}, total={stats['total']:.1f}")
    
    print(f"\nTotal episode reward: {total_reward:.1f}")
    
    # Command generator statistics
    gen_stats = env.command_generator.get_statistics()
    print(f"\nCommand Generator Stats:")
    print(f"  Total commands: {gen_stats['total_commands']}")
    print(f"  Stop commands: {gen_stats['stop_commands']} ({gen_stats['stop_ratio']:.1%})")
    
    env.close()
    print("\nExample completed!")


if __name__ == "__main__":
    example_step_function()

