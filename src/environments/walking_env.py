# walking_env.py
"""
Walking environment for humanoid locomotion.
Command-conditioned on desired world velocity (vx_world, vy_world).
Extends standing environment with velocity tracking rewards.

Uses modular RewardCalculator for clean reward computation.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium.spaces import Box

# Import modular reward calculator and command generator
from src.core.rewards import RewardCalculator, RewardWeights, RewardMetrics
from src.core.command_generator import VelocityCommandGenerator


class WalkingEnv(gym.Wrapper):
    """
    Walking environment that conditions policy on commanded world-frame velocity.
    
    Observations include:
    - Base humanoid observations (365 dims)
    - COM features if enabled (+6 dims)  
    - Commanded velocity (vx_world, vy_world) (+2 dims)
    - History stacking (×4)
    
    Total: (365 + 6 + 2) × 4 = 1492 dims
    """
    
    def __init__(self, render_mode: Optional[str] = None, config=None):
        env_id = "Humanoid-v5"
        print(f"Using {env_id} for walking task")
        
        # Create base environment
        env = gym.make(
            env_id, 
            render_mode=render_mode,
            exclude_current_positions_from_observation=False  # Adds +2 dims (x,y)
        )
        super().__init__(env)
        
        # Configuration
        self.cfg = config or {}
        self.base_target_height = 1.4
        self.max_episode_steps = self.cfg.get('max_episode_steps', 5000)
        self.current_step = 0
        
        # Domain randomization
        self.domain_rand = self.cfg.get('domain_rand', False)
        self.rand_mass_range = self.cfg.get('rand_mass_range', [0.95, 1.05])
        self.rand_friction_range = self.cfg.get('rand_friction_range', [0.95, 1.05])
        
        # Random height initialization for recovery training
        self.random_height_init = self.cfg.get('random_height_init', True)
        self.random_height_prob = self.cfg.get('random_height_prob', 0.3)
        self.random_height_range = self.cfg.get('random_height_range', [-0.3, 0.1])
        
        # Push perturbations for robustness training
        self.push_enabled = self.cfg.get('push_enabled', True)
        self.push_interval = self.cfg.get('push_interval', 200)  # Apply push every N steps
        self.push_magnitude_range = self.cfg.get('push_magnitude_range', [50.0, 150.0])  # Force magnitude (N)
        self.push_duration = self.cfg.get('push_duration', 5)  # Steps to apply push
        self.push_countdown = 0  # Counter for push duration
        self.current_push_force = np.zeros(3)  # Current push force being applied
        
        # Reward caps from config
        reward_caps = self.cfg.get('reward_caps', {})
        self.max_height_maintenance_penalty = reward_caps.get('max_height_maintenance_penalty', 15.0)
        self.recovery_bonus_scale = reward_caps.get('recovery_bonus_scale', 50.0)
        self.termination_penalty_constant = reward_caps.get('termination_penalty_constant', 50.0)
        
        # ========== WALKING-SPECIFIC CONFIG ==========
        self.velocity_weight = float(self.cfg.get('velocity_weight', 5.0))
        self.max_commanded_speed = float(self.cfg.get('max_commanded_speed', 0.0))  # Curriculum controls this
        self.fixed_command = self.cfg.get('fixed_command', None)  # (vx, vy) tuple for inference
        
        # Current commanded velocity (set in reset) - 3 elements per Prompt 1
        self.commanded_vx_world = 0.0
        self.commanded_vy_world = 0.0
        self.commanded_yaw_rate = 0.0  # Per Prompt 1: [vx, vy, yaw_rate]
        self.commanded_speed = 0.0
        self.commanded_angle = 0.0
        
        self.reward_history = {
            'height': [], 'upright': [], 'velocity': [], 
            'angular': [], 'position': [], 'control': [],
            'velocity_tracking': [],  # Track velocity reward
            'jerk': []  # Track action smoothness
        }
        
        # ========== MODULAR REWARD CALCULATOR ==========
        reward_weights = RewardWeights(
            tracking=float(self.cfg.get('reward_tracking_weight', 10.0)),
            direction_bonus=float(self.cfg.get('reward_direction_weight', 5.0)),
            height=float(self.cfg.get('reward_height_weight', 5.0)),
            upright=float(self.cfg.get('reward_upright_weight', 3.0)),
            alive=float(self.cfg.get('reward_alive_weight', 1.0)),
            action_penalty=float(self.cfg.get('reward_action_penalty', 0.005)),
            jerk_penalty=float(self.cfg.get('reward_jerk_penalty', 0.01)),
        )
        self.reward_calculator = RewardCalculator(
            weights=reward_weights,
            target_height=self.base_target_height,
            height_bandwidth=10.0,
            tracking_bandwidth=float(self.cfg.get('reward_tracking_bandwidth', 4.0)),
        )
        
        # ========== VELOCITY COMMAND GENERATOR (Per Prompt 1) ==========
        # Generates [vx, vy, yaw_rate] with uniform sampling at 2-5 second intervals
        # 15% probability of stop command for braking practice
        self.command_generator = VelocityCommandGenerator(
            vx_range=(
                float(self.cfg.get('cmd_vx_min', -0.5)),
                float(self.cfg.get('cmd_vx_max', 1.5))
            ),
            vy_range=(
                float(self.cfg.get('cmd_vy_min', -0.5)),
                float(self.cfg.get('cmd_vy_max', 0.5))
            ),
            yaw_rate_range=(
                float(self.cfg.get('cmd_yaw_min', -1.0)),
                float(self.cfg.get('cmd_yaw_max', 1.0))
            ),
            switch_interval_range=(2.0, 5.0),  # Per Prompt 1: 2-5 seconds
            stop_probability=float(self.cfg.get('stop_probability', 0.15)),  # Per Prompt 1: 15%
        )
        
        # Simulation timestep for command generator
        self.dt = self.env.unwrapped.model.opt.timestep * self.env.unwrapped.frame_skip
        
        # Beta parameter for Gaussian kernel reward (Prompt 3)
        self.tracking_beta = float(self.cfg.get('tracking_beta', 5.0))

        # ======== Enhanced controls (all optional via config) ========
        # Action preprocessing
        self.enable_action_smoothing = bool(self.cfg.get('action_smoothing', False))
        self.action_smoothing_tau = float(self.cfg.get('action_smoothing_tau', 0.2))
        self.enable_action_symmetry = bool(self.cfg.get('action_symmetry', False))
        self.enable_pd_assist = bool(self.cfg.get('pd_assist', False))
        self.pd_kp = float(self.cfg.get('pd_kp', 0.0))
        self.pd_kd = float(self.cfg.get('pd_kd', 0.0))
        self.prev_action = np.zeros(self.env.action_space.shape, dtype=np.float32)

        # Observation processing
        self.enable_history = int(self.cfg.get('obs_history', 0)) > 0
        self.history_len = int(self.cfg.get('obs_history', 0))
        self.obs_history = []
        self.include_com = bool(self.cfg.get('obs_include_com', False))
        self.feature_norm = bool(self.cfg.get('obs_feature_norm', False))
        
        # Humanoid-v5 base observation_space.shape[0] = 350
        base_obs_from_space = int(env.observation_space.shape[0])  
        
        # Actual observations have 15 MORE dimensions than observation_space reports
        base_obs_dim = base_obs_from_space + 15  # Actual observations are 365 dims
        
        # Calculate dimension after all processing steps
        extra_dim = 6 if self.include_com else 0  # COM pos (3) + COM vel (3)
        command_dim = 3  # Per Prompt 1: [vx, vy, yaw_rate]
        feature_dim = base_obs_dim + extra_dim + command_dim
        
        if self.enable_history:
            # History stacking multiplies dimension
            total_dim = feature_dim * self.history_len
        else:
            total_dim = feature_dim
        
        # FREEZE observation dimension now (before VecNormalize initialization)
        self.frozen_obs_dim = total_dim
        
        # Declare observation space with frozen dimension
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.frozen_obs_dim,), 
            dtype=np.float32
        )
        
        print(f"Walking environment observation space configuration:")
        print(f"  Base from env.observation_space: {base_obs_from_space}")
        print(f"  + Position inclusion adjustment: +15 → {base_obs_dim}")
        print(f"  + COM features: {extra_dim}")
        print(f"  + Command features: {command_dim}")
        print(f"  = Per-frame dimension: {feature_dim}")
        print(f"  × History stack: {self.history_len if self.enable_history else 1}")
        print(f"  = FROZEN dimension: {self.frozen_obs_dim}")
        print(f"  Velocity weight: {self.velocity_weight}")
        print(f"  Max commanded speed: {self.max_commanded_speed}")
        print(f"  Action smoothing tau: {self.action_smoothing_tau}")
    
    def reset(self, seed: Optional[int] = None): 
        observation, info = self.env.reset(seed=seed)
        
        default_height = self.env.unwrapped.data.qpos[2]
        
        self.current_step = 0
        self.prev_height = default_height
        self.target_height = self.base_target_height
        self.prev_action[:] = 0.0
        self.obs_history = []  # Clear history
        
        # Clear push perturbation state
        self.push_countdown = 0
        self.current_push_force = np.zeros(3)
        try:
            self.env.unwrapped.data.xfrc_applied[:] = 0.0
        except Exception:
            pass
        
        # Clear reward history
        for key in self.reward_history:
            self.reward_history[key] = []
        
        # Reset reward calculator state
        self.reward_calculator.reset()
        
        # ========== RESET COMMAND GENERATOR (Per Prompt 1 & 3) ==========
        # Reset and get initial command from VelocityCommandGenerator
        if self.fixed_command is not None:
            # Fixed command for inference/testing
            self.commanded_vx_world = float(self.fixed_command[0])
            self.commanded_vy_world = float(self.fixed_command[1])
            self.commanded_yaw_rate = 0.0
        else:
            # Use VelocityCommandGenerator for training (per Prompt 1)
            cmd = self.command_generator.reset(sample_new=True)  # [vx, vy, yaw_rate]
            self.commanded_vx_world = float(cmd[0])
            self.commanded_vy_world = float(cmd[1])
            self.commanded_yaw_rate = float(cmd[2])
        
        # Compute derived values
        self.commanded_speed = np.sqrt(self.commanded_vx_world**2 + self.commanded_vy_world**2)
        if self.commanded_speed > 0:
            self.commanded_angle = np.arctan2(self.commanded_vy_world, self.commanded_vx_world)
        else:
            self.commanded_angle = 0.0
        
        if self.domain_rand:
            # Randomize body masses
            self.env.unwrapped.model.body_mass *= np.random.uniform(
                self.rand_mass_range[0], self.rand_mass_range[1],
                size=self.env.unwrapped.model.body_mass.shape
            )
            
            # Randomize geom friction
            self.env.unwrapped.model.geom_friction[:, 0] *= np.random.uniform(
                self.rand_friction_range[0], self.rand_friction_range[1],
                size=self.env.unwrapped.model.geom_friction.shape[0]
            )
        
        # Random height initialization for recovery training
        if self.random_height_init and np.random.random() < self.random_height_prob:
            perturb = np.random.uniform(self.random_height_range[0], self.random_height_range[1])
            new_height = default_height + perturb
            # Clamp to reasonable range
            new_height = np.clip(new_height, 0.6, 1.6)
            self.env.unwrapped.data.qpos[2] = new_height
            self.prev_height = new_height
            
            # Also add small velocity perturbation
            vel_perturb = np.random.uniform(-0.1, 0.1, size=self.env.unwrapped.data.qvel.shape)
            self.env.unwrapped.data.qvel[:] += vel_perturb
            
            # Re-get observation after perturbation
            observation = self.env.unwrapped._get_obs()
        
        # Process observation with guaranteed dimension (includes commanded velocity)
        observation = self._process_observation(observation)
        
        # Add walking info (Per Prompt 1: [vx, vy, yaw_rate])
        info['commanded_vx'] = self.commanded_vx_world
        info['commanded_vy'] = self.commanded_vy_world
        info['commanded_yaw_rate'] = self.commanded_yaw_rate
        info['commanded_speed'] = self.commanded_speed

        return observation, info
    
    def step(self, action):
        # Action preprocessing
        proc_action = self._process_action(np.asarray(action, dtype=np.float32))
        
        # ========== UPDATE TARGET COMMAND (Per Prompt 3) ==========
        # Call command generator every step - it handles timing internally
        # Commands switch at randomized intervals (2-5 seconds per Prompt 1)
        if self.fixed_command is None:
            cmd = self.command_generator.get_command(self.dt)  # [vx, vy, yaw_rate]
            self.commanded_vx_world = float(cmd[0])
            self.commanded_vy_world = float(cmd[1])
            self.commanded_yaw_rate = float(cmd[2])
            self.commanded_speed = np.sqrt(self.commanded_vx_world**2 + self.commanded_vy_world**2)
            if self.commanded_speed > 0:
                self.commanded_angle = np.arctan2(self.commanded_vy_world, self.commanded_vx_world)
            else:
                self.commanded_angle = 0.0
        
        # Apply push perturbation for robustness training
        self._apply_push_perturbation()

        observation, base_reward, terminated, truncated, info = self.env.step(proc_action)
        self.current_step += 1
        
        # Modify reward for walking
        reward, terminated = self._compute_task_reward(observation, base_reward, info, proc_action)
        
        # Override termination for walking to allow indefinite episodes
        truncated = self.current_step >= self.max_episode_steps
        
        # Add task info 
        info.update(self._get_task_info())
        
        # Process observation with dimension verification (includes commanded velocity)
        observation = self._process_observation(observation)

        return observation, reward, terminated, truncated, info
        
    def _compute_task_reward(self, obs, base_reward, info, action):
        """
        Compute reward following Prompt 3 specification:
        
        R_total = R_tracking + R_upright + R_effort
        
        Where:
        - R_tracking = exp(-β * ||v_target - v_agent||²)  [Gaussian Kernel, β=5.0]
        - R_upright = +10.0 if is_upright() else 0        [Binary survival reward]
        - R_effort = -0.01 * ||action||²                  [Action penalty]
        
        Additional components for stable humanoid training:
        - Termination penalty for falling
        - Yaw rate tracking (when yaw_rate command is non-zero)
        """
        # ========== STATE EXTRACTION ==========
        height = self.env.unwrapped.data.qpos[2]
        quat = self.env.unwrapped.data.qpos[3:7]  # [w, x, y, z] quaternion
        linear_vel = self.env.unwrapped.data.qvel[0:3]  # World-frame COM velocity
        angular_vel = self.env.unwrapped.data.qvel[3:6]  # Angular velocity
        
        # Agent velocity (per Prompt 3: self.agent_velocity)
        agent_vx = linear_vel[0]
        agent_vy = linear_vel[1]
        agent_yaw_rate = angular_vel[2]  # Yaw rate around Z-axis
        
        # Target velocity from command generator
        v_target = np.array([self.commanded_vx_world, self.commanded_vy_world, self.commanded_yaw_rate])
        v_agent = np.array([agent_vx, agent_vy, agent_yaw_rate])
        
        # ========== R_TRACKING: GAUSSIAN KERNEL (Per Prompt 3) ==========
        # R_tracking = exp(-β * ||v_target - v_agent||²)
        # β is a high constant (e.g., 5.0) per Prompt 3
        velocity_error_sq = np.sum((v_target - v_agent) ** 2)
        R_tracking = np.exp(-self.tracking_beta * velocity_error_sq)
        
        # ========== R_UPRIGHT: BINARY SURVIVAL REWARD (Per Prompt 3) ==========
        # +10.0 if is_upright() is True
        def is_upright() -> bool:
            """Check if humanoid is upright (quaternion w close to 1, reasonable height)."""
            return height >= 1.0 and abs(quat[0]) >= 0.7
        
        R_upright = 10.0 if is_upright() else 0.0
        
        # ========== R_EFFORT: ACTION PENALTY (Per Prompt 3) ==========
        # -0.01 * ||action||²
        R_effort = -0.01 * np.sum(action ** 2)
        
        # ========== R_TOTAL (Per Prompt 3) ==========
        R_total = R_tracking + R_upright + R_effort
        
        # ========== ADDITIONAL: TERMINATION CHECK ==========
        # Not in original prompts but needed for stable training
        terminate = (height < 0.75 or height > 2.0 or abs(quat[0]) < 0.5)
        if terminate:
            R_total -= self.termination_penalty_constant  # Penalty for falling
        
        # ========== TRACK REWARD COMPONENTS ==========
        vel_error = np.sqrt(velocity_error_sq)
        self.reward_history['velocity_tracking'].append(R_tracking)
        self.reward_history['upright'].append(R_upright)
        self.reward_history['control'].append(R_effort)
        self.reward_history['height'].append(height)
        self.reward_history['velocity'].append(vel_error)
        
        # Store for jerk calculation
        self.prev_height = height
        
        # ========== DEBUG LOGGING ==========
        if self.current_step % 500 == 0:
            print(f"Step {self.current_step:4d}: "
                f"h={height:.3f}, "
                f"cmd=({self.commanded_vx_world:.2f}, {self.commanded_vy_world:.2f}, {self.commanded_yaw_rate:.2f}), "
                f"actual=({agent_vx:.2f}, {agent_vy:.2f}, {agent_yaw_rate:.2f}), "
                f"vel_err={vel_error:.3f}, "
                f"R_total={R_total:.2f} [track={R_tracking:.2f}, up={R_upright:.1f}, eff={R_effort:.2f}]")
        
        return R_total, terminate

    # ======== Action and Observation Processing ========

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Process actions with optional smoothing, symmetry, and PD control."""
        if self.enable_action_symmetry:
            half = action.shape[-1] // 2
            if half > 0:
                left = action[:half]
                right = action[-half:]
                mean_lr = 0.5 * (left + right)
                action[:half] = mean_lr
                action[-half:] = mean_lr

        if self.enable_pd_assist and (self.pd_kp > 0.0 or self.pd_kd > 0.0):
            try:
                qpos = self.env.unwrapped.data.qpos[7:7+action.shape[-1]]
                qvel = self.env.unwrapped.data.qvel[6:6+action.shape[-1]]
                pd = (-self.pd_kp * qpos) + (-self.pd_kd * qvel)
                action = np.clip(action + pd, -1.0, 1.0)
            except Exception:
                pass

        if self.enable_action_smoothing:
            tau = np.clip(self.action_smoothing_tau, 0.0, 1.0)
            action = (1.0 - tau) * self.prev_action + tau * action

        low, high = self.env.action_space.low, self.env.action_space.high
        action = np.clip(action, low, high)
        self.prev_action = action.copy()
        return action

    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process observation with correct dimension handling, including commanded velocity."""
        features = [obs]

        # Add COM features if enabled
        if self.include_com:
            try:
                com_pos = self.env.unwrapped.data.subtree_com[0]
                com_vel = self.env.unwrapped.data.cdof_dot[:3] if hasattr(self.env.unwrapped.data, 'cdof_dot') else self.env.unwrapped.data.qvel[:3]
                features.append(np.asarray(com_pos, dtype=np.float32))
                features.append(np.asarray(com_vel, dtype=np.float32))
            except Exception as e:
                print(f"Warning: Failed to add COM features: {e}")

        # Add commanded velocity [vx, vy, yaw_rate] per Prompt 1
        features.append(np.array([
            self.commanded_vx_world, 
            self.commanded_vy_world,
            self.commanded_yaw_rate
        ], dtype=np.float32))

        # Concatenate base + COM features + command
        feat_vec = np.concatenate([np.atleast_1d(f).ravel() for f in features]).astype(np.float32)

        # Feature normalization
        if self.feature_norm:
            feat_vec = np.tanh(feat_vec * 0.1)

        # History stacking
        if self.enable_history:
            self.obs_history.append(feat_vec)
            
            if len(self.obs_history) > self.history_len:
                self.obs_history = self.obs_history[-self.history_len:]
            
            if len(self.obs_history) < self.history_len:
                pad_count = self.history_len - len(self.obs_history)
                padded = [np.zeros_like(feat_vec) for _ in range(pad_count)] + self.obs_history
            else:
                padded = self.obs_history
            
            feat_vec = np.concatenate(padded, axis=0)

        # Handle dimension mismatch gracefully
        current_dim = feat_vec.shape[0]
        if current_dim != self.frozen_obs_dim:
            if current_dim > self.frozen_obs_dim:
                feat_vec = feat_vec[:self.frozen_obs_dim]
            else:
                pad = np.zeros((self.frozen_obs_dim - current_dim,), dtype=np.float32)
                feat_vec = np.concatenate([feat_vec, pad], axis=0)

        return feat_vec
        
    def _get_task_info(self):
        """Get task-specific information including velocity tracking."""
        root_x, root_y = self.env.unwrapped.data.qpos[0:2]
        dist = np.sqrt(root_x**2 + root_y**2)
        
        linear_vel = self.env.unwrapped.data.qvel[0:3]
        vel_error = np.sqrt(
            (linear_vel[0] - self.commanded_vx_world)**2 +
            (linear_vel[1] - self.commanded_vy_world)**2
        )
        
        # Include yaw rate
        angular_vel = self.env.unwrapped.data.qvel[3:6]
        actual_yaw_rate = angular_vel[2]
        
        return {
            'height': self.env.unwrapped.data.qpos[2],
            'distance_from_origin': dist,
            'x_position': root_x,
            'y_position': root_y,
            'x_velocity': linear_vel[0],
            'y_velocity': linear_vel[1],
            'z_velocity': linear_vel[2],
            'quaternion_w': self.env.unwrapped.data.qpos[3],
            # Walking-specific (Per Prompt 1: [vx, vy, yaw_rate])
            'commanded_vx': self.commanded_vx_world,
            'commanded_vy': self.commanded_vy_world,
            'commanded_yaw_rate': self.commanded_yaw_rate,
            'commanded_speed': self.commanded_speed,
            'velocity_error': vel_error,
            'actual_speed': np.sqrt(linear_vel[0]**2 + linear_vel[1]**2),
            'actual_yaw_rate': actual_yaw_rate,
        }
    
    def _apply_push_perturbation(self):
        """Apply periodic push perturbations to train robustness and recovery."""
        if not self.push_enabled:
            return
        
        # Check if we're currently applying a push
        if self.push_countdown > 0:
            # Continue applying current push force
            try:
                # Apply external force to torso (body index 1 typically)
                self.env.unwrapped.data.xfrc_applied[1, :3] = self.current_push_force
            except Exception:
                pass
            self.push_countdown -= 1
            
            # Clear force when push ends
            if self.push_countdown == 0:
                try:
                    self.env.unwrapped.data.xfrc_applied[1, :3] = np.zeros(3)
                except Exception:
                    pass
            return
        
        # Check if it's time for a new push
        if self.current_step > 0 and self.current_step % self.push_interval == 0:
            # Random push direction (horizontal only for stability)
            push_angle = np.random.uniform(0, 2 * np.pi)
            push_mag = np.random.uniform(self.push_magnitude_range[0], self.push_magnitude_range[1])
            
            # Create horizontal push force
            self.current_push_force = np.array([
                push_mag * np.cos(push_angle),
                push_mag * np.sin(push_angle),
                0.0  # No vertical component
            ])
            
            self.push_countdown = self.push_duration
            
            # Debug logging (every 1000 steps)
            if self.current_step % 1000 == 0:
                print(f"  Push applied: magnitude={push_mag:.1f}N, angle={np.degrees(push_angle):.0f}°")
    
    def set_max_speed(self, max_speed: float):
        """Set maximum commanded speed (for curriculum)."""
        self.max_commanded_speed = max_speed
        print(f"Max commanded speed set to: {max_speed:.2f} m/s")
    
    def update_curriculum_ranges(self, new_ranges: list):
        """
        Update command ranges for curriculum learning (Per Prompt 1).
        
        Args:
            new_ranges: List of 3 tuples [(vx_min, vx_max), (vy_min, vy_max), (yaw_min, yaw_max)]
        """
        self.command_generator.update_curriculum_ranges(new_ranges)
    
    def set_fixed_command(self, vx: float, vy: float):
        """Set fixed velocity command (for inference/testing)."""
        self.fixed_command = (vx, vy)
        self.commanded_vx_world = vx
        self.commanded_vy_world = vy
        self.commanded_speed = np.sqrt(vx**2 + vy**2)
        if self.commanded_speed > 0:
            self.commanded_angle = np.arctan2(vy, vx)
        else:
            self.commanded_angle = 0.0
    
    def get_observation_info(self):
        """Helper method to understand observation space"""
        obs, _ = self.reset()
        print("\nObservation Analysis:")
        print(f"Raw observation size: {len(obs)}")
        print(f"Frozen dimension: {self.frozen_obs_dim}")
        print(f"Actual height (qpos[2]): {self.env.unwrapped.data.qpos[2]}")
        print(f"Commanded velocity: ({self.commanded_vx_world:.2f}, {self.commanded_vy_world:.2f})")
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


def make_walking_env(render_mode=None, config=None):
    """Create walking environment with given config."""
    return WalkingEnv(render_mode=render_mode, config=config)


if __name__ == "__main__":
    print("Testing Walking Environment")
    print("=" * 60)
    
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.2,
        'random_height_init': False,
        'max_commanded_speed': 1.0,  # Allow up to 1 m/s
        'velocity_weight': 5.0,
        'reward_caps': {
            'max_height_maintenance_penalty': 15.0,
            'recovery_bonus_scale': 50.0,
            'termination_penalty_constant': 50.0
        }
    }
    
    env = make_walking_env(render_mode=None, config=config)
    
    obs, info = env.reset()
    print(f"\n Reset observation shape: {obs.shape}")
    print(f" Expected frozen dimension: {env.frozen_obs_dim}")
    print(f" Commanded velocity [vx, vy, yaw_rate]: ({info['commanded_vx']:.2f}, {info['commanded_vy']:.2f}, {info['commanded_yaw_rate']:.2f})")
    
    print("\nRunning 200 steps...")
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 50 == 0:
            print(f"Step {step}: height={info['height']:.3f}, "
                  f"vel_err={info['velocity_error']:.3f}, reward={reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
    
    env.close()
    print(f"\n Test completed!")

