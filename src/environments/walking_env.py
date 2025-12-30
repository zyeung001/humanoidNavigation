# walking_env.py
"""
Walking environment for humanoid locomotion
Command-conditioned on desired world velocity (vx_world, vy_world, yaw_rate)
Extends standing environment with velocity tracking rewards

Uses modular RewardCalculator for clean reward computation
Integrates VelocityCommandGenerator for proper command sampling
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium.spaces import Box

# Import modular reward calculator
from src.core.rewards import RewardCalculator, RewardWeights, RewardMetrics
# Import velocity command generator 
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
        
        # FIX 5: Consistency reward for reducing velocity error variance
        self.recent_vel_errors = []
        self.consistency_window = int(self.cfg.get('reward_consistency_window', 100))
        self.consistency_weight = float(self.cfg.get('reward_consistency_weight', 5.0))
        
        # Reward caps from config
        reward_caps = self.cfg.get('reward_caps', {})
        self.max_height_maintenance_penalty = reward_caps.get('max_height_maintenance_penalty', 15.0)
        self.recovery_bonus_scale = reward_caps.get('recovery_bonus_scale', 50.0)
        self.termination_penalty_constant = reward_caps.get('termination_penalty_constant', 50.0)
        
        # ========== WALKING-SPECIFIC CONFIG ==========
        self.velocity_weight = float(self.cfg.get('velocity_weight', 5.0))
        self.max_commanded_speed = float(self.cfg.get('max_commanded_speed', 0.0))  # Curriculum controls this
        self.fixed_command = self.cfg.get('fixed_command', None)  # (vx, vy, yaw_rate) tuple for inference
        
        # Yaw rate tracking 
        self.include_yaw_rate = bool(self.cfg.get('include_yaw_rate', True))
        self.max_yaw_rate = float(self.cfg.get('max_yaw_rate', 1.0))  # rad/s
        self.yaw_rate_weight = float(self.cfg.get('yaw_rate_weight', 3.0))
        
        # Current commanded velocity (set in reset)
        self.commanded_vx_world = 0.0
        self.commanded_vy_world = 0.0
        self.commanded_yaw_rate = 0.0  # Added yaw_rate
        self.commanded_speed = 0.0
        self.commanded_angle = 0.0
        
        # ========== VELOCITY COMMAND GENERATOR ==========
        # Use the VelocityCommandGenerator for proper command sampling
        self.use_command_generator = bool(self.cfg.get('use_command_generator', True))
        if self.use_command_generator:
            self.command_generator = VelocityCommandGenerator(
                vx_range=(0.0, self.max_commanded_speed),  # Curriculum controls upper bound
                vy_range=(-self.max_commanded_speed * 0.5, self.max_commanded_speed * 0.5),
                yaw_rate_range=(-self.max_yaw_rate, self.max_yaw_rate),
                switch_interval_range=tuple(self.cfg.get('command_switch_interval', [2.0, 5.0])),
                stop_probability=float(self.cfg.get('stop_probability', 0.15)),
            )
        else:
            self.command_generator = None
        
        # Simulation dt (from Mujoco)
        self.dt = self.env.unwrapped.model.opt.timestep * self.env.unwrapped.frame_skip
        
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
        command_dim = 3 if self.include_yaw_rate else 2  # vx, vy, [yaw_rate]
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
        
        # FIX 5: Clear consistency tracking
        self.recent_vel_errors = []
        
        # Clear reward history
        for key in self.reward_history:
            self.reward_history[key] = []
        
        # Reset reward calculator state
        self.reward_calculator.reset()
        
        # ========== SET COMMANDED VELOCITY ==========
        if self.fixed_command is not None:
            # Fixed command for inference/testing
            self.commanded_vx_world = float(self.fixed_command[0])
            self.commanded_vy_world = float(self.fixed_command[1])
            self.commanded_yaw_rate = float(self.fixed_command[2]) if len(self.fixed_command) > 2 else 0.0
            self.commanded_speed = np.sqrt(self.commanded_vx_world**2 + self.commanded_vy_world**2)
            if self.commanded_speed > 0:
                self.commanded_angle = np.arctan2(self.commanded_vy_world, self.commanded_vx_world)
            else:
                self.commanded_angle = 0.0
        elif self.use_command_generator and self.command_generator is not None:
            # Use VelocityCommandGenerator 
            # Update curriculum ranges based on current max speed
            self.command_generator.update_curriculum_ranges([
                (0.0, self.max_commanded_speed),
                (-self.max_commanded_speed * 0.5, self.max_commanded_speed * 0.5),
                (-self.max_yaw_rate, self.max_yaw_rate)
            ])
            # Force new command at episode start
            command = self.command_generator.force_new_command()
            self.commanded_vx_world = float(command[0])
            self.commanded_vy_world = float(command[1])
            self.commanded_yaw_rate = float(command[2]) if self.include_yaw_rate else 0.0
            self.commanded_speed = np.sqrt(self.commanded_vx_world**2 + self.commanded_vy_world**2)
            if self.commanded_speed > 0:
                self.commanded_angle = np.arctan2(self.commanded_vy_world, self.commanded_vx_world)
            else:
                self.commanded_angle = 0.0
        else:
            # Fallback: Sample random velocity command for training
            self.commanded_speed = np.random.uniform(0.0, self.max_commanded_speed)
            self.commanded_angle = np.random.uniform(0.0, 2 * np.pi)
            self.commanded_vx_world = self.commanded_speed * np.cos(self.commanded_angle)
            self.commanded_vy_world = self.commanded_speed * np.sin(self.commanded_angle)
            self.commanded_yaw_rate = np.random.uniform(-self.max_yaw_rate, self.max_yaw_rate) if self.include_yaw_rate else 0.0
        
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
        
        # Add walking info
        info['commanded_vx'] = self.commanded_vx_world
        info['commanded_vy'] = self.commanded_vy_world
        info['commanded_yaw_rate'] = self.commanded_yaw_rate
        info['commanded_speed'] = self.commanded_speed

        return observation, info
    
    def step(self, action):
        # Action preprocessing
        proc_action = self._process_action(np.asarray(action, dtype=np.float32))
        
        # ========== UPDATE COMMAND  ==========
        if self.use_command_generator and self.command_generator is not None and self.fixed_command is None:
            command = self.command_generator.get_command(self.dt)
            self.commanded_vx_world = float(command[0])
            self.commanded_vy_world = float(command[1])
            self.commanded_yaw_rate = float(command[2]) if self.include_yaw_rate else 0.0
            self.commanded_speed = np.sqrt(self.commanded_vx_world**2 + self.commanded_vy_world**2)
            if self.commanded_speed > 0:
                self.commanded_angle = np.arctan2(self.commanded_vy_world, self.commanded_vx_world)
        
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
        Compute walking reward with VELOCITY TRACKING AS DOMINANT OBJECTIVE.
        
        FIX: The agent was stuck at Stage 0 because survival rewards (~40-50/step) 
        dominated velocity tracking rewards (~5-10/step). The agent found a local 
        optimum: "stand still and survive = high reward".
        
        Key changes:
        1. Velocity tracking reward is now 3-4x larger (dominant component)
        2. Explicit standing penalty when walking is commanded (-15/step)
        3. Survival rewards are CONDITIONAL on velocity tracking performance
        4. Progress bonus for ANY movement in right direction
        """
        # ========== STATE EXTRACTION ==========
        height = self.env.unwrapped.data.qpos[2]
        quat = self.env.unwrapped.data.qpos[3:7]  # [w, x, y, z] quaternion
        linear_vel = self.env.unwrapped.data.qvel[0:3]  # World-frame COM velocity
        angular_vel = self.env.unwrapped.data.qvel[3:6]
        joint_vel = self.env.unwrapped.data.qvel[6:]
        
        com_vel_x, com_vel_y = linear_vel[0], linear_vel[1]
        actual_speed = np.sqrt(com_vel_x**2 + com_vel_y**2)
        height_error = abs(height - self.base_target_height)
        
        # ========== MODULAR REWARD CALCULATOR ==========
        v_target = np.array([self.commanded_vx_world, self.commanded_vy_world])
        v_actual = np.array([com_vel_x, com_vel_y])
        
        reward_metrics = self.reward_calculator.compute(
            v_target=v_target,
            v_actual=v_actual,
            height=height,
            quaternion=quat,
            action=action,
            prev_action=self.prev_action if hasattr(self, 'prev_action') else action,
        )
        
        vel_error = reward_metrics.velocity_error
        
        # ========== FIX 1: ENHANCED VELOCITY TRACKING (DOMINANT REWARD) ==========
        # Use a WIDER Gaussian (bandwidth 2.0 instead of 4.0) for better gradient
        # AND multiply by a much larger weight
        tracking_bandwidth = float(self.cfg.get('reward_tracking_bandwidth', 2.0))
        tracking_weight = float(self.cfg.get('reward_tracking_weight', 25.0))  # INCREASED
        
        # Core tracking reward with lenient Gaussian
        velocity_tracking_reward = tracking_weight * np.exp(-tracking_bandwidth * vel_error**2)
        
        # ========== FIX 2: PROGRESS BONUS (reward ANY movement toward target) ==========
        progress_bonus = 0.0
        if self.commanded_speed > 0.05:
            # Project actual velocity onto commanded direction
            if self.commanded_speed > 0:
                direction_cmd = np.array([self.commanded_vx_world, self.commanded_vy_world]) / self.commanded_speed
            else:
                direction_cmd = np.array([1.0, 0.0])
            
            projected_speed = np.dot(v_actual, direction_cmd)
            
            # Reward for moving in RIGHT direction (even if speed is wrong)
            if projected_speed > 0:
                # Proportional to how much of the target we achieved
                speed_ratio = min(1.0, projected_speed / self.commanded_speed)
                progress_bonus = 15.0 * speed_ratio  # Up to +15 for matching speed
                
                # Extra bonus for getting close to target speed
                if abs(projected_speed - self.commanded_speed) < 0.1:
                    progress_bonus += 5.0
            else:
                # Moving in WRONG direction - small penalty
                progress_bonus = -3.0
        
        # ========== FIX 3: STANDING PENALTY (breaks "stand still" exploit) ==========
        standing_penalty = 0.0
        if self.commanded_speed > 0.1:
            if actual_speed < 0.05:
                # SIGNIFICANT per-step penalty for standing when walking commanded
                standing_penalty = -15.0
            elif actual_speed < 0.1:
                standing_penalty = -8.0
            elif actual_speed < self.commanded_speed * 0.3:
                # Not trying hard enough
                standing_penalty = -3.0
        
        # ========== YAW RATE TRACKING REWARD ==========
        actual_yaw_rate = angular_vel[2]
        yaw_error = abs(actual_yaw_rate - self.commanded_yaw_rate)
        yaw_tracking_reward = 0.0
        if self.include_yaw_rate:
            yaw_tracking_reward = self.yaw_rate_weight * np.exp(-3.0 * yaw_error**2)
        
        # ========== FIX 4: CONDITIONAL HEIGHT REWARD ==========
        # Height reward is REDUCED when not tracking velocity properly
        # This prevents exploiting survival rewards while ignoring velocity
        base_height_reward = 0.0
        if height < 1.0:
            base_height_reward = -30.0 + 25.0 * np.clip(height / 1.0, 0.0, 1.0)
        elif height < 1.2:
            base_height_reward = -5.0 + 10.0 * (height - 1.0) / 0.2
        elif height < 1.35:
            base_height_reward = 5.0 + 5.0 * (height - 1.2) / 0.15
        elif height < 1.5:
            base_height_reward = 10.0
        else:
            base_height_reward = 8.0 - 5.0 * np.clip((height - 1.5) / 0.2, 0.0, 1.0)
        
        # CONDITION: Scale height reward by velocity tracking quality
        if self.commanded_speed > 0.1:
            # If velocity error is high, reduce height reward
            tracking_quality = np.exp(-2.0 * vel_error**2)  # 0-1 scale
            height_reward = base_height_reward * (0.3 + 0.7 * tracking_quality)
        else:
            height_reward = base_height_reward
        
        # ========== CONDITIONAL UPRIGHT REWARD ==========
        upright_error = 1.0 - abs(quat[0])
        if height >= 1.15:
            base_upright = 5.0 * np.exp(-6.0 * upright_error**2)
        elif height >= 1.0:
            base_upright = 3.0 * np.exp(-6.0 * upright_error**2)
        else:
            base_upright = 0.0
        
        # Condition on velocity tracking when walking is commanded
        if self.commanded_speed > 0.1:
            tracking_quality = np.exp(-2.0 * vel_error**2)
            upright_reward = base_upright * (0.4 + 0.6 * tracking_quality)
        else:
            upright_reward = base_upright
        
        # ========== STABILITY REWARD (smaller, conditional) ==========
        angular_momentum = np.sum(np.square(angular_vel))
        if height >= 1.2:
            stability_reward = 2.0 * np.exp(-1.5 * angular_momentum)
        elif height >= 1.0:
            stability_reward = 1.0 * np.exp(-1.5 * angular_momentum)
        else:
            stability_reward = 0.0
        
        # ========== SMOOTHNESS REWARD (reduced) ==========
        joint_velocity_magnitude = np.sum(np.square(joint_vel))
        smoothness_reward = 1.0 * np.exp(-0.08 * joint_velocity_magnitude)
        
        # ========== CONTROL COST ==========
        jerk_penalty = reward_metrics.jerk_penalty
        control_cost = -0.003 * np.sum(np.square(action))
        
        # ========== HEIGHT MAINTENANCE (reduced) ==========
        height_velocity = height - self.prev_height if hasattr(self, 'prev_height') else 0.0
        height_maintenance = 0.0
        if height_velocity < -0.005:
            height_maintenance = -50.0 * np.clip(abs(height_velocity), 0.0, 0.1)
        elif abs(height_velocity) < 0.003:
            height_maintenance = 1.0
        
        # ========== RECOVERY BONUS (reduced) ==========
        recovery_bonus = 0.0
        if height < 1.0 and height_velocity > 0.01:
            recovery_scale = (1.0 - height) / 0.4
            recovery_bonus = 25.0 * height_velocity * recovery_scale
        
        # ========== WALKING BONUS (movement shaping) ==========
        walking_bonus = 0.0
        if self.commanded_speed > 0.1:
            projected_speed = (
                com_vel_x * np.cos(self.commanded_angle) + 
                com_vel_y * np.sin(self.commanded_angle)
            )
            forward_progress = max(0.0, projected_speed)
            walking_bonus = 5.0 * np.tanh(forward_progress * 3.0)
        
        # Velocity penalty when standing is commanded
        velocity_penalty = 0.0
        if self.commanded_speed < 0.1:
            speed = np.linalg.norm(linear_vel[:2])
            velocity_penalty = -2.0 * np.clip(speed - 0.2, 0.0, 2.0)
        
        # ========== CONSISTENCY BONUS ==========
        consistency_bonus = 0.0
        self.recent_vel_errors.append(vel_error)
        if len(self.recent_vel_errors) > self.consistency_window:
            self.recent_vel_errors = self.recent_vel_errors[-self.consistency_window:]
        
        if len(self.recent_vel_errors) >= 50:
            recent_std = np.std(self.recent_vel_errors[-50:])
            recent_mean = np.mean(self.recent_vel_errors[-50:])
            
            if recent_mean < 0.5 and recent_std < 0.25:
                consistency_bonus = self.consistency_weight * (1.0 - recent_std / 0.25) * (1.0 - recent_mean / 0.5)
        
        # ========== DURATION BONUS (conditional on tracking) ==========
        sustained_bonus = 0.0
        if self.current_step > 0 and self.current_step % 100 == 0:
            if height_error < 0.25 and upright_error < 0.2 and height >= 1.1:
                # Base survival bonus
                sustained_bonus = 5.0
                # EXTRA bonus only if tracking velocity well
                if self.commanded_speed > 0.1 and vel_error < 0.4:
                    sustained_bonus += 20.0  # Significant bonus for tracking while surviving
                elif self.commanded_speed > 0.1 and vel_error < 0.6:
                    sustained_bonus += 10.0
                elif self.commanded_speed < 0.1:
                    sustained_bonus += 3.0  # Small bonus for standing still correctly
                
                if self.current_step >= 500:
                    sustained_bonus += 3.0
                if self.current_step >= 1000:
                    sustained_bonus += 3.0
        
        # ========== TERMINATION ==========
        terminate = (height < 0.75 or height > 2.0 or abs(quat[0]) < 0.6)
        termination_penalty = 0.0
        if terminate:
            if height < 0.75 or abs(quat[0]) < 0.6:
                termination_penalty = -self.termination_penalty_constant
            else:
                termination_penalty = -self.termination_penalty_constant * 0.5
        
        # ========== TOTAL REWARD ==========
        # NEW STRUCTURE: Velocity tracking is now ~60-70% of positive rewards
        total_reward = (
            # PRIMARY: Velocity tracking (dominant)
            velocity_tracking_reward +  # Up to +25 per step
            progress_bonus +            # Up to +20 per step  
            standing_penalty +          # -15 per step if standing when walking commanded
            yaw_tracking_reward +       # Up to +3 per step
            
            # SECONDARY: Survival (conditional on tracking)
            height_reward +             # Up to +10 (reduced from +30)
            upright_reward +            # Up to +5 (reduced from +10)
            stability_reward +          # Up to +2 (reduced from +4)
            smoothness_reward +         # Up to +1 (reduced from +2)
            
            # PENALTIES
            jerk_penalty +
            control_cost +
            height_maintenance +
            velocity_penalty +
            
            # BONUSES
            recovery_bonus +
            walking_bonus +
            sustained_bonus +
            consistency_bonus +
            termination_penalty
        )
        
        # Update state
        self.prev_height = height
        
        # ========== TRACK REWARD COMPONENTS ==========
        self.reward_history['height'].append(height_reward)
        self.reward_history['upright'].append(upright_reward)
        self.reward_history['velocity'].append(stability_reward)
        self.reward_history['control'].append(control_cost)
        self.reward_history['velocity_tracking'].append(velocity_tracking_reward + progress_bonus)
        self.reward_history['jerk'].append(jerk_penalty)
        
        # ========== DETAILED DEBUG LOGGING ==========
        if self.current_step % 500 == 0:
            yaw_info = f", yaw_err={yaw_error:.3f}" if self.include_yaw_rate else ""
            print(f"Step {self.current_step:4d}: "
                f"h={height:.3f}, cmd=({self.commanded_vx_world:.2f},{self.commanded_vy_world:.2f}), "
                f"actual=({com_vel_x:.2f},{com_vel_y:.2f}){yaw_info}, "
                f"vel_err={vel_error:.3f}, r={total_reward:6.1f} "
                f"[track={velocity_tracking_reward:.1f}, prog={progress_bonus:.1f}, stand_pen={standing_penalty:.1f}]")
        
        return total_reward, terminate

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

        # Add commanded velocity
        # Include yaw_rate if enabled 
        if self.include_yaw_rate:
            features.append(np.array([self.commanded_vx_world, self.commanded_vy_world, self.commanded_yaw_rate], dtype=np.float32))
        else:
            features.append(np.array([self.commanded_vx_world, self.commanded_vy_world], dtype=np.float32))

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
        angular_vel = self.env.unwrapped.data.qvel[3:6]
        
        # Velocity error (vx, vy)
        vel_error = np.sqrt(
            (linear_vel[0] - self.commanded_vx_world)**2 +
            (linear_vel[1] - self.commanded_vy_world)**2
        )
        
        # Yaw rate error (angular velocity around Z axis)
        actual_yaw_rate = angular_vel[2]  # Rotation around z-axis
        yaw_rate_error = abs(actual_yaw_rate - self.commanded_yaw_rate)
        
        return {
            'height': self.env.unwrapped.data.qpos[2],
            'distance_from_origin': dist,
            'x_position': root_x,
            'y_position': root_y,
            'x_velocity': linear_vel[0],
            'y_velocity': linear_vel[1],
            'z_velocity': linear_vel[2],
            'quaternion_w': self.env.unwrapped.data.qpos[3],
            # Walking-specific
            'commanded_vx': self.commanded_vx_world,
            'commanded_vy': self.commanded_vy_world,
            'commanded_yaw_rate': self.commanded_yaw_rate,
            'commanded_speed': self.commanded_speed,
            'velocity_error': vel_error,
            'velocity_error_x': abs(linear_vel[0] - self.commanded_vx_world),
            'velocity_error_y': abs(linear_vel[1] - self.commanded_vy_world),
            'yaw_rate_error': yaw_rate_error,
            'actual_yaw_rate': actual_yaw_rate,
            'actual_speed': np.sqrt(linear_vel[0]**2 + linear_vel[1]**2),
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
    print(f" Commanded velocity: ({info['commanded_vx']:.2f}, {info['commanded_vy']:.2f})")
    
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

