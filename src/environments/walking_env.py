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
from typing import Optional
from gymnasium.spaces import Box

# Import modular reward calculator
from src.core.rewards import RewardCalculator, RewardWeights
# Import velocity command generator 
from src.core.command_generator import VelocityCommandGenerator


class WalkingEnv(gym.Wrapper):
    """
    Walking environment that conditions policy on commanded world-frame velocity.
    
    Observations include:
    - Base humanoid observations (365 dims)
    - COM features if enabled (+6 dims)  
    - Commanded velocity (vx_world, vy_world) (+2 dims)
    - History stacking (x4)
    
    Total: (365 + 6 + 2) x 4 = 1492 dims
    """
    
    def __init__(self, render_mode: Optional[str] = None, config=None):
        env_id = "Humanoid-v5"
        print(f"Using {env_id} for walking task")
        
        # Create base environment
        xml_file = (config or {}).get('xml_file', None)
        make_kwargs = dict(
            render_mode=render_mode,
            exclude_current_positions_from_observation=False,  # Adds +2 dims (x,y)
        )
        if xml_file is not None:
            make_kwargs['xml_file'] = xml_file
        env = gym.make(env_id, **make_kwargs)
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
        
        # ========== FEET AIR TIME TRACKING (enables rotation) ==========
        # Robot can't rotate with feet planted — must lift feet to pivot.
        # Reward periodic foot lifting (SOTA: Isaac Lab, Walk These Ways, legged_gym).
        self.feet_air_time_weight = float(self.cfg.get('feet_air_time_weight', 1.0))
        self.feet_air_time = np.zeros(2)  # [right_foot, left_foot] air duration
        self.feet_contact_prev = np.ones(2, dtype=bool)  # Previous contact state
        self.foot_body_ids = [6, 9]  # right_foot=body6, left_foot=body9
        self.min_air_time = 0.15  # Humanoid-v5 gait has ~0.1-0.3s foot air time.
                                  # SOTA uses 0.4s but that's for quadrupeds/larger humanoids.
                                  # 0.15s filters shuffling while matching this humanoid's gait.
        self.contact_force_threshold = 5.0  # Force threshold for contact detection (N)

        # Arm posture penalty (prevent chicken-wing arms, allow natural swing)
        self.arm_posture_weight = float(self.cfg.get('reward_arm_posture_weight', 0.0))
        self.arm_joint_indices = slice(18, 24)  # qpos indices for 6 arm joints
        # FIX: np.zeros was WRONG — at qpos=0 arms point forward (0.18m ahead of shoulder)
        # These angles place arms hanging straight down at sides (verified via FK)
        self.arm_ref_angles = np.array([0.81, -0.97, -0.85, -0.81, 0.97, -0.85], dtype=np.float32)
        # Deadzone: deviations within this range are NOT penalized (allows natural arm swing)
        self.arm_deadzone = float(self.cfg.get('arm_posture_deadzone', 0.3))
        # Cap: maximum per-step penalty magnitude (prevents arm penalty from dominating tracking)
        self.arm_penalty_cap = float(self.cfg.get('arm_penalty_cap', 1.0))

        # FIX 5: Consistency reward for reducing velocity error variance
        self.recent_vel_errors = []
        self.consistency_window = int(self.cfg.get('reward_consistency_window', 100))
        self.consistency_weight = float(self.cfg.get('reward_consistency_weight', 5.0))
        
        # FIX 6: Relaxed termination with grace period for walking learning
        # Issue: Height < 0.75m terminates immediately, not giving agent enough
        # experience to learn stepping patterns. Solution: Add grace period.
        self.termination_grace_period = int(self.cfg.get('termination_grace_period', 100))
        self.termination_height_threshold = float(self.cfg.get('termination_height_threshold', 0.70))  # Raised: prevents crouching exploit
        self.termination_height_recovery_window = int(self.cfg.get('termination_height_recovery_window', 80))
        self.low_height_steps = 0  # Counter for consecutive low height steps
        
        # Early episode protection - don't terminate for first N steps
        self.early_termination_protection = int(self.cfg.get('early_termination_protection', 150))
        
        # Reward caps from config
        reward_caps = self.cfg.get('reward_caps', {})
        self.max_height_maintenance_penalty = reward_caps.get('max_height_maintenance_penalty', 15.0)
        self.recovery_bonus_scale = reward_caps.get('recovery_bonus_scale', 50.0)
        self.termination_penalty_constant = reward_caps.get('termination_penalty_constant', 25.0)
        
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

        # ========== NEW OBSERVATION STRUCTURE ==========
        # FIX: Command signal was invisible (buried in history, crushed by normalization)
        # Solution: Body features only in history, command block appended ONCE

        # Body features (stacked in history)
        extra_dim = 6 if self.include_com else 0  # COM pos (3) + COM vel (3)
        self.body_dim_per_frame = base_obs_dim + extra_dim  # 371

        # Command block (appended ONCE, not stacked)
        # [vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual, err_vx, err_vy, err_speed, err_angle]
        self.command_block_dim = 9

        if self.enable_history:
            # Body features stacked, command block appended once
            stacked_body_dim = self.body_dim_per_frame * self.history_len  # 1484
            total_dim = stacked_body_dim + self.command_block_dim  # 1493
        else:
            total_dim = self.body_dim_per_frame + self.command_block_dim

        # FREEZE observation dimension now (before VecNormalize initialization)
        self.frozen_obs_dim = total_dim

        # Declare observation space with frozen dimension
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.frozen_obs_dim,),
            dtype=np.float32
        )

        print("Walking environment observation space configuration (NEW):")
        print(f"  Base from env.observation_space: {base_obs_from_space}")
        print(f"  + Position inclusion adjustment: +15 -> {base_obs_dim}")
        print(f"  + COM features: {extra_dim}")
        print(f"  = Body dim per frame: {self.body_dim_per_frame}")
        print(f"  x History stack: {self.history_len if self.enable_history else 1}")
        print(f"  = Stacked body dim: {self.body_dim_per_frame * self.history_len if self.enable_history else self.body_dim_per_frame}")
        print(f"  + Command block (ONCE): {self.command_block_dim}")
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
        
        # Reset feet air time tracking
        self.feet_air_time[:] = 0.0
        self.feet_contact_prev[:] = True

        # Clear push perturbation state
        self.push_countdown = 0
        self.current_push_force = np.zeros(3)
        try:
            self.env.unwrapped.data.xfrc_applied[:] = 0.0
        except Exception:
            pass
        
        # FIX 5: Clear consistency tracking
        self.recent_vel_errors = []
        
        # FIX 6: Reset grace period counter
        self.low_height_steps = 0
        
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
            # Save original values on first call, then restore before randomizing
            # Without this, masses/frictions drift multiplicatively across resets
            if not hasattr(self, '_orig_body_mass'):
                self._orig_body_mass = self.env.unwrapped.model.body_mass.copy()
                self._orig_geom_friction = self.env.unwrapped.model.geom_friction.copy()

            self.env.unwrapped.model.body_mass[:] = self._orig_body_mass * np.random.uniform(
                self.rand_mass_range[0], self.rand_mass_range[1],
                size=self._orig_body_mass.shape
            )

            self.env.unwrapped.model.geom_friction[:, 0] = self._orig_geom_friction[:, 0] * np.random.uniform(
                self.rand_friction_range[0], self.rand_friction_range[1],
                size=self._orig_geom_friction.shape[0]
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
        if self.fixed_command is not None:
            # Read fixed command every step (e.g. navigation controller updates it)
            self.commanded_vx_world = float(self.fixed_command[0])
            self.commanded_vy_world = float(self.fixed_command[1])
            self.commanded_yaw_rate = float(self.fixed_command[2]) if len(self.fixed_command) > 2 else 0.0
            self.commanded_speed = np.sqrt(self.commanded_vx_world**2 + self.commanded_vy_world**2)
            if self.commanded_speed > 0:
                self.commanded_angle = np.arctan2(self.commanded_vy_world, self.commanded_vx_world)
        elif self.use_command_generator and self.command_generator is not None:
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

        com_vel_x, com_vel_y = linear_vel[0], linear_vel[1]
        actual_speed = np.sqrt(com_vel_x**2 + com_vel_y**2)

        # Torso "up" direction: how vertical the humanoid is (1.0 = upright, 0 = horizontal).
        # This is yaw-invariant — pure z-rotation gives up_z=1.0 regardless of heading.
        # The old abs(quat[0]) check conflated yaw with tilt, causing false termination
        # when the humanoid faces any direction other than +x.
        up_z = 1.0 - 2.0 * (quat[1]**2 + quat[2]**2)
        
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
        
        # ========== VELOCITY TRACKING (DOMINANT SIGNAL) ==========
        tracking_bandwidth = float(self.cfg.get('reward_tracking_bandwidth', 5.0))
        tracking_weight = float(self.cfg.get('reward_tracking_weight', 5.0))

        # Core tracking reward with Gaussian kernel
        velocity_tracking_reward = tracking_weight * np.exp(-tracking_bandwidth * vel_error**2)

        # ========== PROGRESS BONUS (continuous, no threshold discontinuities) ==========
        progress_bonus = 0.0
        if self.commanded_speed > 0.05:
            # Project actual velocity onto commanded direction
            if self.commanded_speed > 0:
                direction_cmd = np.array([self.commanded_vx_world, self.commanded_vy_world]) / self.commanded_speed
            else:
                direction_cmd = np.array([1.0, 0.0])

            projected_speed = np.dot(v_actual, direction_cmd)

            # Continuous progress bonus: smooth ratio, no thresholds
            speed_ratio = np.clip(projected_speed / self.commanded_speed, -0.5, 1.5)
            progress_bonus = 1.5 * speed_ratio

        # ========== STANDING PENALTY (REMOVED — tracking reward handles this) ==========
        standing_penalty = 0.0
        
        # ========== YAW RATE TRACKING REWARD ==========
        # Gaussian yaw tracking. sigma=0.5 (bandwidth=4) is wider than SOTA
        # sigma=0.25 because we're fine-tuning a model with large yaw errors
        # (0.5-1.0 rad/s). SOTA uses tight sigma because yaw is trained from
        # scratch when errors are small. Tighten sigma once errors drop below 0.3.
        actual_yaw_rate = angular_vel[2]
        yaw_error = abs(actual_yaw_rate - self.commanded_yaw_rate)
        yaw_tracking_reward = 0.0
        if self.include_yaw_rate:
            yaw_tracking_reward = self.yaw_rate_weight * np.exp(-4.0 * yaw_error**2)

        # ========== FEET AIR TIME REWARD (enables rotation) ==========
        # Robot must lift feet to rotate — friction prevents rotation with feet planted.
        # CONTINUOUS: per-step reward for each foot currently in the air past min_air_time.
        # This gives dense gradient every step (vs sparse landing-only which fires once
        # per gait cycle and was invisible in training — air=0.0 in 99% of step logs).
        feet_air_time_reward = 0.0
        if self.feet_air_time_weight > 0:
            for i, body_id in enumerate(self.foot_body_ids):
                contact_force = np.linalg.norm(self.env.unwrapped.data.cfrc_ext[body_id])
                in_contact = contact_force > self.contact_force_threshold
                if in_contact:
                    self.feet_air_time[i] = 0.0
                else:
                    self.feet_air_time[i] += self.dt
                    # Continuous reward: scales with air time, capped at 1.0
                    if self.feet_air_time[i] > self.min_air_time:
                        air_bonus = min(self.feet_air_time[i] - self.min_air_time, 0.3) / 0.3
                        feet_air_time_reward += self.feet_air_time_weight * air_bonus
                self.feet_contact_prev[i] = in_contact

        # ========== ANTI-DRIFT PENALTY (during turn-in-place) ==========
        # When only yaw is commanded (vel~0), penalize any linear velocity.
        # Prevents agent from drifting instead of rotating in place.
        anti_drift_penalty = 0.0
        if self.commanded_speed < 0.05 and abs(self.commanded_yaw_rate) > 0.05:
            anti_drift_penalty = -1.0 * min(actual_speed, 0.5)
        
        # ========== HEIGHT REWARD (halved — survival must not compete with tracking) ==========
        base_height_reward = 0.0
        if height < 1.0:
            base_height_reward = -1.5 + 1.25 * np.clip(height / 1.0, 0.0, 1.0)
        elif height < 1.2:
            base_height_reward = -0.25 + 0.5 * (height - 1.0) / 0.2
        elif height < 1.35:
            base_height_reward = 0.25 + 0.25 * (height - 1.2) / 0.15
        elif height < 1.5:
            base_height_reward = 0.5
        else:
            base_height_reward = 0.4 - 0.25 * np.clip((height - 1.5) / 0.2, 0.0, 1.0)
        height_reward = base_height_reward

        # ========== UPRIGHT REWARD (halved — survival must not compete with tracking) ==========
        upright_error = 1.0 - max(up_z, 0.0)  # 0 when upright, 1 when horizontal
        if height >= 1.15:
            base_upright = 0.25 * np.exp(-6.0 * upright_error**2)
        elif height >= 1.0:
            base_upright = 0.15 * np.exp(-6.0 * upright_error**2)
        else:
            base_upright = 0.0
        upright_reward = base_upright

        # ========== ZEROED COMPONENTS (kept for WandB logging compatibility) ==========
        stability_reward = 0.0    # Penalizes angular momentum — walking REQUIRES this
        smoothness_reward = 0.0   # Penalizes joint velocity — walking REQUIRES this

        # ========== CONTROL COST ==========
        jerk_penalty = 0.0        # Zeroed — penalizes gait changes
        control_cost = -0.0003 * np.sum(np.square(action))

        # ========== ARM POSTURE PENALTY (keep arms at sides for sim-to-real) ==========
        # Deadzone allows natural arm swing without penalty.
        # Cap prevents arm penalty from ever dominating the tracking signal.
        arm_posture_penalty = 0.0
        if self.arm_posture_weight > 0:
            arm_qpos = self.env.unwrapped.data.qpos[self.arm_joint_indices]
            arm_deviations = np.abs(arm_qpos - self.arm_ref_angles)
            # Only penalize deviation beyond the deadzone
            penalized = np.maximum(arm_deviations - self.arm_deadzone, 0.0)
            raw_penalty = -self.arm_posture_weight * np.sum(penalized**2)
            arm_posture_penalty = max(raw_penalty, -self.arm_penalty_cap)

        # ========== ZEROED COMPONENTS (kept for WandB logging compatibility) ==========
        height_maintenance = 0.0   # Zeroed — redundant with height reward
        recovery_bonus = 0.0       # Zeroed — redundant with height reward
        walking_bonus = 0.0        # Zeroed — redundant with progress_bonus
        velocity_penalty = 0.0     # Zeroed — tracking reward handles this
        consistency_bonus = 0.0    # Zeroed — noisy, creates conflicting advantages
        sustained_bonus = 0.0      # Zeroed — noisy sparse bonuses
        
        # ========== TERMINATION (FIX 6: Grace period for walking) ==========
        # Issue: Immediate termination at height < 0.75 prevents learning stepping
        # Solution: Allow recovery time, protect early episode, use lower threshold
        
        termination_penalty = 0.0
        terminate = False
        
        # Hard termination conditions (always apply)
        if height > 2.0:
            terminate = True
            termination_penalty = -self.termination_penalty_constant * 0.5
        elif up_z < 0.0:  # Severely tilted (torso past horizontal, yaw-invariant)
            terminate = True
            termination_penalty = -self.termination_penalty_constant
        
        # Soft termination with grace period (height-based)
        elif height < self.termination_height_threshold:  # Default 0.70 (lowered from 0.75)
            self.low_height_steps += 1
            
            # Early episode protection - don't terminate in first N steps
            if self.current_step < self.early_termination_protection:
                terminate = False
                # Still penalize but allow recovery (scaled down 10x)
                termination_penalty = -0.5  # FIX: was -5.0

            # Grace period - allow recovery attempts
            elif self.low_height_steps >= self.termination_height_recovery_window:
                terminate = True
                termination_penalty = -self.termination_penalty_constant
            else:
                # Penalize but don't terminate yet (scaled down 10x)
                termination_penalty = -0.05 * min(self.low_height_steps, 30)  # FIX: was -0.5
        else:
            # Height is OK, reset counter
            self.low_height_steps = 0

            # Mild orientation penalty (scaled down 10x)
            if up_z < 0.5:  # Torso tilted > 60° from vertical
                termination_penalty = -0.5  # FIX: was -5.0
        
        # ========== TOTAL REWARD ==========
        # SOTA-aligned: velocity tracking + yaw tracking + feet air time.
        # Zeroed components kept in sum for WandB logging compatibility.
        total_reward = (
            # PRIMARY: Velocity tracking (dominant)
            velocity_tracking_reward +  # Up to +5.0 per step
            progress_bonus +            # Up to +2.25 per step (continuous)
            yaw_tracking_reward +       # Up to +3.0 per step (Gaussian, sigma=0.5)

            # LOCOMOTION: Feet air time (enables rotation by rewarding foot lifting)
            feet_air_time_reward +      # Up to +1.0 per step (continuous, both feet airborne)

            # SECONDARY: Survival (halved — must not compete with tracking)
            height_reward +             # Up to +0.5
            upright_reward +            # Up to +0.25

            # PENALTIES
            control_cost +              # Small action regularization
            arm_posture_penalty +       # Curriculum-gated (0 in Stage 0)
            anti_drift_penalty +        # Penalize drift during turn-in-place

            # ZEROED (kept for WandB compatibility)
            standing_penalty +          # 0.0
            stability_reward +          # 0.0
            smoothness_reward +         # 0.0
            jerk_penalty +              # 0.0
            height_maintenance +        # 0.0
            velocity_penalty +          # 0.0
            recovery_bonus +            # 0.0
            walking_bonus +             # 0.0
            sustained_bonus +           # 0.0
            consistency_bonus +         # 0.0
            termination_penalty
        )
        
        # Update state
        self.prev_height = height
        
        # ========== TRACK REWARD COMPONENTS (for logging) ==========
        self.reward_history['height'].append(height_reward)
        self.reward_history['upright'].append(upright_reward)
        self.reward_history['velocity'].append(stability_reward)
        self.reward_history['control'].append(control_cost)
        self.reward_history['velocity_tracking'].append(velocity_tracking_reward + progress_bonus)
        self.reward_history['jerk'].append(jerk_penalty)

        # ========== COMPREHENSIVE REWARD BREAKDOWN (Phase 4: Diagnostics) ==========
        self._last_reward_components = {
            'velocity_tracking': velocity_tracking_reward,
            'progress_bonus': progress_bonus,
            'standing_penalty': standing_penalty,
            'yaw_tracking': yaw_tracking_reward,
            'height': height_reward,
            'upright': upright_reward,
            'stability': stability_reward,
            'smoothness': smoothness_reward,
            'jerk_penalty': jerk_penalty,
            'control_cost': control_cost,
            'arm_posture_penalty': arm_posture_penalty,
            'height_maintenance': height_maintenance,
            'velocity_penalty': velocity_penalty,
            'recovery_bonus': recovery_bonus,
            'walking_bonus': walking_bonus,
            'sustained_bonus': sustained_bonus,
            'consistency_bonus': consistency_bonus,
            'feet_air_time': feet_air_time_reward,
            'anti_drift': anti_drift_penalty,
            'termination_penalty': termination_penalty,
            'total': total_reward,
        }

        # ========== TERMINATION CAUSE TRACKING ==========
        if terminate:
            if height > 2.0:
                self._termination_cause = 'height_too_high'
            elif up_z < 0.0:
                self._termination_cause = 'severely_tilted'
            elif height < self.termination_height_threshold:
                self._termination_cause = 'height_too_low'
            else:
                self._termination_cause = 'unknown'
        else:
            self._termination_cause = None

        # ========== BEHAVIOR METRICS (for standing-still detection) ==========
        self._behavior_metrics = {
            'is_standing': actual_speed < 0.05,
            'is_slow': actual_speed < 0.1,
            'standing_penalty_applied': standing_penalty < 0,
            'velocity_error': vel_error,
            'actual_speed': actual_speed,
            'commanded_speed': self.commanded_speed,
            'speed_ratio': actual_speed / max(self.commanded_speed, 0.01),
        }

        # ========== DETAILED DEBUG LOGGING ==========
        if self.current_step % 2500 == 0 and self.current_step > 0:
            yaw_info = f", yaw_err={yaw_error:.3f}" if self.include_yaw_rate else ""
            print(f"Step {self.current_step:4d}: "
                f"h={height:.3f}, cmd=({self.commanded_vx_world:.2f},{self.commanded_vy_world:.2f}), "
                f"actual=({com_vel_x:.2f},{com_vel_y:.2f}){yaw_info}, "
                f"vel_err={vel_error:.3f}, r={total_reward:6.1f} "
                f"[track={velocity_tracking_reward:.1f}, prog={progress_bonus:.1f}, yaw={yaw_tracking_reward:.1f}, air={feet_air_time_reward:.1f}, arm={arm_posture_penalty:.1f}]")

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
        """
        Process observation with NEW structure for command visibility.

        FIX: Previously, command was buried in 1496-dim observation and crushed by
        tanh normalization. Now we:
        1. Stack BODY features only in history (no command)
        2. Append command block ONCE at the end with explicit velocity errors
        3. Normalize body with tanh(0.1*x), but keep command block in [-1, 1]

        New observation structure:
        - Stacked body: [body_t-3, body_t-2, body_t-1, body_t] = 1484 dims
        - Command block: [vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual,
                         err_vx, err_vy, err_speed, err_angle] = 9 dims
        - Total: 1493 dims
        """
        # ========== BODY FEATURES ==========
        body_features = [obs]

        # Add COM features if enabled
        if self.include_com:
            try:
                com_pos = self.env.unwrapped.data.subtree_com[0]
                com_vel = self.env.unwrapped.data.cdof_dot[:3] if hasattr(self.env.unwrapped.data, 'cdof_dot') else self.env.unwrapped.data.qvel[:3]
                body_features.append(np.asarray(com_pos, dtype=np.float32))
                body_features.append(np.asarray(com_vel, dtype=np.float32))
            except Exception as e:
                print(f"Warning: Failed to add COM features: {e}")

        # Concatenate body features (NO command here)
        body_vec = np.concatenate([np.atleast_1d(f).ravel() for f in body_features]).astype(np.float32)

        # Normalize body features with tanh
        if self.feature_norm:
            body_vec = np.tanh(body_vec * 0.1)

        # ========== HISTORY STACKING (body only) ==========
        if self.enable_history:
            self.obs_history.append(body_vec)

            if len(self.obs_history) > self.history_len:
                self.obs_history = self.obs_history[-self.history_len:]

            if len(self.obs_history) < self.history_len:
                pad_count = self.history_len - len(self.obs_history)
                padded = [np.zeros_like(body_vec) for _ in range(pad_count)] + self.obs_history
            else:
                padded = self.obs_history

            stacked_body = np.concatenate(padded, axis=0)
        else:
            stacked_body = body_vec

        # ========== COMMAND BLOCK (appended ONCE, not stacked) ==========
        # Get actual velocity from environment
        linear_vel = self.env.unwrapped.data.qvel[0:3]
        vx_actual = linear_vel[0]
        vy_actual = linear_vel[1]

        # Compute explicit velocity errors
        err_vx = self.commanded_vx_world - vx_actual
        err_vy = self.commanded_vy_world - vy_actual
        err_speed = np.sqrt(err_vx**2 + err_vy**2)

        # Error angle (direction mismatch)
        if self.commanded_speed > 0.01:
            cmd_angle = np.arctan2(self.commanded_vy_world, self.commanded_vx_world)
            actual_speed = np.sqrt(vx_actual**2 + vy_actual**2)
            if actual_speed > 0.01:
                actual_angle = np.arctan2(vy_actual, vx_actual)
                err_angle = cmd_angle - actual_angle
                # Wrap to [-pi, pi]
                err_angle = (err_angle + np.pi) % (2 * np.pi) - np.pi
            else:
                err_angle = 0.0
        else:
            err_angle = 0.0

        # Normalize command block to [-1, 1] range (NOT crushed by tanh)
        max_speed = max(self.max_commanded_speed, 1.0)  # Avoid division by zero
        max_yaw = max(self.max_yaw_rate, 1.0)

        command_block = np.array([
            np.clip(self.commanded_vx_world / max_speed, -1.0, 1.0),   # vx_cmd normalized
            np.clip(self.commanded_vy_world / max_speed, -1.0, 1.0),   # vy_cmd normalized
            np.clip(self.commanded_yaw_rate / max_yaw, -1.0, 1.0),     # yaw_cmd normalized
            np.clip(vx_actual / max_speed, -1.0, 1.0),                 # vx_actual normalized
            np.clip(vy_actual / max_speed, -1.0, 1.0),                 # vy_actual normalized
            np.clip(err_vx / max_speed, -1.0, 1.0),                    # err_vx normalized
            np.clip(err_vy / max_speed, -1.0, 1.0),                    # err_vy normalized
            np.clip(err_speed / max_speed, 0.0, 1.0),                  # err_speed normalized (always positive)
            np.clip(err_angle / np.pi, -1.0, 1.0),                     # err_angle normalized
        ], dtype=np.float32)

        # ========== CONCATENATE: stacked body + command block ==========
        feat_vec = np.concatenate([stacked_body, command_block], axis=0)

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
        
        info = {
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

        # ========== INCLUDE REWARD COMPONENTS (Phase 4: Diagnostics) ==========
        if hasattr(self, '_last_reward_components'):
            for key, value in self._last_reward_components.items():
                info[f'reward/{key}'] = value

        # ========== INCLUDE TERMINATION CAUSE ==========
        if hasattr(self, '_termination_cause') and self._termination_cause is not None:
            info['termination_cause'] = self._termination_cause

        # ========== INCLUDE BEHAVIOR METRICS ==========
        if hasattr(self, '_behavior_metrics'):
            for key, value in self._behavior_metrics.items():
                info[f'behavior/{key}'] = value

        return info
    
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
    print("\n Test completed!")

