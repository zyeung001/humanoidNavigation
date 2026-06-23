# standing_env.py


import logging
import os
from collections import deque

import gymnasium as gym
import numpy as np
from typing import Optional
from gymnasium.spaces import Box

from .model_spec import introspect_model, is_custom_xml

logger = logging.getLogger(__name__)


class StandingEnv(gym.Wrapper):


    def __init__(self, render_mode: Optional[str] = None, config=None):
        env_id = "Humanoid-v5"
        print(f"Using {env_id} for standing task")

        # Custom MJCF support. When xml_file is unset, the standard Humanoid-v5
        # model is used and the rest of the env behaves byte-identically to the
        # legacy path (so concurrent std-humanoid training is unaffected).
        cfg_local = config or {}
        xml_file = cfg_local.get('xml_file', None)
        make_kwargs = dict(
            render_mode=render_mode,
            exclude_current_positions_from_observation=False,  # Adds +2 dims (x,y)
        )
        if xml_file is not None:
            # Gymnasium's expand_model_path rejects bare relative paths.
            xml_file = os.path.abspath(xml_file)
            make_kwargs['xml_file'] = xml_file
        env = gym.make(env_id, **make_kwargs)
        super().__init__(env)

        # Introspect when a custom xml is loaded; derive obs dim / standing
        # height / scaling so the reward function's literals tuned for the
        # 1.40 m Humanoid-v5 keep working on a robot of any size.
        self.model_spec = introspect_model(xml_file) if is_custom_xml(xml_file) else None
        self._is_custom = self.model_spec is not None
        if self._is_custom:
            ms = self.model_spec
            self.height_scale = ms.standing_com_z / 1.40   # maps real COM-z into the legacy 0..1.6 regime
            self.nominal_qpos_z = ms.standing_qpos_z
            print(f"  Custom model: {ms.xml_file}")
            print(f"    standing_com_z={ms.standing_com_z:.3f}m  qpos_z={ms.standing_qpos_z:+.3f}m")
            print(f"    height_scale={self.height_scale:.3f}  feet={ms.foot_body_ids}  obs_dim={ms.actual_obs_dim}")
        else:
            self.height_scale = 1.0
            self.nominal_qpos_z = 1.40

        # Configuration
        self.cfg = cfg_local
        self.base_target_height = 1.4
        self.max_episode_steps = self.cfg.get('max_episode_steps', 5000)
        self.current_step = 0
        
        self.domain_rand = self.cfg.get('domain_rand', False)
        self.rand_mass_range = self.cfg.get('rand_mass_range', [0.95, 1.05])
        self.rand_friction_range = self.cfg.get('rand_friction_range', [0.95, 1.05])
        # Nominal model params, captured lazily on first domain-rand reset so
        # randomization is applied relative to nominal (not compounded).
        self._nominal_body_mass = None
        self._nominal_geom_friction = None

        # ======== sim2real robustness (all OFF by default => standard path unchanged) ========
        # Observation noise + per-episode sensor BIASES. A real IMU/encoder set has both
        # per-step noise and a constant per-power-cycle offset; the policy must be robust to
        # both or it limit-cycles on hardware. Only affects the proprioceptive obs path.
        self.obs_noise = bool(self.cfg.get('obs_noise', False))
        self.noise_projgrav = float(self.cfg.get('noise_projgrav', 0.02))   # IMU tilt jitter
        self.noise_angvel = float(self.cfg.get('noise_angvel', 0.05))       # gyro noise (rad/s)
        self.noise_jpos = float(self.cfg.get('noise_jpos', 0.005))          # encoder noise (rad)
        self.noise_jvel = float(self.cfg.get('noise_jvel', 0.20))           # finite-diff vel noise
        self.bias_jpos_std = float(self.cfg.get('bias_jpos', 0.015))        # per-ep encoder offset
        self.bias_angvel_std = float(self.cfg.get('bias_angvel', 0.02))     # per-ep gyro bias
        self._bias_jpos = None
        self._bias_angvel = None
        # Actuator-gain randomization: real position servos differ from the MJCF kp/kv and vary
        # unit-to-unit. Scale kp/kv per episode so the policy doesn't overfit one plant.
        self.actuator_rand = bool(self.cfg.get('actuator_rand', False))
        self.actuator_rand_range = self.cfg.get('actuator_rand_range', [0.8, 1.2])
        self._nominal_gainprm = None
        self._nominal_biasprm = None

        # Actuator LAG model: the dominant sim2real gap for the real robot. Bench step-response
        # (scripts/deploy/servo_step_response.py) measured the SCS servos at ~50 ms dead time +
        # ~120 ms first-order time constant (max ~2.1 rad/s) -- the sim position actuator instead
        # tracks its target the SAME control step. A policy trained on the instant actuator relies
        # on fast corrections the real servo can't follow -> it limit-cycles and falls. This models
        # the real response by feeding the actuator a DELAYED + low-pass-filtered version of the
        # commanded target, randomized per episode (dead time + tau cover unit-to-unit + load).
        # The obs's last_action stays the COMMANDED value and jpos reflects the lagged result, so
        # the policy sees command-vs-actual divergence exactly like hardware. Default off (legacy
        # byte-identical). Training-only: at deploy the REAL servo supplies the lag, so do NOT
        # enable this in the deploy loop (it would double-lag).
        self.actuator_lag = bool(self.cfg.get('actuator_lag', False))
        self.actuator_delay_ms = self.cfg.get('actuator_delay_ms', [30.0, 70.0])
        self.actuator_tau_ms = self.cfg.get('actuator_tau_ms', [90.0, 220.0])
        self._control_dt = float(self.env.unwrapped.dt)   # 0.025 s (40 Hz) for the real model
        self._lag_buffer = None
        self._servo_state = None
        self._lag_alpha = 1.0

        #Random height initialization for recovery training
        self.random_height_init = self.cfg.get('random_height_init', True)
        self.random_height_prob = self.cfg.get('random_height_prob', 0.3)
        self.random_height_range = self.cfg.get('random_height_range', [-0.3, 0.1])
        
        # Reward caps from config
        reward_caps = self.cfg.get('reward_caps', {})
        self.max_height_maintenance_penalty = reward_caps.get('max_height_maintenance_penalty', 15.0)
        self.recovery_bonus_scale = reward_caps.get('recovery_bonus_scale', 50.0)
        self.termination_penalty_constant = reward_caps.get('termination_penalty_constant', 50.0)
        
        self.reward_history = {
            'height': [], 'upright': [], 'velocity': [], 
            'angular': [], 'position': [], 'control': []
        }

        # ======== Enhanced controls (all optional via config) ========
        # Action preprocessing
        self.enable_action_smoothing = bool(self.cfg.get('action_smoothing', False))
        self.action_smoothing_tau = float(self.cfg.get('action_smoothing_tau', 0.5))  # Default 0.5 now
        self.enable_action_symmetry = bool(self.cfg.get('action_symmetry', False))
        self.enable_pd_assist = bool(self.cfg.get('pd_assist', False))
        self.pd_kp = float(self.cfg.get('pd_kp', 0.0))
        self.pd_kd = float(self.cfg.get('pd_kd', 0.0))
        self.prev_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        # Action-rate penalty: discourage high-frequency setpoint jitter (the
        # "taut string" shake). Penalizes ||a_t - a_{t-1}||^2 of the applied
        # (smoothed) action. 0.0 = off, so the legacy std-humanoid path is
        # byte-identical.
        self.action_rate_weight = float(self.cfg.get('action_rate_penalty', 0.0))
        self._last_action_rate = np.zeros(self.env.action_space.shape, dtype=np.float32)

        # Yaw-rate damping. Penalizes (base yaw rate)^2 to discourage the slow
        # in-place spin that proprioceptive obs can't correct via heading
        # (projected gravity is yaw-invariant) but CAN sense via the gyro
        # (qvel[5] / base_ang_vel). 0.0 = off (legacy std path byte-identical).
        self.yaw_rate_weight = float(self.cfg.get('yaw_rate_penalty', 0.0))

        # Observation processing
        self.enable_history = int(self.cfg.get('obs_history', 0)) > 0
        self.history_len = int(self.cfg.get('obs_history', 0))
        self.obs_history = []
        self.include_com = bool(self.cfg.get('obs_include_com', False))
        self.feature_norm = bool(self.cfg.get('obs_feature_norm', False))

        # ----- Proprioceptive (deployable) observation -----
        # When True, the observation is rebuilt from ONLY quantities a real robot
        # can measure: base orientation -> projected gravity (IMU), base angular
        # velocity (gyro), joint positions (encoders), joint velocities, and the
        # last applied action. This replaces Humanoid-v5's stock obs, ~400 dims of
        # which (cinert/cvel/cfrc_ext/COM) are privileged sim state with no real
        # sensor. Default False keeps the legacy privileged obs byte-identical.
        self.proprioceptive = bool(self.cfg.get('proprioceptive_obs', False))
        self.include_foot_contact = bool(self.cfg.get('obs_foot_contact', False))
        if self.proprioceptive:
            # COM (subtree_com) is not measurable on hardware -> never include it.
            self.include_com = False
        # Nominal (home-keyframe) joint pose. Proprioceptive joint obs is reported
        # RELATIVE to this, matching the real robot's encoder-minus-home convention
        # (the v2 home keyframe is a bent standing pose, not all-zeros).
        m_unwrapped = self.env.unwrapped.model
        self.n_joints = int(m_unwrapped.nu)
        if m_unwrapped.nkey > 0:
            self.default_joint_pos = np.asarray(
                m_unwrapped.key_qpos[0][7:7 + self.n_joints], dtype=np.float32
            )
        else:
            self.default_joint_pos = np.zeros(self.n_joints, dtype=np.float32)

        # Humanoid-v5 reports a smaller observation_space than it actually
        # returns (the +15 below is the historical correction for the default
        # capsule model). Custom MJCFs don't have this discrepancy — measure
        # the actual stepped obs dim instead of trusting the magic number.
        base_obs_from_space = int(env.observation_space.shape[0])
        if self.proprioceptive:
            # proj_grav(3) + base_ang_vel(3) + joint_pos(nj) + joint_vel(nj) + last_action(nj)
            base_obs_dim = 6 + 3 * self.n_joints
            if self.include_foot_contact:
                base_obs_dim += 2  # per-foot contact booleans (FSR)
        elif self._is_custom:
            base_obs_dim = self.model_spec.actual_obs_dim
        else:
            base_obs_dim = base_obs_from_space + 15  # Actual observations are 365 dims

        # Calculate dimension after all processing steps
        extra_dim = 6 if self.include_com else 0  # COM pos (3) + COM vel (3)
        feature_dim = base_obs_dim + extra_dim
        
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
        
        print("Observation space configuration:")
        if self.proprioceptive:
            print(f"  PROPRIOCEPTIVE (deployable) obs: "
                  f"proj_grav(3)+ang_vel(3)+jpos({self.n_joints})+jvel({self.n_joints})"
                  f"+last_action({self.n_joints})"
                  f"{'+foot_contact(2)' if self.include_foot_contact else ''} = {base_obs_dim}")
        print(f"  Base from env.observation_space: {base_obs_from_space}")
        print(f"  + Position inclusion adjustment: +15 -> {base_obs_dim}")
        print(f"  + COM features: {extra_dim} -> {feature_dim}")
        print(f"  x History stack: {self.history_len if self.enable_history else 1}")
        print(f"  = FROZEN dimension: {self.frozen_obs_dim}")
        print(f"  Feature normalization: {self.feature_norm}")
        print(f"  Action smoothing tau: {self.action_smoothing_tau}")
        print(f"  Random height init: {self.random_height_init} (prob={self.random_height_prob})")
        print(f"  Max height maintenance penalty: {self.max_height_maintenance_penalty}")
        print(f"  Recovery bonus scale: {self.recovery_bonus_scale}")
        print(f"  Termination penalty (constant): {self.termination_penalty_constant}")
    
    def _get_height(self) -> float:
        """Height signal in the legacy 1.40m regime.

        Default humanoid: returns qpos[2] directly (the freejoint z).
        Custom model: returns subtree-COM z scaled by 1.40/standing_com_z, so a
        standing robot reads ~1.40 regardless of its real size and the reward
        function's hardcoded thresholds (0.75, 1.0, 1.2, 1.40, 1.6, 2.0) keep
        their meaning. The COM is also robust to fusion2urdf's scattered body
        frame origins, where qpos[2] would be an arbitrary number.
        """
        if self._is_custom:
            return float(self.env.unwrapped.data.subtree_com[0][2]) / self.height_scale
        return float(self.env.unwrapped.data.qpos[2])

    def reset(self, seed: Optional[int] = None):
        observation, info = self.env.reset(seed=seed)

        default_height = self._get_height()

        self.current_step = 0
        self.prev_height = default_height
        self.target_height = self.base_target_height
        self.prev_action[:] = 0.0
        self._last_action_rate[:] = 0.0
        self.obs_history = []  # Clear history
        
        # Clear reward history
        for key in self.reward_history:
            self.reward_history[key] = []
        
        if self.domain_rand:
            m = self.env.unwrapped.model
            # Capture nominal params once, then randomize RELATIVE to nominal every
            # reset. The old code multiplied the live arrays in place, so the
            # randomization compounded across episodes and mass/friction drifted
            # away over a long run.
            if self._nominal_body_mass is None:
                self._nominal_body_mass = m.body_mass.copy()
                self._nominal_geom_friction = m.geom_friction.copy()
            m.body_mass[:] = self._nominal_body_mass * np.random.uniform(
                self.rand_mass_range[0], self.rand_mass_range[1],
                size=m.body_mass.shape
            )
            m.geom_friction[:, 0] = self._nominal_geom_friction[:, 0] * np.random.uniform(
                self.rand_friction_range[0], self.rand_friction_range[1],
                size=m.geom_friction.shape[0]
            )

        # Per-episode constant sensor biases (the offset a real IMU/encoder set holds for a
        # whole power-cycle). Sampled once here, added in _proprioceptive_features.
        if self.obs_noise:
            self._bias_jpos = np.random.normal(0.0, self.bias_jpos_std, size=self.n_joints).astype(np.float32)
            self._bias_angvel = np.random.normal(0.0, self.bias_angvel_std, size=3).astype(np.float32)

        # Actuator-gain randomization: scale each position servo's kp/kv per episode, relative
        # to nominal (captured once). gainprm[:,0]=kp, biasprm[:,1]=-kp, biasprm[:,2]=-kv.
        if self.actuator_rand:
            m = self.env.unwrapped.model
            if self._nominal_gainprm is None:
                self._nominal_gainprm = m.actuator_gainprm.copy()
                self._nominal_biasprm = m.actuator_biasprm.copy()
            scale = np.random.uniform(
                self.actuator_rand_range[0], self.actuator_rand_range[1],
                size=m.actuator_gainprm.shape[0]
            )
            m.actuator_gainprm[:, 0] = self._nominal_gainprm[:, 0] * scale
            m.actuator_biasprm[:, 1] = self._nominal_biasprm[:, 1] * scale
            m.actuator_biasprm[:, 2] = self._nominal_biasprm[:, 2] * scale

        # Actuator lag: sample a per-episode dead time + first-order time constant, build the
        # delay line and reset the filter state. Dead time -> integer control-step delay; tau ->
        # discrete first-order coefficient alpha = 1 - exp(-dt/tau).
        if self.actuator_lag:
            dt = self._control_dt
            delay_ms = np.random.uniform(self.actuator_delay_ms[0], self.actuator_delay_ms[1])
            tau_ms = np.random.uniform(self.actuator_tau_ms[0], self.actuator_tau_ms[1])
            delay_steps = max(0, int(round((delay_ms / 1000.0) / dt)))
            self._lag_alpha = float(1.0 - np.exp(-dt / max(tau_ms / 1000.0, 1e-4)))
            zero = np.zeros(self.env.action_space.shape, dtype=np.float32)
            self._lag_buffer = deque([zero.copy() for _ in range(delay_steps + 1)],
                                     maxlen=delay_steps + 1)
            self._servo_state = zero.copy()

        # Random height initialization for recovery training. Skip for custom
        # models: the legacy [0.6, 1.6] clip assumes a 1.40 m-target capsule
        # humanoid and would push a smaller robot through the floor or ceiling.
        if (not self._is_custom) and self.random_height_init and np.random.random() < self.random_height_prob:
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
        
        # Process observation with guaranteed dimension
        if self.enable_history or self.include_com or self.feature_norm or self.proprioceptive:
            observation = self._process_observation(observation)

        return observation, info
    
    def step(self, action):
        # Action preprocessing. Capture the previously-applied (smoothed) action
        # before _process_action overwrites prev_action, so the reward can charge
        # the per-step action rate.
        prev_applied = self.prev_action.copy()
        proc_action = self._process_action(np.asarray(action, dtype=np.float32))
        self._last_action_rate = proc_action - prev_applied

        # The COMMANDED (smoothed) action is what the obs reports as last_action and what the
        # reward charges for control/rate. The ACTUATOR, however, receives the lagged signal so
        # the resulting joint motion (and thus the jpos obs) trails the command like a real servo.
        actuator_action = self._apply_actuator_lag(proc_action) if self.actuator_lag else proc_action
        observation, base_reward, terminated, truncated, info = self.env.step(actuator_action)
        self.current_step += 1
        
        # Modify reward for standing
        reward, terminated = self._compute_task_reward(observation, base_reward, info, proc_action)
        
        # Override termination for standing to allow indefinite episodes
        truncated = self.current_step >= self.max_episode_steps
        
        # Add task info 
        info.update(self._get_task_info())
        
        # Process observation with dimension verification
        if self.enable_history or self.include_com or self.feature_norm or self.proprioceptive:
            observation = self._process_observation(observation)

        return observation, reward, terminated, truncated, info
        
    def _compute_task_reward(self, obs, base_reward, info, action):
        # ========== STATE EXTRACTION ==========
        # _get_height() returns qpos[2] for the default humanoid (legacy
        # behavior) and a scaled subtree-COM z for custom models, so all of
        # the literal thresholds below stay calibrated.
        height = self._get_height()
        root_x, root_y = self.env.unwrapped.data.qpos[0:2]
        quat = self.env.unwrapped.data.qpos[3:7]  # [w, x, y, z] quaternion
        
        linear_vel = self.env.unwrapped.data.qvel[0:3]
        angular_vel = self.env.unwrapped.data.qvel[3:6]
        joint_vel = self.env.unwrapped.data.qvel[6:]  # Joint velocities
        
        # Target height
        target_height = self.base_target_height
        height_error = abs(height - target_height)
        
        # ========== CORE REWARD: ASYMMETRIC HEIGHT REWARD ==========
        if height < 1.0:
            height_reward = -50.0 + 40.0 * np.clip(height / 1.0, 0.0, 1.0)  
        elif height < 1.2:
            height_reward = -10.0 + 15.0 * (height - 1.0) / 0.2  
        elif height < 1.35:
            height_reward = 5.0 + 20.0 * (height - 1.2) / 0.15 
        elif height < 1.45:  # Peak reward zone around 1.40m
            height_reward = 25.0 + 25.0 * (1.0 - abs(height - 1.40) / 0.05)  
        elif height < 1.6:
            height_reward = 40.0 - 15.0 * (height - 1.45) / 0.15  
        else:
            height_reward = 25.0 - 20.0 * np.clip((height - 1.6) / 0.2, 0.0, 1.0)  
        
        # ========== UPRIGHT ORIENTATION REWARD  ==========
        # Yaw-invariant tilt from projected gravity: gz = -1 upright, 0 horizontal.
        # upright_cos = +1 upright, 0 horizontal, -1 inverted. (quat_w was WRONG
        # here: a pure yaw spin shrinks quat_w and falsely reads as tipping.)
        proj_grav = self._projected_gravity(quat)
        upright_cos = -float(proj_grav[2])
        upright_error = 1.0 - upright_cos
        if height >= 1.2:
            upright_reward = 12.0 * np.exp(-8.0 * upright_error**2)  
        elif height >= 1.0:
            upright_reward = 7.0 * np.exp(-8.0 * upright_error**2)  
        else:
            upright_reward = 0.0
        
        # ========== STABILITY REWARD (RESCALED) ==========
        angular_momentum = np.sum(np.square(angular_vel))
        if height >= 1.3:
            stability_reward = 7.0 * np.exp(-2.0 * angular_momentum)
        elif height >= 1.2:
            stability_reward = 5.0 * np.exp(-2.0 * angular_momentum) 
        elif height >= 1.0:
            stability_reward = 2.5 * np.exp(-2.0 * angular_momentum)
        else:
            stability_reward = 0.0
        
        # ========== SMOOTHNESS REWARD ==========
        joint_velocity_magnitude = np.sum(np.square(joint_vel))
        smoothness_reward = 2.5 * np.exp(-0.1 * joint_velocity_magnitude) 
        
        # ========== CONTROL COST ========== 
        control_cost = -0.005 * np.sum(np.square(action))

        # ==========  ACTION-RATE PENALTY (anti-jitter / sim2real smoothness) ==========
        # Charge the squared per-step change in the applied target. Capped so a
        # single transient can't spike PPO's value targets.
        if self.action_rate_weight > 0.0:
            action_rate_penalty = -min(
                self.action_rate_weight * float(np.sum(np.square(self._last_action_rate))),
                20.0,
            )
        else:
            action_rate_penalty = 0.0

        # ==========  YAW-RATE DAMPING (anti-spin) ==========
        # angular_vel = qvel[3:6] (base frame); [2] is yaw rate. Penalize its
        # square, capped, so a transient can't spike PPO value targets.
        if self.yaw_rate_weight > 0.0:
            yaw_rate_penalty = -min(
                self.yaw_rate_weight * float(angular_vel[2]) ** 2,
                10.0,
            )
        else:
            yaw_rate_penalty = 0.0

        # ==========  CAPPED HEIGHT MAINTENANCE PENALTY ==========
        height_velocity = height - self.prev_height if hasattr(self, 'prev_height') else 0.0
        if height_velocity < -0.003:  # Losing height
            # CAP the penalty to prevent catastrophic negative rewards
            capped_velocity = np.clip(abs(height_velocity), 0.0, 0.1)
            height_maintenance = -150.0 * capped_velocity  # Max penalty: -15
        elif abs(height_velocity) < 0.003:  
            height_maintenance = 5.0  
        else:  # Gaining height (not penalized)
            height_maintenance = 0.0
        
        # ==========  RECOVERY BONUS ==========
        # Explicitly reward recovering from low heights
        recovery_bonus = 0.0
        if height < 1.0 and height_velocity > 0.01:  # Rising from low height
            # Scale bonus by how low we are (lower = more bonus for rising)
            recovery_scale = (1.0 - height) / 0.4  # 0 at h=1.0, 1.0 at h=0.6
            recovery_bonus = self.recovery_bonus_scale * height_velocity * recovery_scale
        
        # ========== VELOCITY PENALTY  ==========
        speed = np.linalg.norm(linear_vel[:2])  # XY velocity only
        velocity_penalty = -1.0 * np.clip(speed - 0.5, 0.0, 2.0)  
        
        # ========== DURATION BONUS (for indefinite standing) ==========
        # Reward agent for maintaining good standing over long periods
        sustained_bonus = 0.0
        if self.current_step > 0 and self.current_step % 100 == 0:
            if height_error < 0.08 and upright_error < 0.08 and height >= 1.32:
                # Base bonus for 100 steps of good standing
                sustained_bonus = 100.0
                
                # ADDITIONAL: Progressive bonus for longer durations
                if self.current_step >= 500:
                    sustained_bonus += 50.0  # Extra reward past 500 steps
                if self.current_step >= 1000:
                    sustained_bonus += 50.0  # Extra reward past 1000 steps
                if self.current_step >= 2000:
                    sustained_bonus += 100.0  # Big reward for ultra-long standing
        
        # ========== : CONSTANT TERMINATION PENALTY ==========
        # No longer scales with time - removes learned helplessness
        termination_penalty = 0.0
        
        # ========== TERMINATION CONDITIONS ==========
        # Tip-over test uses yaw-invariant tilt (upright_cos < 0.5 ~= >60 deg
        # tilt), NOT quat_w (which fired at ~106 deg of harmless yaw).
        terminate = (
            height < 0.75 or
            height > 2.0 or
            upright_cos < 0.5
        )

        #  Apply CONSTANT termination penalty (not time-dependent)
        if terminate:
            if height < 0.75:  # Fell too low
                termination_penalty = -self.termination_penalty_constant
            elif upright_cos < 0.5:  # Tipped over
                termination_penalty = -self.termination_penalty_constant
            else:  # Other terminations (height too high)
                termination_penalty = -self.termination_penalty_constant * 0.5
        
        # ========== TOTAL REWARD ==========
        total_reward = (
            height_reward +
            upright_reward +
            stability_reward +
            smoothness_reward +
            control_cost +
            height_maintenance +
            recovery_bonus +
            velocity_penalty +
            sustained_bonus +
            action_rate_penalty +
            yaw_rate_penalty +
            termination_penalty
        )
        
        # Update previous height for next step
        self.prev_height = height
        
        # ========== TRACK REWARD COMPONENTS ==========
        self.reward_history['height'].append(height_reward)
        self.reward_history['upright'].append(upright_reward)
        self.reward_history['velocity'].append(stability_reward)
        self.reward_history['control'].append(control_cost)
        
        # ========== DEBUG LOGGING  ==========
        if self.current_step % 500 == 0:
            print(f"Step {self.current_step:4d}: "
                f"h={height:.3f} (err={height_error:.3f}), "
                f"tilt_cos={upright_cos:.3f}, yaw_rate={float(angular_vel[2]):+.2f}, "
                f"r={total_reward:6.1f} "
                f"[h={height_reward:6.1f}, u={upright_reward:4.1f}, "
                f"stab={stability_reward:4.1f}, recov={recovery_bonus:4.1f}, "
                f"rate={action_rate_penalty:5.1f}, yaw={yaw_rate_penalty:5.1f}, "
                f"bonus={sustained_bonus:.0f}]")
        
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
            except (AttributeError, IndexError):
                pass

        if self.enable_action_smoothing:
            tau = np.clip(self.action_smoothing_tau, 0.0, 1.0)
            action = (1.0 - tau) * self.prev_action + tau * action

        low, high = self.env.action_space.low, self.env.action_space.high
        action = np.clip(action, low, high)
        self.prev_action = action.copy()
        return action

    def _apply_actuator_lag(self, cmd: np.ndarray) -> np.ndarray:
        """Model the real servo: dead-time delay then first-order lag toward the command.

        cmd is the commanded (smoothed) target. Returns the value actually sent to the MuJoCo
        actuator this step. The delay line gives transport dead time; the EMA gives the ~120 ms
        mechanical time constant (and, implicitly, the ~2 rad/s slew ceiling). Clipped to the
        action range so the lagged signal is always a valid actuator command."""
        self._lag_buffer.append(cmd.copy())
        delayed = self._lag_buffer[0]            # oldest sample = command from delay_steps ago
        self._servo_state = self._servo_state + self._lag_alpha * (delayed - self._servo_state)
        low, high = self.env.action_space.low, self.env.action_space.high
        return np.clip(self._servo_state, low, high).astype(np.float32)

    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process observation with correct dimension handling."""
        if self.proprioceptive:
            # Rebuild obs from measurable-only signals; ignore the privileged
            # Humanoid-v5 obs entirely.
            feat_vec = self._proprioceptive_features()
        else:
            features = [obs]

            # Add COM features if enabled
            if self.include_com:
                try:
                    com_pos = self.env.unwrapped.data.subtree_com[0]
                    com_vel = self.env.unwrapped.data.cdof_dot[:3] if hasattr(self.env.unwrapped.data, 'cdof_dot') else self.env.unwrapped.data.qvel[:3]
                    features.append(np.asarray(com_pos, dtype=np.float32))
                    features.append(np.asarray(com_vel, dtype=np.float32))
                except (AttributeError, IndexError) as e:
                    logger.warning("Failed to add COM features: %s", e)

            # Concatenate base + COM features
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
                # Truncate if larger
                feat_vec = feat_vec[:self.frozen_obs_dim]
            else:
                # Pad if smaller
                pad = np.zeros((self.frozen_obs_dim - current_dim,), dtype=np.float32)
                feat_vec = np.concatenate([feat_vec, pad], axis=0)

        return feat_vec

    # ======== Proprioceptive (deployable) observation ========

    @staticmethod
    def _projected_gravity(quat: np.ndarray) -> np.ndarray:
        """Gravity direction expressed in the base body frame.

        World gravity points down ([0,0,-1]); rotating it into the base frame by
        the root orientation gives a 3-vector that is [0,0,-1] when upright and
        tilts as the torso tips. This is the standard sim2real orientation cue and
        is reconstructed on hardware from the IMU's fused orientation.

        quat is MuJoCo order [w, x, y, z].
        Convention check: real IMU accel at rest reads +1g UP, i.e. specific
        force = -projected_gravity. Deployment should feed -accel_normalized here.
        """
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        gx = 2.0 * (w * y - x * z)
        gy = -2.0 * (y * z + w * x)
        gz = 2.0 * (x * x + y * y) - 1.0
        return np.array([gx, gy, gz], dtype=np.float32)

    def _foot_contacts(self) -> np.ndarray:
        """Per-foot contact booleans — the signal the FSRs will provide.

        Derived in sim from the foot bodies' external contact force magnitude.
        Defaults off; only used when obs_foot_contact is set.
        """
        data = self.env.unwrapped.data
        ids = self.model_spec.foot_body_ids if self._is_custom else []
        out = []
        for bid in list(ids)[:2]:
            f = float(np.linalg.norm(data.cfrc_ext[bid][3:6]))
            out.append(1.0 if f > 1.0 else 0.0)
        while len(out) < 2:
            out.append(0.0)
        return np.asarray(out, dtype=np.float32)

    def _proprioceptive_features(self) -> np.ndarray:
        """Build the per-frame observation from measurable-only signals.

        Order (must match deployment exactly):
          projected_gravity(3) | base_ang_vel(3) | joint_pos-default(nj) |
          joint_vel(nj) | last_action(nj) | [foot_contact(2)]
        """
        data = self.env.unwrapped.data
        # base orientation -> projected gravity (IMU fused orientation on hardware)
        proj_grav = self._projected_gravity(np.asarray(data.qpos[3:7]))
        # base angular velocity, free-joint local frame (matches IMU gyro)
        base_ang_vel = np.asarray(data.qvel[3:6], dtype=np.float32)
        # joint angles relative to home pose (encoder-minus-home on hardware)
        jpos = np.asarray(data.qpos[7:7 + self.n_joints], dtype=np.float32) - self.default_joint_pos
        # joint velocities (finite-difference of encoders on hardware)
        jvel = np.asarray(data.qvel[6:6 + self.n_joints], dtype=np.float32)
        # last applied (smoothed) action
        last_action = np.asarray(self.prev_action, dtype=np.float32).ravel()

        # sim2real: per-episode bias + per-step Gaussian noise on the measured channels
        # (last_action is internal, left clean). Off unless obs_noise is set.
        if self.obs_noise:
            proj_grav = proj_grav + np.random.normal(0.0, self.noise_projgrav, size=3).astype(np.float32)
            base_ang_vel = base_ang_vel + np.random.normal(0.0, self.noise_angvel, size=3).astype(np.float32)
            jpos = jpos + np.random.normal(0.0, self.noise_jpos, size=self.n_joints).astype(np.float32)
            jvel = jvel + np.random.normal(0.0, self.noise_jvel, size=self.n_joints).astype(np.float32)
            if self._bias_angvel is not None:
                base_ang_vel = base_ang_vel + self._bias_angvel
            if self._bias_jpos is not None:
                jpos = jpos + self._bias_jpos

        feats = [proj_grav, base_ang_vel, jpos, jvel, last_action]
        if self.include_foot_contact:
            feats.append(self._foot_contacts())
        return np.concatenate([np.atleast_1d(f).ravel() for f in feats]).astype(np.float32)

    def _get_task_info(self):
        """Get task-specific information"""
        root_x, root_y = self.env.unwrapped.data.qpos[0:2]
        dist = np.sqrt(root_x**2 + root_y**2)
        return {
            'height': self._get_height(),
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
        print(f"Raw observation size: {len(obs)}")
        print(f"Frozen dimension: {self.frozen_obs_dim}")
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
    return StandingEnv(render_mode=render_mode, config=config)

