"""
walking_curriculum.py

Curriculum learning for walking task.
Progressive 3-stage curriculum for ~2ft (0.6m) humanoid sim-to-real transfer.
Speeds are realistic for small bipedal robots (max ~0.8 m/s).
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np

from .walking_env import WalkingEnv


class WalkingCurriculumEnv(WalkingEnv):
    """
    3-Stage Curriculum for Walking (~2ft / 0.6m humanoid, sim-to-real):
    0: 0.15 m/s - Baby steps, learn balance while moving (NO perturbations, forward only)
    1: 0.40 m/s - Normal walking pace (light perturbations, wider direction)
    2: 0.80 m/s - Brisk walking near real-world max (moderate perturbations, full direction)

    Advancement: Multiple paths to advance:
    1. Standard: success_rate >= threshold AND avg_vel_error < tolerance
    2. Length-based: avg_episode_length > 2x min AND fall_rate < 15%
    3. Improvement-based: consistent improvement over training window
    """

    def __init__(self, render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).copy()
        self.stage = int(cfg.get('curriculum_start_stage', 0))
        self.max_stage = int(cfg.get('curriculum_max_stage', 1))
        self.advance_after = int(cfg.get('curriculum_advance_after', 20))  # Increased for more stable estimate
        self.success_buffer = []
        self.velocity_error_buffer = []
        self.episode_length_buffer = []
        self.stage_success_threshold = float(cfg.get('curriculum_success_rate', 0.35))  # RELAXED: 35%

        # 3-stage curriculum for ~2ft humanoid (sim-to-real)
        speed_stages_cfg = cfg.get('curriculum_max_speed_stages', [0.15, 0.4, 0.8])
        self.speed_stages = speed_stages_cfg
        self.standing_probability = [0.0] * len(speed_stages_cfg)
        self.velocity_tolerances = [0.15, 0.20, 0.30, 0.40]
        self.min_episode_lengths = [200, 300, 400, 500]
        self.height_tolerances = [0.45, 0.40, 0.35, 0.30]

        # Direction diversity - DISABLED by default for Stage 0
        # Stage 0 should be boring: forward only, fixed speed
        self.direction_diversity = cfg.get('direction_diversity', False)

        # Episode velocity error tracking
        self.episode_velocity_errors = []
        self.episode_yaw_errors = []
        self.yaw_error_buffer = []
        self.yaw_error_tolerances = [0.40, 0.35, 0.30, 0.25]
        self.stabilization_steps = 50  # Skip first N steps when computing average

        # Warmup tracking
        self.total_episodes = 0
        self.warmup_episodes = int(cfg.get('warmup_episodes', 50))

        # Fall tracking for alternative advancement
        self.fall_buffer = []  # Track terminated episodes (falls)

        # ========== CURRICULUM-GATED ARM PENALTY ==========
        # Stage 0: no arm penalty (learn to walk freely, chicken wings OK)
        # Later stages: gradually introduce arm posture penalty
        self._arm_penalty_stage_weights = cfg.get('arm_penalty_stage_weights', [0.0, 0.4, 0.6, 0.8])
        self._arm_penalty_ramp_rate = float(cfg.get('arm_penalty_ramp_rate', 0.000003))
        self._arm_penalty_target = self._arm_penalty_stage_weights[min(self.stage, len(self._arm_penalty_stage_weights) - 1)]
        self._arm_penalty_current = self._arm_penalty_target
        # Force arm weight to stage target BEFORE super().__init__() so WalkingEnv reads it
        cfg['reward_arm_posture_weight'] = self._arm_penalty_target

        self._apply_stage_settings(cfg, self.stage)
        super().__init__(render_mode=render_mode, config=cfg)

        # Start arm penalty at the stage target immediately (no ramp from 0)
        # This avoids wasting steps re-ramping on every resume
        self._arm_penalty_current = self._arm_penalty_target
        self.arm_posture_weight = self._arm_penalty_target
        print(f"  Arm penalty: current={self._arm_penalty_current:.3f}, target={self._arm_penalty_target:.3f}, ramp_rate={self._arm_penalty_ramp_rate}")
        
        # Override max_commanded_speed from curriculum
        self.max_commanded_speed = self.speed_stages[self.stage]
        
        print(f"  Walking curriculum initialized at stage {self.stage}")
        print(f"  Max speed: {self.max_commanded_speed:.2f} m/s")
        print(f"  Standing probability: {self.standing_probability[self.stage]:.0%}")
        print(f"  Velocity tolerance: ±{self.velocity_tolerances[self.stage]:.2f} m/s")
        print(f"  Min episode length: {self.min_episode_lengths[self.stage]} steps")
        print(f"  Push enabled: {self.push_enabled}")

    def _apply_stage_settings(self, cfg: Dict[str, Any], stage: int) -> None:
        """Configure curriculum stage with progressive difficulty.

        3-stage curriculum for ~2ft humanoid (sim-to-real):
        Stage 0: Baby steps (0.30 m/s) - no perturbations, forward only
        Stage 1: Normal walk (0.60 m/s) - light perturbations, wider direction
        Stage 2: Brisk walk (1.00 m/s) - moderate perturbations, full direction
        Stage 3: Fast walk (1.50 m/s) - full perturbations, domain rand, sim-to-real ready
        """
        cfg['max_commanded_speed'] = self.speed_stages[min(stage, len(self.speed_stages) - 1)]

        if stage == 0:
            # Stage 0 - Baby steps, learn to balance while moving
            cfg['domain_rand'] = False
            cfg['max_episode_steps'] = 2000
            cfg['push_enabled'] = False
            cfg['random_height_init'] = False
            cfg['random_height_prob'] = 0.0
            cfg['velocity_weight'] = 6.0
        elif stage == 1:
            # Stage 1 - Normal walking pace, light perturbations
            cfg['domain_rand'] = False
            cfg['max_episode_steps'] = 2500
            cfg['push_enabled'] = True
            cfg['push_magnitude_range'] = [10.0, 40.0]
            cfg['push_interval'] = 350
            cfg['random_height_init'] = False
            cfg['random_height_prob'] = 0.0
            cfg['velocity_weight'] = 5.0
        elif stage == 2:
            # Stage 2 - Faster walking, same robustness as Stage 1
            # Only change is speed (0.45 m/s). No domain rand, no random init.
            # Robustness additions deferred to Stage 3.
            cfg['domain_rand'] = False
            cfg['max_episode_steps'] = 3000
            cfg['push_enabled'] = True
            cfg['push_magnitude_range'] = [10.0, 40.0]
            cfg['push_interval'] = 350
            cfg['random_height_init'] = False
            cfg['random_height_prob'] = 0.0
            cfg['velocity_weight'] = 5.0
        elif stage == 3:
            # Stage 3 - Same as Stage 2 + light domain rand only
            # Pushes stay at Stage 2 levels — isolate domain rand as the new challenge
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.97, 1.03]
            cfg['rand_friction_range'] = [0.97, 1.03]
            cfg['max_episode_steps'] = 3000
            cfg['push_enabled'] = True
            cfg['push_magnitude_range'] = [10.0, 40.0]
            cfg['push_interval'] = 350
            cfg['random_height_init'] = False
            cfg['random_height_prob'] = 0.0
            cfg['random_velocity_init'] = False
            cfg['velocity_weight'] = 5.0
        else:
            # Stage 4+ - Full sim-to-real hardening
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.90, 1.10]
            cfg['rand_friction_range'] = [0.90, 1.10]
            cfg['max_episode_steps'] = 4000
            cfg['push_enabled'] = True
            cfg['push_magnitude_range'] = [30.0, 100.0]
            cfg['push_interval'] = 250
            cfg['random_height_init'] = True
            cfg['random_height_prob'] = 0.20
            cfg['random_velocity_init'] = True
            cfg['random_velocity_range'] = [-0.3, 0.3]
            cfg['velocity_weight'] = 5.0

    def reset(self, seed: Optional[int] = None):
        # Clear episode error tracking
        self.episode_velocity_errors = []
        self.episode_yaw_errors = []
        
        # Update max speed for current stage
        current_stage = min(self.stage, len(self.speed_stages) - 1)
        self.max_commanded_speed = self.speed_stages[current_stage]
        
        # Update command generator ranges based on curriculum stage
        # Yaw range scales with stage: Stage 0 gets ±0.3, Stage 1 gets ±0.5, Stage 2+ gets full range
        # This gives the command generator curriculum-appropriate yaw for within-episode resampling
        if hasattr(self, 'command_generator') and self.command_generator is not None:
            max_yaw = getattr(self, 'max_yaw_rate', 1.0)
            yaw_scales = [0.3, 0.5, 0.75, 1.0]  # Per-stage yaw scaling
            yaw_scale = yaw_scales[min(current_stage, len(yaw_scales) - 1)]
            self.command_generator.update_curriculum_ranges([
                (0.0, self.max_commanded_speed),
                (-self.max_commanded_speed * 0.5, self.max_commanded_speed * 0.5),
                (-max_yaw * yaw_scale, max_yaw * yaw_scale)
            ])
        
        # Mix standing and walking commands based on curriculum stage
        standing_prob = self.standing_probability[current_stage]
        # Turn-in-place probability: configurable, default 20%
        # 5% was too low — agent saw ~250 TIP episodes in 3M steps, not enough to learn.
        # 15% was tried before with sparse air time + pure Gaussian — destabilized VF.
        # Now with continuous air time + denser yaw signal, 20% should be safe.
        tip_prob = float(self.cfg.get('turn_in_place_prob', 0.20))
        turn_in_place_prob = tip_prob if current_stage >= 0 else 0.0
        if np.random.random() < standing_prob:
            # Force standing command with yaw_rate = 0
            self.fixed_command = (0.0, 0.0, 0.0)
        elif np.random.random() < turn_in_place_prob:
            # Turn in place: zero speed, non-zero yaw — essential for maze navigation
            max_yaw = getattr(self, 'max_yaw_rate', 1.0)
            yaw_scales = [0.3, 0.5, 0.75, 1.0]  # Match command generator stage scaling
            yaw_scale = yaw_scales[min(current_stage, len(yaw_scales) - 1)]
            yaw_rate = np.random.uniform(-max_yaw, max_yaw) * yaw_scale
            # Avoid near-zero yaw (not useful training signal)
            if abs(yaw_rate) < 0.1:
                yaw_rate = 0.1 * (1.0 if yaw_rate >= 0 else -1.0)
            self.fixed_command = (0.0, 0.0, yaw_rate)
        elif self.direction_diversity:
            # Direction diversity also uses command generator now (same reasoning).
            # Command generator handles direction variety with within-episode resampling.
            self.fixed_command = None
        else:
            # CRITICAL FIX: Let command generator handle ALL walking episodes.
            # SOTA (legged_gym, Isaac Lab, Walk These Ways) resamples commands
            # every 2-6 seconds WITHIN each episode. Our old approach fixed
            # commands per-episode — 80% of episodes never changed yaw, so
            # the policy never practiced turning during walking.
            # Command generator ranges are already set above with stage-appropriate
            # yaw scaling (Stage 0: ±0.3, Stage 1: ±0.5, Stage 2+: full range).
            self.fixed_command = None
        
        # Store the curriculum's fixed_command before calling parent reset
        curriculum_fixed_command = self.fixed_command

        obs, info = super().reset(seed=seed)

        # FIX: Restore curriculum's fixed_command after parent reset
        # Previously, setting self.fixed_command = None here would override
        # the Stage 0 forward-only constraint, allowing random lateral commands
        if curriculum_fixed_command is not None:
            self.fixed_command = curriculum_fixed_command
            # Also update the commanded velocity to match
            self.commanded_vx_world = float(curriculum_fixed_command[0])
            self.commanded_vy_world = float(curriculum_fixed_command[1])
            self.commanded_yaw_rate = float(curriculum_fixed_command[2]) if len(curriculum_fixed_command) > 2 else 0.0
            self.commanded_speed = np.sqrt(self.commanded_vx_world**2 + self.commanded_vy_world**2)
            if self.commanded_speed > 0:
                self.commanded_angle = np.arctan2(self.commanded_vy_world, self.commanded_vx_world)
            else:
                self.commanded_angle = 0.0

        return obs, info

    def step(self, action):
        # Ramp arm penalty toward target BEFORE super().step() so reward uses updated weight
        if self._arm_penalty_current < self._arm_penalty_target:
            self._arm_penalty_current = min(
                self._arm_penalty_current + self._arm_penalty_ramp_rate,
                self._arm_penalty_target
            )
        elif self._arm_penalty_current > self._arm_penalty_target:
            # Allow downward ramp too (e.g., if resuming at wrong stage)
            self._arm_penalty_current = max(
                self._arm_penalty_current - self._arm_penalty_ramp_rate,
                self._arm_penalty_target
            )
        self.arm_posture_weight = self._arm_penalty_current

        obs, reward, terminated, truncated, info = super().step(action)
        
        # Track velocity and yaw error every step for episode average
        if 'velocity_error' in info:
            self.episode_velocity_errors.append(info['velocity_error'])
        if 'yaw_rate_error' in info:
            self.episode_yaw_errors.append(info['yaw_rate_error'])

        done = bool(terminated or truncated)
        if done:
            self.total_episodes += 1
            height = info.get('height', 0.0)
            
            current_stage = min(self.stage, len(self.speed_stages) - 1)
            current_vel_tol = self.velocity_tolerances[current_stage]
            min_length = self.min_episode_lengths[current_stage]
            
            # Use episode-average velocity error (after stabilization)
            if len(self.episode_velocity_errors) > self.stabilization_steps:
                # Skip first N steps (stabilization period)
                stable_errors = self.episode_velocity_errors[self.stabilization_steps:]
                velocity_error = np.mean(stable_errors)
                velocity_error_std = np.std(stable_errors)
            elif self.episode_velocity_errors:
                velocity_error = np.mean(self.episode_velocity_errors)
                velocity_error_std = np.std(self.episode_velocity_errors)
            else:
                velocity_error = info.get('velocity_error', 0.0)
                velocity_error_std = 0.0
            
            # Compute episode-average yaw error (after stabilization)
            if len(self.episode_yaw_errors) > self.stabilization_steps:
                stable_yaw = self.episode_yaw_errors[self.stabilization_steps:]
                yaw_error = np.mean(stable_yaw)
            elif self.episode_yaw_errors:
                yaw_error = np.mean(self.episode_yaw_errors)
            else:
                yaw_error = info.get('yaw_rate_error', 0.0)

            # Update info with episode-average error
            info['velocity_error'] = velocity_error
            info['velocity_error_std'] = velocity_error_std
            info['velocity_error_full_episode'] = np.mean(self.episode_velocity_errors) if self.episode_velocity_errors else 0.0
            info['yaw_error_avg'] = yaw_error

            # Buffer yaw error for advancement check
            self.yaw_error_buffer.append(yaw_error)
            if len(self.yaw_error_buffer) > self.advance_after * 2:
                self.yaw_error_buffer = self.yaw_error_buffer[-self.advance_after:]

            # Track episode length
            self.episode_length_buffer.append(self.current_step)
            if len(self.episode_length_buffer) > self.advance_after * 2:
                self.episode_length_buffer = self.episode_length_buffer[-self.advance_after:]
            
            # Track falls for alternative advancement
            self.fall_buffer.append(1 if terminated else 0)
            if len(self.fall_buffer) > self.advance_after:
                self.fall_buffer = self.fall_buffer[-self.advance_after:]
            
            # No warmup leniency - enforce strict velocity tracking from the start
            # This prevents the standing-still exploit from being masked
            vel_tolerance_multiplier = 1.0  # No buffer - velocity error must be below tolerance
            
            # SUCCESS CRITERIA - RELAXED for Stage 0 to encourage movement
            # Key insight: The agent needs to learn that MOVING is good
            # We can refine accuracy in later stages
            criteria_met = 0
            
            # Velocity error within tolerance
            vel_ok = velocity_error < current_vel_tol * vel_tolerance_multiplier
            if vel_ok:
                criteria_met += 1
            
            # Height maintained (relaxed for walking)
            height_ok = height > 0.95 and height < 1.65
            if height_ok:
                criteria_met += 1
            
            # Episode lasted long enough
            long_enough = self.current_step >= min_length
            if long_enough:
                criteria_met += 1
            
            # Not terminated due to falling
            not_fallen = not terminated
            
            # NEW: Did the agent actually MOVE? (anti-standing-still check)
            actual_speed_info = info.get('actual_speed', 0.0)
            commanded_speed = info.get('commanded_speed', self.commanded_speed)

            # Stage 0: STRICT movement requirement - must actually be moving
            # This prevents standing-still from passing Stage 0
            if current_stage == 0:
                if commanded_speed > 0.05:
                    # Require at least 50% of commanded speed
                    min_required_speed = commanded_speed * 0.50
                    is_moving = actual_speed_info >= min_required_speed
                    success = bool(not_fallen and long_enough and vel_ok and is_moving)
                else:
                    # Standing command - just need to survive
                    success = bool(not_fallen and long_enough and vel_ok)
            elif current_stage == 1:
                # Stage 1: Slightly relaxed - require some movement attempt
                attempted_movement = actual_speed_info > 0.05 or commanded_speed < 0.05
                success = bool(
                    not_fallen and
                    long_enough and
                    (vel_ok or (height_ok and attempted_movement))
                )
            elif current_stage == 2:
                # Stage 2: Full criteria - velocity tracking + stability
                success = bool(not_fallen and long_enough and vel_ok and height_ok)
            else:
                # Later stages: Standard criteria
                success = bool((not_fallen and criteria_met >= 2) or criteria_met == 3)

            self.success_buffer.append(1 if success else 0)
            self.velocity_error_buffer.append(velocity_error)
            
            if len(self.success_buffer) > self.advance_after:
                self.success_buffer = self.success_buffer[-self.advance_after:]
                self.velocity_error_buffer = self.velocity_error_buffer[-self.advance_after:]

            # advancing logic - Multiple paths
            if len(self.success_buffer) == self.advance_after:
                avg_recent_vel_error = np.mean(self.velocity_error_buffer)
                avg_ep_length = np.mean(self.episode_length_buffer[-self.advance_after:]) if self.episode_length_buffer else 0
                success_rate = np.mean(self.success_buffer)
                fall_rate = np.mean(self.fall_buffer) if self.fall_buffer else 1.0

                # Yaw error gate: agent must demonstrate yaw tracking to advance.
                # Without this, agent advances with 0% yaw accuracy by walking straight.
                include_yaw = getattr(self, 'include_yaw_rate', False)
                yaw_tol = self.yaw_error_tolerances[min(current_stage, len(self.yaw_error_tolerances) - 1)]
                if include_yaw and len(self.yaw_error_buffer) >= self.advance_after:
                    avg_recent_yaw_error = np.mean(self.yaw_error_buffer[-self.advance_after:])
                    yaw_ok = avg_recent_yaw_error < yaw_tol
                else:
                    # Yaw disabled or not enough data — don't block advancement
                    avg_recent_yaw_error = 0.0
                    yaw_ok = True

                # Path 1 - Standard advancement
                standard_advance = (
                    success_rate >= self.stage_success_threshold and
                    avg_recent_vel_error < current_vel_tol * 1.3 and
                    yaw_ok
                )

                # Path 2 - lenght based
                # if agent consistently survives long episodes without falling
                # TIGHTENED: Also requires minimum success rate to prevent premature advancement
                length_based_advance = (
                    avg_ep_length > min_length * 3.0 and   # Tightened from 2.5x
                    fall_rate < 0.10 and                    # Tightened from 0.15
                    avg_recent_vel_error < current_vel_tol * 1.3 and  # Tightened from 1.6
                    success_rate >= 0.20 and                # NEW: Must have some success
                    yaw_ok
                )

                # Path 3 - improvement based - DISABLED
                # This path was too lenient and allowed advancement without real skill
                improvement_advance = False

                # Path 4 - MOVEMENT-BASED - DISABLED
                # This path allowed advancement with low success rates
                movement_advance = False

                # Only advance via standard or length-based (both require real performance)
                should_advance = standard_advance or length_based_advance
                
                if should_advance and self.stage < self.max_stage:
                    old_stage = self.stage
                    self.stage += 1
                    
                    cfg = self.cfg.copy()
                    self._apply_stage_settings(cfg, self.stage)
                    
                    # Update environment settings
                    self.max_commanded_speed = self.speed_stages[self.stage]
                    self.domain_rand = cfg.get('domain_rand', self.domain_rand)
                    self.rand_mass_range = cfg.get('rand_mass_range', self.rand_mass_range)
                    self.rand_friction_range = cfg.get('rand_friction_range', self.rand_friction_range)
                    self.max_episode_steps = cfg.get('max_episode_steps', self.max_episode_steps)
                    self.random_height_init = cfg.get('random_height_init', self.random_height_init)
                    self.random_height_prob = cfg.get('random_height_prob', self.random_height_prob)
                    self.velocity_weight = cfg.get('velocity_weight', self.velocity_weight)
                    self.push_enabled = cfg.get('push_enabled', self.push_enabled)
                    self.push_magnitude_range = cfg.get('push_magnitude_range', self.push_magnitude_range)
                    self.push_interval = cfg.get('push_interval', self.push_interval)
                    self.cfg = cfg
                    
                    # Update arm penalty target for new stage
                    self._arm_penalty_target = self._arm_penalty_stage_weights[
                        min(self.stage, len(self._arm_penalty_stage_weights) - 1)
                    ]

                    # Reset success buffers for new stage
                    self.success_buffer = []
                    self.velocity_error_buffer = []
                    self.yaw_error_buffer = []
                    self.fall_buffer = []

                    # Identify which path led to advancement
                    advancement_path = []
                    if standard_advance:
                        advancement_path.append("standard")
                    if length_based_advance:
                        advancement_path.append("length-based")
                    if improvement_advance:
                        advancement_path.append("improvement")
                    if movement_advance:
                        advancement_path.append("movement-based")
                    
                    print(f"\n{'='*60}")
                    print(f" CURRICULUM ADVANCED: Stage {old_stage} -> {self.stage}")
                    print(f"  Path: {', '.join(advancement_path)}")
                    print(f"  New max speed: {self.max_commanded_speed:.2f} m/s")
                    print(f"  Push enabled: {self.push_enabled}")
                    print(f"  Velocity tolerance: ±{self.velocity_tolerances[self.stage]:.2f} m/s")
                    print(f"  Arm penalty target: {self._arm_penalty_target:.2f} (current: {self._arm_penalty_current:.3f})")
                    print(f"  Success rate was: {success_rate:.1%}")
                    print(f"  Avg velocity error was: {avg_recent_vel_error:.3f} m/s")
                    print(f"  Avg yaw error was: {avg_recent_yaw_error:.3f} rad/s (tol: {yaw_tol:.2f})")
                    print(f"  Fall rate was: {fall_rate:.1%}")
                    print(f"{'='*60}\n")
                    
                    info['curriculum_stage_advanced'] = self.stage
                    info['advancement_path'] = ', '.join(advancement_path)
                    
                elif self.stage == self.max_stage:
                    if not getattr(self, '_mastered_logged', False):
                        print(f" Stage {self.stage} MASTERED (success rate: {success_rate:.1%}, "
                              f"avg vel error: {avg_recent_vel_error:.3f} m/s)")
                        self._mastered_logged = True

            # always log curriculum progress
            info['curriculum_stage'] = self.stage
            info['curriculum_max_speed'] = self.max_commanded_speed
            info['curriculum_success_rate'] = float(np.mean(self.success_buffer)) if self.success_buffer else 0.0
            info['curriculum_avg_vel_error'] = float(np.mean(self.velocity_error_buffer)) if self.velocity_error_buffer else 0.0
            info['curriculum_avg_yaw_error'] = float(np.mean(self.yaw_error_buffer)) if self.yaw_error_buffer else 0.0
            info['curriculum_yaw_tolerance'] = self.yaw_error_tolerances[min(self.stage, len(self.yaw_error_tolerances) - 1)]
            info['curriculum_velocity_tolerance'] = self.velocity_tolerances[self.stage]
            info['curriculum_min_length'] = self.min_episode_lengths[self.stage]
            info['curriculum_episode_success'] = success
            info['curriculum_fall_rate'] = float(np.mean(self.fall_buffer)) if self.fall_buffer else 0.0
            info['curriculum_avg_ep_length'] = float(np.mean(self.episode_length_buffer[-self.advance_after:])) if self.episode_length_buffer else 0.0
            info['arm_penalty_current'] = self._arm_penalty_current
            info['arm_penalty_target'] = self._arm_penalty_target

        return obs, reward, terminated, truncated, info
    
    def evaluate_stage_readiness(self, model, n_episodes: int = 20) -> Dict[str, Any]:
        """
        Evaluate agent's readiness to advance stage.
        Runs clean episodes WITHOUT perturbations.
        
        Args:
            model: Trained model for prediction
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Temporarily disable perturbations
        original_push_enabled = self.push_enabled
        original_random_height = self.random_height_init
        original_random_height_prob = self.random_height_prob
        
        self.push_enabled = False
        self.random_height_init = False
        self.random_height_prob = 0.0
        
        eval_errors = []
        eval_errors_std = []
        eval_lengths = []
        eval_heights = []
        eval_terminated = []
        
        for ep in range(n_episodes):
            obs, _ = self.reset()
            episode_errors = []
            episode_heights = []
            done = False
            steps = 0
            terminated = False
            
            while not done and steps < 1000:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = self.step(action)
                episode_errors.append(info.get('velocity_error', 0.0))
                episode_heights.append(info.get('height', 1.4))
                done = term or trunc
                terminated = term
                steps += 1
            
            # Use stable period errors (after first 50 steps)
            if len(episode_errors) > 50:
                stable_errors = episode_errors[50:]
                avg_error = np.mean(stable_errors)
                std_error = np.std(stable_errors)
            else:
                avg_error = np.mean(episode_errors) if episode_errors else 1.0
                std_error = np.std(episode_errors) if episode_errors else 0.0
            
            eval_errors.append(avg_error)
            eval_errors_std.append(std_error)
            eval_lengths.append(steps)
            eval_heights.append(np.mean(episode_heights))
            eval_terminated.append(terminated)
        
        # Restore settings
        self.push_enabled = original_push_enabled
        self.random_height_init = original_random_height
        self.random_height_prob = original_random_height_prob
        
        current_tol = self.velocity_tolerances[self.stage]
        success_rate = np.mean([e < current_tol for e in eval_errors])
        fall_rate = np.mean(eval_terminated)
        
        return {
            'mean_error': float(np.mean(eval_errors)),
            'std_error': float(np.std(eval_errors)),
            'mean_error_std': float(np.mean(eval_errors_std)),
            'success_rate': float(success_rate),
            'fall_rate': float(fall_rate),
            'mean_length': float(np.mean(eval_lengths)),
            'mean_height': float(np.mean(eval_heights)),
            'velocity_tolerance': current_tol,
            'stage': self.stage,
            'n_episodes': n_episodes,
            'recommendation': 'advance' if success_rate >= 0.4 and fall_rate < 0.2 else 'continue_training'
        }
    
    def get_curriculum_info(self):
        """Get current curriculum state for logging."""
        current_stage = min(self.stage, len(self.speed_stages) - 1)
        return {
            'stage': self.stage,
            'max_speed': self.max_commanded_speed,
            'standing_probability': self.standing_probability[current_stage],
            'velocity_tolerance': self.velocity_tolerances[current_stage],
            'min_episode_length': self.min_episode_lengths[current_stage],
            'success_rate': np.mean(self.success_buffer) if self.success_buffer else 0.0,
            'avg_velocity_error': np.mean(self.velocity_error_buffer) if self.velocity_error_buffer else 0.0,
            'fall_rate': np.mean(self.fall_buffer) if self.fall_buffer else 0.0,
            'total_episodes': self.total_episodes,
            'push_enabled': self.push_enabled,
            'arm_penalty_current': self._arm_penalty_current,
            'arm_penalty_target': self._arm_penalty_target,
        }


def make_walking_curriculum_env(render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    """Create walking curriculum environment."""
    return WalkingCurriculumEnv(render_mode=render_mode, config=config)
