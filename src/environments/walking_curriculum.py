"""
walking_curriculum.py

Curriculum learning for walking task.
Progressive 3-stage curriculum for ~2ft (0.6m) humanoid sim-to-real transfer.
Speeds are realistic for small bipedal robots (max ~0.8 m/s).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
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
        self.velocity_tolerances = [0.12, 0.20, 0.30]
        self.min_episode_lengths = [200, 300, 400]
        self.height_tolerances = [0.45, 0.40, 0.35]
        
        # Direction diversity - DISABLED by default for Stage 0
        # Stage 0 should be boring: forward only, fixed speed
        self.direction_diversity = cfg.get('direction_diversity', False)
        
        # Episode velocity error tracking
        self.episode_velocity_errors = []
        self.stabilization_steps = 50  # Skip first N steps when computing average
        
        # Warmup tracking
        self.total_episodes = 0
        self.warmup_episodes = int(cfg.get('warmup_episodes', 50))
        
        # Fall tracking for alternative advancement
        self.fall_buffer = []  # Track terminated episodes (falls)
        
        self._apply_stage_settings(cfg, self.stage)
        super().__init__(render_mode=render_mode, config=cfg)
        
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
        Stage 0: Baby steps (0.15 m/s) - no perturbations, forward only
        Stage 1: Normal walk (0.40 m/s) - light perturbations, wider direction
        Stage 2: Brisk walk (0.80 m/s) - moderate perturbations, full direction
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
        else:
            # Stage 2 - Brisk walking, moderate perturbations, domain rand
            cfg['domain_rand'] = True
            cfg['max_episode_steps'] = 3000
            cfg['push_enabled'] = True
            cfg['push_magnitude_range'] = [20.0, 80.0]
            cfg['push_interval'] = 300
            cfg['random_height_init'] = True
            cfg['random_height_prob'] = 0.15
            cfg['velocity_weight'] = 5.0

    def reset(self, seed: Optional[int] = None):
        # Clear episode velocity error tracking
        self.episode_velocity_errors = []
        
        # Update max speed for current stage
        current_stage = min(self.stage, len(self.speed_stages) - 1)
        self.max_commanded_speed = self.speed_stages[current_stage]
        
        # Update command generator ranges based on curriculum stage
        if hasattr(self, 'command_generator') and self.command_generator is not None:
            max_yaw = getattr(self, 'max_yaw_rate', 1.0)
            self.command_generator.update_curriculum_ranges([
                (0.0, self.max_commanded_speed),
                (-self.max_commanded_speed * 0.5, self.max_commanded_speed * 0.5),
                (-max_yaw, max_yaw)
            ])
        
        # Mix standing and walking commands based on curriculum stage
        standing_prob = self.standing_probability[current_stage]
        if np.random.random() < standing_prob:
            # Force standing command with yaw_rate = 0
            self.fixed_command = (0.0, 0.0, 0.0)
        elif self.direction_diversity:
            direction_type = np.random.choice(['forward', 'diagonal', 'lateral', 'backward', 'random'],
                                             p=[0.5, 0.25, 0.15, 0.02, 0.08])

            # Stage 0-1: Fixed speed at max to avoid variance
            # Later stages: Allow speed variation
            if current_stage <= 1:
                speed = self.max_commanded_speed  # Fixed speed
            else:
                speed = np.random.uniform(0.2, self.max_commanded_speed)
            
            if direction_type == 'forward':
                angle = np.random.uniform(-np.pi/6, np.pi/6)
            elif direction_type == 'diagonal':
                base_angles = [np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4]
                angle = np.random.choice(base_angles) + np.random.uniform(-np.pi/8, np.pi/8)
            elif direction_type == 'lateral':
                angle = np.random.choice([np.pi/2, -np.pi/2]) + np.random.uniform(-np.pi/8, np.pi/8)
            elif direction_type == 'backward':
                angle = np.pi + np.random.uniform(-np.pi/6, np.pi/6)
            else:
                angle = np.random.uniform(-np.pi, np.pi)
            
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            # Sample yaw rate with reduced probability for early stages
            max_yaw = getattr(self, 'max_yaw_rate', 1.0)
            yaw_rate = np.random.uniform(-max_yaw, max_yaw) * (0.3 + 0.7 * current_stage / max(self.max_stage, 1))
            self.fixed_command = (vx, vy, yaw_rate)
        else:
            # No direction diversity
            if current_stage == 0:
                # Stage 0: Forward only, fixed speed, tiny angle variance
                # This prevents standing-still exploit while keeping the task simple
                angle = np.random.uniform(-np.pi/12, np.pi/12)  # ±15 degrees
                speed = self.max_commanded_speed  # Fixed, not sampled
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)
                yaw_rate = 0.0
                self.fixed_command = (vx, vy, yaw_rate)
            elif current_stage == 1:
                # Stage 1: Slightly more variety in direction
                angle = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees
                speed = self.max_commanded_speed
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)
                yaw_rate = 0.0
                self.fixed_command = (vx, vy, yaw_rate)
            elif current_stage == 2:
                # Stage 2: Full direction range, variable speed
                angle = np.random.uniform(-np.pi/4, np.pi/4)  # ±45 degrees
                speed = np.random.uniform(0.2, self.max_commanded_speed)
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)
                max_yaw = getattr(self, 'max_yaw_rate', 1.0)
                yaw_rate = np.random.uniform(-max_yaw * 0.5, max_yaw * 0.5)
                self.fixed_command = (vx, vy, yaw_rate)
            else:
                # Later stages: let command generator handle variety
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
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Track velocity error every step for episode average
        if 'velocity_error' in info:
            self.episode_velocity_errors.append(info['velocity_error'])

        done = bool(terminated or truncated)
        if done:
            self.total_episodes += 1
            height = info.get('height', 0.0)
            
            current_stage = min(self.stage, len(self.speed_stages) - 1)
            current_speed = self.speed_stages[current_stage]
            current_vel_tol = self.velocity_tolerances[current_stage]
            min_length = self.min_episode_lengths[current_stage]
            height_tol = self.height_tolerances[current_stage]
            
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
            
            # Update info with episode-average error
            info['velocity_error'] = velocity_error
            info['velocity_error_std'] = velocity_error_std
            info['velocity_error_full_episode'] = np.mean(self.episode_velocity_errors) if self.episode_velocity_errors else 0.0
            
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
                
                # Path 1 - Standard advancement
                standard_advance = (
                    success_rate >= self.stage_success_threshold and 
                    avg_recent_vel_error < current_vel_tol * 1.3
                )
                
                # Path 2 - lenght based
                # if agent consistently survives long episodes without falling
                length_based_advance = (
                    avg_ep_length > min_length * 2.5 and  # 2.5x required length
                    fall_rate < 0.15 and                   # <15% falls
                    avg_recent_vel_error < current_vel_tol * 1.6  # More lenient error
                )
                
                # Path 3 - improvement based
                # if agent shows consistent improvement
                improvement_advance = False
                if len(self.velocity_error_buffer) >= self.advance_after:
                    early_errors = self.velocity_error_buffer[:self.advance_after // 2]
                    recent_errors = self.velocity_error_buffer[-(self.advance_after // 2):]
                    improvement = np.mean(early_errors) - np.mean(recent_errors)
                    improvement_advance = (
                        improvement > 0.08 and  # At least 0.08 m/s improvement
                        np.mean(recent_errors) < current_vel_tol * 1.5 and
                        success_rate >= 0.25  # RELAXED from 0.30
                    )
                
                # Path 4 - MOVEMENT-BASED (Stage 0-1 only)
                # Key insight: For Stage 0, agent must actually be tracking velocity
                # Tightened criteria to prevent standing-still exploitation
                movement_advance = False
                if current_stage <= 1:
                    # Check if agent is actually tracking velocity (not just standing)
                    if self.velocity_error_buffer:
                        avg_error = np.mean(self.velocity_error_buffer[-10:])
                        # is_tracking: velocity error must be within tolerance
                        is_tracking = avg_error < current_vel_tol * 1.2

                        movement_advance = (
                            is_tracking and
                            fall_rate < 0.20 and           # Tightened from 0.25
                            avg_ep_length > min_length * 2.0 and  # Tightened from 1.5
                            success_rate >= 0.35           # Tightened from 0.20
                        )
                
                # advance if any success
                should_advance = standard_advance or length_based_advance or improvement_advance or movement_advance
                
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
                    
                    # Reset success buffers for new stage
                    self.success_buffer = []
                    self.velocity_error_buffer = []
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
                    print(f" CURRICULUM ADVANCED: Stage {old_stage} → {self.stage}")
                    print(f"  Path: {', '.join(advancement_path)}")
                    print(f"  New max speed: {self.max_commanded_speed:.2f} m/s")
                    print(f"  Push enabled: {self.push_enabled}")
                    print(f"  Velocity tolerance: ±{self.velocity_tolerances[self.stage]:.2f} m/s")
                    print(f"  Success rate was: {success_rate:.1%}")
                    print(f"  Avg velocity error was: {avg_recent_vel_error:.3f} m/s")
                    print(f"  Fall rate was: {fall_rate:.1%}")
                    print(f"{'='*60}\n")
                    
                    info['curriculum_stage_advanced'] = self.stage
                    info['advancement_path'] = ', '.join(advancement_path)
                    
                elif self.stage == self.max_stage:
                    print(f" Stage {self.stage} MASTERED (success rate: {success_rate:.1%}, "
                          f"avg vel error: {avg_recent_vel_error:.3f} m/s)")

            # always log curriculum progress
            info['curriculum_stage'] = self.stage
            info['curriculum_max_speed'] = self.max_commanded_speed
            info['curriculum_success_rate'] = float(np.mean(self.success_buffer)) if self.success_buffer else 0.0
            info['curriculum_avg_vel_error'] = float(np.mean(self.velocity_error_buffer)) if self.velocity_error_buffer else 0.0
            info['curriculum_velocity_tolerance'] = self.velocity_tolerances[self.stage]
            info['curriculum_min_length'] = self.min_episode_lengths[self.stage]
            info['curriculum_episode_success'] = success
            info['curriculum_fall_rate'] = float(np.mean(self.fall_buffer)) if self.fall_buffer else 0.0
            info['curriculum_avg_ep_length'] = float(np.mean(self.episode_length_buffer[-self.advance_after:])) if self.episode_length_buffer else 0.0

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
        }


def make_walking_curriculum_env(render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    """Create walking curriculum environment."""
    return WalkingCurriculumEnv(render_mode=render_mode, config=config)
