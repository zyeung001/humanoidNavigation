"""
walking_curriculum.py

Curriculum learning for walking task.
Progressive stages from standing (0 m/s) to fast walking (3 m/s).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np

from .walking_env import WalkingEnv


class WalkingCurriculumEnv(WalkingEnv):
    """
    Curriculum Stages for Walking:
    0: 0.3 m/s max - Very slow walk (NO perturbations)
    1: 0.6 m/s max - Slow walk (gentle perturbations)
    2: 1.0 m/s max - Normal walk
    3: 1.5 m/s max - Fast walk
    4: 2.0 m/s max - Light jog
    5: 2.5 m/s max - Jog
    6: 3.0 m/s max - Fast jog / run
    
    Advancement: Multiple paths to advance:
    1. Standard: success_rate >= threshold AND avg_vel_error < tolerance
    2. Length-based: avg_episode_length > 2x min AND fall_rate < 15%
    3. Improvement-based: consistent improvement over training window
    """

    def __init__(self, render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).copy()
        self.stage = int(cfg.get('curriculum_start_stage', 0))
        self.max_stage = int(cfg.get('curriculum_max_stage', 6))
        self.advance_after = int(cfg.get('curriculum_advance_after', 20))  # Increased for more stable estimate
        self.success_buffer = []
        self.velocity_error_buffer = []
        self.episode_length_buffer = []
        self.stage_success_threshold = float(cfg.get('curriculum_success_rate', 0.35))  # RELAXED: 35%

        # Speed stages (m/s) - Progressive difficulty
        self.speed_stages = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        # Probability of standing command per stage
        # Stage 0-1: More standing practice to build stability
        self.standing_probability = [0.08, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0]
        
        # RELAXED velocity error tolerances for curriculum advancement
        # Stage 0 is VERY lenient - we just want the agent to START moving
        # Key insight: 0.3 m/s commanded, 0.9 m/s tolerance = agent can be 3x off and still pass
        self.velocity_tolerances = [0.9, 0.75, 0.6, 0.5, 0.45, 0.4, 0.35]
        
        # Minimum episode lengths - RELAXED for curriculum progression
        # Stage 0: Just survive 60 steps (low bar to start)
        self.min_episode_lengths = [60, 100, 150, 200, 300, 400, 500]
        
        # Height tolerances (walking naturally has lower COM)
        self.height_tolerances = [0.45, 0.40, 0.35, 0.30, 0.27, 0.24, 0.20]
        
        # Direction diversity
        self.direction_diversity = cfg.get('direction_diversity', True)
        
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
        
        FIX 3: Stage 0 has NO perturbations for stability-first learning.
        """
        cfg['max_commanded_speed'] = self.speed_stages[min(stage, len(self.speed_stages) - 1)]
        
        if stage == 0:
            # FIX 3: Stage 0 - Focus on basic walking, NO perturbations
            cfg['domain_rand'] = False
            cfg['max_episode_steps'] = 2000  # Longer episodes for learning
            cfg['push_enabled'] = False  # CRITICAL: No pushes during Stage 0
            cfg['random_height_init'] = False  # CRITICAL: Start at normal height
            cfg['random_height_prob'] = 0.0
            cfg['velocity_weight'] = 6.0  # Higher tracking weight
        elif stage == 1:
            # Stage 1: Introduce gentle perturbations
            cfg['domain_rand'] = False
            cfg['max_episode_steps'] = 2000
            cfg['push_enabled'] = True
            cfg['push_magnitude_range'] = [20.0, 50.0]  # Gentler pushes
            cfg['push_interval'] = 400  # Less frequent
            cfg['random_height_init'] = True
            cfg['random_height_prob'] = 0.10  # Only 10%
            cfg['velocity_weight'] = 5.0
        elif stage == 2:
            # Stage 2: Normal difficulty
            cfg['domain_rand'] = False
            cfg['max_episode_steps'] = 2500
            cfg['push_enabled'] = True
            cfg['push_magnitude_range'] = [30.0, 80.0]
            cfg['push_interval'] = 300
            cfg['random_height_init'] = True
            cfg['random_height_prob'] = 0.15
            cfg['velocity_weight'] = 5.0
        elif stage <= 4:
            # Mid walking stages: moderate randomization
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.98, 1.02]
            cfg['rand_friction_range'] = [0.98, 1.02]
            cfg['max_episode_steps'] = 2500
            cfg['push_enabled'] = True
            cfg['push_magnitude_range'] = [40.0, 100.0]
            cfg['push_interval'] = 250
            cfg['random_height_init'] = True
            cfg['random_height_prob'] = 0.20
            cfg['velocity_weight'] = 4.0
        else:
            # Final stages: full difficulty
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.95, 1.05]
            cfg['rand_friction_range'] = [0.95, 1.05]
            cfg['max_episode_steps'] = 3000
            cfg['push_enabled'] = True
            cfg['push_magnitude_range'] = [50.0, 150.0]
            cfg['push_interval'] = 200
            cfg['random_height_init'] = True
            cfg['random_height_prob'] = 0.25
            cfg['velocity_weight'] = 3.0

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
            
            speed = np.random.uniform(0.1, self.max_commanded_speed)
            
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
            yaw_rate = np.random.uniform(-max_yaw, max_yaw) * (0.3 + 0.7 * current_stage / 6.0)
            self.fixed_command = (vx, vy, yaw_rate)
        else:
            # Let command generator handle it
            self.fixed_command = None
        
        obs, info = super().reset(seed=seed)
        self.fixed_command = None
        
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
            
            # Warmup period with relaxed criteria
            if self.total_episodes < self.warmup_episodes:
                vel_tolerance_multiplier = 1.5  # 50% more lenient during warmup
            else:
                vel_tolerance_multiplier = 1.2  # Normal 20% buffer
            
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
            # For Stage 0-1, reward any movement attempt
            actual_speed_info = info.get('actual_speed', 0.0)
            attempted_movement = actual_speed_info > 0.08 or self.commanded_speed < 0.1
            
            # Stage 0-1: Be very lenient to get the agent moving
            if current_stage <= 1:
                # Pass if: survived + attempted movement + (velocity OK OR height OK)
                success = bool(
                    not_fallen and 
                    long_enough and
                    (vel_ok or (height_ok and attempted_movement))
                )
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
                # Key insight: For Stage 0, just getting the agent to MOVE is a win
                # We can refine accuracy in later stages
                movement_advance = False
                if current_stage <= 1:
                    # Check if agent is actually moving (not just standing)
                    if self.velocity_error_buffer:
                        # If commanded 0.3 m/s and velocity error < 0.7, agent is moving somewhat
                        avg_error = np.mean(self.velocity_error_buffer[-10:])
                        is_moving = avg_error < current_speed + 0.2  # If error < cmd + 0.2, agent moved
                        
                        movement_advance = (
                            is_moving and
                            fall_rate < 0.25 and  # Can fall sometimes while learning
                            avg_ep_length > min_length * 1.5 and  # Survives reasonably
                            success_rate >= 0.20  # Very low bar for Stage 0
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
