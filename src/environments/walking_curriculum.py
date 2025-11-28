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
    0: 0.0 m/s max - Standing only (leverage standing model)
    1: 0.5 m/s max - Slow walk
    2: 1.0 m/s max - Normal walk
    3: 1.5 m/s max - Fast walk
    4: 2.0 m/s max - Light jog
    5: 2.5 m/s max - Jog
    6: 3.0 m/s max - Fast jog / run
    
    Advancement: avg velocity error < tolerance over last N episodes AND >70% episodes > min length
    """

    def __init__(self, render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).copy()
        self.stage = int(cfg.get('curriculum_start_stage', 0))
        self.max_stage = int(cfg.get('curriculum_max_stage', 6))
        self.advance_after = int(cfg.get('curriculum_advance_after', 20))
        self.success_buffer = []
        self.velocity_error_buffer = []
        self.stage_success_threshold = float(cfg.get('curriculum_success_rate', 0.70))

        # Speed stages (m/s) - IMPORTANT: Stage 0 now includes walking practice
        # Stage 0: Mix of standing (50%) and slow walking (50% at 0-0.5 m/s)
        self.speed_stages = [0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # Stage 0 now has 0.5 m/s max
        
        # Probability of standing command per stage (rest is walking)
        self.standing_probability = [0.4, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0]
        
        # Velocity error tolerances (m/s) - relaxed for early stages
        self.velocity_tolerances = [0.45, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]  # More lenient
        
        # Minimum episode lengths - relaxed for early stages
        self.min_episode_lengths = [500, 600, 800, 800, 1000, 1000, 1000]
        
        # Height tolerances (relaxed for walking - natural walking height is lower than standing)
        # Standing: 1.4m, Walking typically: 1.2-1.3m → need ±0.25m tolerance
        self.height_tolerances = [0.25, 0.25, 0.25, 0.20, 0.20, 0.18, 0.18]
        
        self._apply_stage_settings(cfg, self.stage)
        super().__init__(render_mode=render_mode, config=cfg)
        
        # Override max_commanded_speed from curriculum
        self.max_commanded_speed = self.speed_stages[self.stage]
        
        print(f"  Walking curriculum initialized at stage {self.stage}")
        print(f"  Max speed: {self.max_commanded_speed:.2f} m/s")
        print(f"  Standing probability: {self.standing_probability[self.stage]:.0%}")
        print(f"  Velocity tolerance: ±{self.velocity_tolerances[self.stage]:.2f} m/s")
        print(f"  Min episode length: {self.min_episode_lengths[self.stage]} steps")

    def _apply_stage_settings(self, cfg: Dict[str, Any], stage: int) -> None:
        """Configure curriculum stage with progressive difficulty."""
        # Set max speed for this stage
        cfg['max_commanded_speed'] = self.speed_stages[min(stage, len(self.speed_stages) - 1)]
        
        if stage == 0:
            # Stage 0: Mixed standing (40%) and slow walking (60% at 0-0.5 m/s)
            cfg['domain_rand'] = False
            cfg['max_episode_steps'] = 1500  # Shorter episodes for faster iteration
            cfg['random_height_init'] = True
            cfg['random_height_prob'] = 0.15  # Reduced - focus on walking first
            cfg['velocity_weight'] = 5.0  # Moderate penalty for velocity error
        elif stage <= 2:
            # Early walking stages: focus on achieving commanded velocity
            cfg['domain_rand'] = False
            cfg['max_episode_steps'] = 2000
            cfg['random_height_init'] = True
            cfg['random_height_prob'] = 0.1
            cfg['velocity_weight'] = 5.0  # Same moderate weight
        elif stage <= 4:
            # Mid walking stages: moderate randomization
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.98, 1.02]
            cfg['rand_friction_range'] = [0.98, 1.02]
            cfg['max_episode_steps'] = 2500
            cfg['random_height_init'] = True
            cfg['random_height_prob'] = 0.1
            cfg['velocity_weight'] = 4.0  # Slightly reduced
        else:
            # Final stages: full difficulty
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.95, 1.05]
            cfg['rand_friction_range'] = [0.95, 1.05]
            cfg['max_episode_steps'] = 3000
            cfg['random_height_init'] = True
            cfg['random_height_prob'] = 0.1
            cfg['velocity_weight'] = 3.0  # Lower for stability at high speeds

    def reset(self, seed: Optional[int] = None):
        # Update max speed for current stage
        current_stage = min(self.stage, len(self.speed_stages) - 1)
        self.max_commanded_speed = self.speed_stages[current_stage]
        
        # Mix standing and walking commands based on curriculum stage
        standing_prob = self.standing_probability[current_stage]
        if np.random.random() < standing_prob:
            # Force standing command for this episode
            self.fixed_command = (0.0, 0.0)
        else:
            # Allow walking command (sampled in parent reset)
            self.fixed_command = None
        
        obs, info = super().reset(seed=seed)
        
        # Clear fixed_command after reset so next episode can choose fresh
        self.fixed_command = None
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        done = bool(terminated or truncated)
        if done:
            height = info.get('height', 0.0)
            velocity_error = info.get('velocity_error', 0.0)
            
            current_stage = min(self.stage, len(self.speed_stages) - 1)
            current_speed = self.speed_stages[current_stage]
            current_vel_tol = self.velocity_tolerances[current_stage]
            min_length = self.min_episode_lengths[current_stage]
            height_tol = self.height_tolerances[current_stage]
            
            # Get average velocity error over episode
            avg_vel_error = np.mean(self.reward_history.get('velocity_tracking', [0.0])) if self.reward_history.get('velocity_tracking') else 0.0
            # Convert from reward to error (reward = -error^2 * weight)
            # Approximate: just use final velocity error
            
            # SUCCESS CRITERIA (all must be met):
            # 1. Velocity error within tolerance
            vel_ok = velocity_error < current_vel_tol
            # 2. Height maintained (within tolerance of target)
            height_ok = abs(height - self.base_target_height) < height_tol
            # 3. Episode lasted long enough
            long_enough = self.current_step >= min_length
            # 4. Not terminated due to falling
            not_fallen = not terminated
            
            success = bool(vel_ok and height_ok and long_enough and not_fallen)

            self.success_buffer.append(1 if success else 0)
            self.velocity_error_buffer.append(velocity_error)
            
            if len(self.success_buffer) > self.advance_after:
                self.success_buffer = self.success_buffer[-self.advance_after:]
                self.velocity_error_buffer = self.velocity_error_buffer[-self.advance_after:]

            # ADVANCE: Check both success rate and average velocity error
            avg_recent_vel_error = np.mean(self.velocity_error_buffer) if self.velocity_error_buffer else float('inf')
            success_rate = np.mean(self.success_buffer) if self.success_buffer else 0.0
            
            if (len(self.success_buffer) == self.advance_after and 
                success_rate >= self.stage_success_threshold and
                avg_recent_vel_error < current_vel_tol):
                
                if self.stage < self.max_stage:
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
                    self.cfg = cfg
                    
                    # Reset success buffers for new stage
                    self.success_buffer = []
                    self.velocity_error_buffer = []
                    
                    print(f"\n{'='*60}")
                    print(f" CURRICULUM ADVANCED: Stage {old_stage} → {self.stage}")
                    print(f"  New max speed: {self.max_commanded_speed:.2f} m/s")
                    print(f"  Standing probability: {self.standing_probability[self.stage]:.0%}")
                    print(f"  Velocity tolerance: ±{self.velocity_tolerances[self.stage]:.2f} m/s")
                    print(f"  Min episode length: {self.min_episode_lengths[self.stage]} steps")
                    print(f"  Domain randomization: {self.domain_rand}")
                    print(f"{'='*60}\n")
                    
                    info['curriculum_stage_advanced'] = self.stage
                elif self.stage == self.max_stage:
                    print(f" Stage {self.stage} MASTERED (success rate: {success_rate:.1%}, "
                          f"avg vel error: {avg_recent_vel_error:.3f} m/s)")

            # ENHANCED INFO: Always log curriculum progress
            info['curriculum_stage'] = self.stage
            info['curriculum_max_speed'] = self.max_commanded_speed
            info['curriculum_success_rate'] = float(success_rate)
            info['curriculum_avg_vel_error'] = float(avg_recent_vel_error) if self.velocity_error_buffer else 0.0
            info['curriculum_velocity_tolerance'] = self.velocity_tolerances[self.stage]
            info['curriculum_min_length'] = self.min_episode_lengths[self.stage]
            info['curriculum_episode_success'] = success

        return obs, reward, terminated, truncated, info
    
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
        }


def make_walking_curriculum_env(render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    """Create walking curriculum environment."""
    return WalkingCurriculumEnv(render_mode=render_mode, config=config)

