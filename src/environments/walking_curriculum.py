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
        self.advance_after = int(cfg.get('curriculum_advance_after', 15))  # Reduced for faster iteration
        self.success_buffer = []
        self.velocity_error_buffer = []
        self.episode_length_buffer = []  # Track episode lengths for advancement
        self.stage_success_threshold = float(cfg.get('curriculum_success_rate', 0.50))  # RELAXED: 50% instead of 70%

        # Speed stages (m/s) - Progressive difficulty
        # Stage 0: 0.3 m/s - Very slow walk (easier to learn)
        # Stage 1: 0.6 m/s - Slow walk  
        # Stage 2: 1.0 m/s - Normal walk
        # Stage 3: 1.5 m/s - Fast walk
        # Stage 4: 2.0 m/s - Light jog
        # Stage 5: 2.5 m/s - Jog
        # Stage 6: 3.0 m/s - Fast jog / run
        self.speed_stages = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        # Probability of standing command per stage (rest is walking)
        # FIXED: Very low standing probability - we want to learn walking!
        self.standing_probability = [0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Velocity error tolerances (m/s) - RELAXED significantly for early stages
        # The model needs time to learn, so be lenient at first
        self.velocity_tolerances = [0.6, 0.55, 0.5, 0.45, 0.4, 0.4, 0.4]
        
        # Minimum episode lengths - MUCH MORE RELAXED for curriculum progression
        # Start very easy and increase gradually
        self.min_episode_lengths = [100, 150, 200, 300, 400, 500, 600]
        
        # Height tolerances (walking naturally has lower COM)
        # Standing: 1.4m, Walking: 1.1-1.3m typical
        self.height_tolerances = [0.35, 0.35, 0.30, 0.28, 0.25, 0.22, 0.20]
        
        # Direction diversity - train in all directions from the start
        self.direction_diversity = cfg.get('direction_diversity', True)
        
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
        elif self.direction_diversity:
            # Sample diverse velocity commands including:
            # - Forward (most common for learning)
            # - Diagonal (45°, 135°, etc.)
            # - Lateral (90°, 270°)
            # - Backward (rare, harder)
            direction_type = np.random.choice(['forward', 'diagonal', 'lateral', 'backward', 'random'],
                                             p=[0.5, 0.25, 0.15, 0.02, 0.08])
            
            speed = np.random.uniform(0.1, self.max_commanded_speed)
            
            if direction_type == 'forward':
                angle = np.random.uniform(-np.pi/6, np.pi/6)  # ±30° from forward
            elif direction_type == 'diagonal':
                base_angles = [np.pi/4, 3*np.pi/4, -np.pi/4, -3*np.pi/4]
                angle = np.random.choice(base_angles) + np.random.uniform(-np.pi/8, np.pi/8)
            elif direction_type == 'lateral':
                angle = np.random.choice([np.pi/2, -np.pi/2]) + np.random.uniform(-np.pi/8, np.pi/8)
            elif direction_type == 'backward':
                angle = np.pi + np.random.uniform(-np.pi/6, np.pi/6)
            else:  # random
                angle = np.random.uniform(-np.pi, np.pi)
            
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            self.fixed_command = (vx, vy)
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
            
            # Track episode length for progress
            self.episode_length_buffer.append(self.current_step)
            if len(self.episode_length_buffer) > self.advance_after * 2:
                self.episode_length_buffer = self.episode_length_buffer[-self.advance_after:]
            
            # RELAXED SUCCESS CRITERIA - focus on survival and basic tracking
            # Only need to meet 2 out of 3 criteria for "soft success"
            criteria_met = 0
            
            # 1. Velocity error within tolerance (or close)
            vel_ok = velocity_error < current_vel_tol * 1.2  # 20% buffer
            if vel_ok:
                criteria_met += 1
            
            # 2. Height maintained (allow more variation during walking)
            height_ok = height > 1.0 and height < 1.6  # Wide range for walking
            if height_ok:
                criteria_met += 1
            
            # 3. Episode lasted long enough
            long_enough = self.current_step >= min_length
            if long_enough:
                criteria_met += 1
            
            # 4. Not terminated due to falling (bonus criteria)
            not_fallen = not terminated
            
            # SUCCESS: Either didn't fall + 2 criteria, or all 3 criteria
            success = bool((not_fallen and criteria_met >= 2) or criteria_met == 3)

            self.success_buffer.append(1 if success else 0)
            self.velocity_error_buffer.append(velocity_error)
            
            if len(self.success_buffer) > self.advance_after:
                self.success_buffer = self.success_buffer[-self.advance_after:]
                self.velocity_error_buffer = self.velocity_error_buffer[-self.advance_after:]

            # ADVANCE: Check success rate OR improvement in episode length
            avg_recent_vel_error = np.mean(self.velocity_error_buffer) if self.velocity_error_buffer else float('inf')
            avg_ep_length = np.mean(self.episode_length_buffer[-self.advance_after:]) if self.episode_length_buffer else 0
            success_rate = np.mean(self.success_buffer) if self.success_buffer else 0.0
            
            # Advancement conditions (more lenient):
            # 1. Standard: success rate >= threshold AND velocity error < tolerance
            # 2. Alternative: avg episode length > 2x min AND not falling frequently
            standard_advance = (success_rate >= self.stage_success_threshold and 
                               avg_recent_vel_error < current_vel_tol * 1.3)
            length_based_advance = (avg_ep_length > min_length * 2.0 and 
                                   success_rate >= 0.3)  # 30% is enough if episodes are long
            
            if (len(self.success_buffer) == self.advance_after and 
                (standard_advance or length_based_advance)):
                
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

