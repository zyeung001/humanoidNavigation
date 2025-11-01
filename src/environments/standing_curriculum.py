"""
StandingEnv with HEIGHT-BASED curriculum progression.

CRITICAL IMPROVEMENT: Gradually increase target height instead of jumping stages.

This prevents the agent from learning stability at wrong height and forces
progressive learning from 1.0m up to 1.4m target.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np

from .standing_env import StandingEnv


class StandingCurriculumEnv(StandingEnv):
    """StandingEnv with HEIGHT-BASED curriculum.

    Key innovation: curriculum gradually increases target height from easy to hard.
    
    Stages:
    0: 1.0m target - Learn basic balance at low, stable height
    1: 1.2m target - Intermediate height requires more control
    2: 1.3m target - Approaching full height, significant balance challenge
    3: 1.4m target - Full standing, maximum difficulty
    """

    def __init__(self, render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).copy()
        self.stage = int(cfg.get('curriculum_start_stage', 0))
        self.max_stage = int(cfg.get('curriculum_max_stage', 3))
        self.advance_after = int(cfg.get('curriculum_advance_after', 20))  # More patience
        self.success_buffer = []
        self.stage_success_threshold = float(cfg.get('curriculum_success_rate', 0.75))

        # Define height targets for each stage
        self.height_targets = [1.0, 1.2, 1.3, 1.4]
        
        # Apply initial stage settings onto config
        self._apply_stage_settings(cfg, self.stage)

        super().__init__(render_mode=render_mode, config=cfg)
        
        # Override base target height with curriculum height
        self.base_target_height = self.height_targets[self.stage]
        print(f"Curriculum initialized at stage {self.stage}, target height: {self.base_target_height:.2f}m")

    def _apply_stage_settings(self, cfg: Dict[str, Any], stage: int) -> None:
        """Configure curriculum stage - simpler, just domain randomization."""
        # All stages use same basic settings, only target height changes
        if stage <= 1:
            # Early stages: easier conditions
            cfg['domain_rand'] = False
            cfg['max_episode_steps'] = 3000
        elif stage == 2:
            # Mid stage: light domain randomization
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.98, 1.02]
            cfg['rand_friction_range'] = [0.98, 1.02]
            cfg['max_episode_steps'] = 4000
        else:
            # Final stage: full difficulty
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.95, 1.05]
            cfg['rand_friction_range'] = [0.95, 1.05]
            cfg['max_episode_steps'] = 5000

    def reset(self, seed: Optional[int] = None):
        # Update target height for current stage
        self.base_target_height = self.height_targets[min(self.stage, len(self.height_targets) - 1)]
        return super().reset(seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        done = bool(terminated or truncated)
        if done:
            # Success criterion: standing close to CURRENT stage target height
            height = info.get('height', 0.0)
            current_target = self.height_targets[min(self.stage, len(self.height_targets) - 1)]
            height_error = abs(height - current_target)
            
            # More lenient success criteria for curriculum
            height_ok = height_error < 0.20  # 20cm tolerance
            long_enough = self.current_step >= min(500, self.max_episode_steps // 3)
            success = bool(height_ok and long_enough and not terminated)

            self.success_buffer.append(1 if success else 0)
            if len(self.success_buffer) > self.advance_after:
                self.success_buffer = self.success_buffer[-self.advance_after:]

            # Advance if success rate met
            if len(self.success_buffer) == self.advance_after and np.mean(self.success_buffer) >= self.stage_success_threshold:
                if self.stage < self.max_stage:
                    self.stage += 1
                    self.base_target_height = self.height_targets[min(self.stage, len(self.height_targets) - 1)]
                    
                    cfg = self.cfg.copy()
                    self._apply_stage_settings(cfg, self.stage)
                    
                    # Update local settings
                    self.domain_rand = cfg.get('domain_rand', self.domain_rand)
                    self.rand_mass_range = cfg.get('rand_mass_range', self.rand_mass_range)
                    self.rand_friction_range = cfg.get('rand_friction_range', self.rand_friction_range)
                    self.max_episode_steps = cfg.get('max_episode_steps', self.max_episode_steps)
                    self.cfg = cfg
                    
                    print(f"✓ Curriculum advanced to stage {self.stage}, new target height: {self.base_target_height:.2f}m")
                    info['curriculum_stage_advanced'] = self.stage

            info['curriculum_stage'] = self.stage
            info['curriculum_target_height'] = self.base_target_height

        return obs, reward, terminated, truncated, info


def make_standing_curriculum_env(render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    return StandingCurriculumEnv(render_mode=render_mode, config=config)