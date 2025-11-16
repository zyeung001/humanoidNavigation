"""

standing_curriculum.py
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np

from .standing_env import StandingEnv


class StandingCurriculumEnv(StandingEnv):
    """
    Stages (finer progression):
    0: 1.00m target - Basic balance at low height
    1: 1.15m target - Early height increase
    2: 1.25m target - Mid-range balance challenge
    3: 1.35m target - High balance precision required
    4: 1.40m target - FINAL TARGET, maximum difficulty
    """

    def __init__(self, render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).copy()
        self.stage = int(cfg.get('curriculum_start_stage', 0))
        self.max_stage = int(cfg.get('curriculum_max_stage', 4))
        self.advance_after = int(cfg.get('curriculum_advance_after', 20))  
        self.success_buffer = []
        self.stage_success_threshold = float(cfg.get('curriculum_success_rate', 0.60))  


        self.height_targets = [1.00, 1.15, 1.25, 1.35, 1.40]
        
        self.height_tolerances = [0.20, 0.15, 0.12, 0.10, 0.08]  
        
        self.min_episode_lengths = [100, 200, 400, 800, 1200] 
        
        self._apply_stage_settings(cfg, self.stage)
        super().__init__(render_mode=render_mode, config=cfg)
        
        self.base_target_height = self.height_targets[self.stage]
        print(f"  curriculum initialized at stage {self.stage}")
        print(f"  Target: {self.base_target_height:.2f}m ± {self.height_tolerances[self.stage]:.2f}m")
        print(f"  Min episode length: {self.min_episode_lengths[self.stage]} steps")

    def _apply_stage_settings(self, cfg: Dict[str, Any], stage: int) -> None:
        """Configure curriculum stage with progressive difficulty."""
        if stage <= 1:
            # Early stages: no randomization, shorter episodes
            cfg['domain_rand'] = False
            cfg['max_episode_steps'] = 2000
        elif stage <= 3:
            # Mid stages: light randomization
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.98, 1.02]
            cfg['rand_friction_range'] = [0.98, 1.02]
            cfg['max_episode_steps'] = 3000
        else:
            # Final stage: full difficulty + longer episodes
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
            height = info.get('height', 0.0)
            current_target = self.height_targets[min(self.stage, len(self.height_targets) - 1)]
            current_tolerance = self.height_tolerances[min(self.stage, len(self.height_tolerances) - 1)]
            min_length = self.min_episode_lengths[min(self.stage, len(self.min_episode_lengths) - 1)]
            
            height_error = abs(height - current_target)
            
            # SUCCESS CRITERIA (all must be met):
            # 1. Height within tight tolerance
            height_ok = height_error < current_tolerance
            # 2. Episode lasted long enough (not just luck)
            long_enough = self.current_step >= min_length
            # 3. Not terminated due to falling (truncated = timeout is OK)
            not_fallen = not terminated
            
            success = bool(height_ok and long_enough and not_fallen)

            self.success_buffer.append(1 if success else 0)
            if len(self.success_buffer) > self.advance_after:
                self.success_buffer = self.success_buffer[-self.advance_after:]

            # ADVANCE: Only if sustained high success rate
            if (len(self.success_buffer) == self.advance_after and 
                np.mean(self.success_buffer) >= self.stage_success_threshold):
                
                if self.stage < self.max_stage:
                    old_stage = self.stage
                    self.stage += 1
                    self.base_target_height = self.height_targets[self.stage]
                    
                    cfg = self.cfg.copy()
                    self._apply_stage_settings(cfg, self.stage)
                    
                    # Update environment settings
                    self.domain_rand = cfg.get('domain_rand', self.domain_rand)
                    self.rand_mass_range = cfg.get('rand_mass_range', self.rand_mass_range)
                    self.rand_friction_range = cfg.get('rand_friction_range', self.rand_friction_range)
                    self.max_episode_steps = cfg.get('max_episode_steps', self.max_episode_steps)
                    self.cfg = cfg
                    
                    # Reset success buffer for new stage
                    self.success_buffer = []
                    
                    print(f"\n{'='*60}")
                    print(f" CURRICULUM ADVANCED: Stage {old_stage} → {self.stage}")
                    print(f"  New target: {self.base_target_height:.2f}m ± {self.height_tolerances[self.stage]:.2f}m")
                    print(f"  Min episode length: {self.min_episode_lengths[self.stage]} steps")
                    print(f"  Domain randomization: {self.domain_rand}")
                    print(f"{'='*60}\n")
                    
                    info['curriculum_stage_advanced'] = self.stage
                elif self.stage == self.max_stage:
                    # Already at final stage, log mastery
                    print(f" Stage {self.stage} MASTERED (success rate: {np.mean(self.success_buffer):.1%})")

            # ENHANCED INFO: Always log curriculum progress
            info['curriculum_stage'] = self.stage
            info['curriculum_target_height'] = self.base_target_height
            info['curriculum_success_rate'] = float(np.mean(self.success_buffer)) if self.success_buffer else 0.0
            info['curriculum_height_tolerance'] = self.height_tolerances[self.stage]
            info['curriculum_min_length'] = self.min_episode_lengths[self.stage]
            info['curriculum_episode_success'] = success

        return obs, reward, terminated, truncated, info


def make_standing_curriculum_env(render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    return StandingCurriculumEnv(render_mode=render_mode, config=config)