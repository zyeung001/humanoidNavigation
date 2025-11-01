"""
Curriculum wrapper for StandingEnv.

Stages progress from easier to harder conditions using success metrics.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np

from .standing_env import StandingEnv


class StandingCurriculumEnv(StandingEnv):
    """StandingEnv with curriculum progression.

    Stages (example):
    0: No domain rand, shorter episodes, generous thresholds
    1: Enable small domain rand
    2: Stronger rand, longer episodes
    3: Add action symmetry and obs history
    """

    def __init__(self, render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).copy()
        self.stage = int(cfg.get('curriculum_start_stage', 0))
        self.max_stage = int(cfg.get('curriculum_max_stage', 3))
        self.advance_after = int(cfg.get('curriculum_advance_after', 10))  # episodes
        self.success_buffer = []
        self.stage_success_threshold = float(cfg.get('curriculum_success_rate', 0.8))

        # Apply initial stage settings onto config
        self._apply_stage_settings(cfg, self.stage)

        super().__init__(render_mode=render_mode, config=cfg)

    def _apply_stage_settings(self, cfg: Dict[str, Any], stage: int) -> None:
        if stage <= 0:
            cfg['domain_rand'] = False
            cfg['max_episode_steps'] = min(int(cfg.get('max_episode_steps', 5000)), 2000)
            cfg['action_smoothing'] = False
            cfg['obs_history'] = 0
        elif stage == 1:
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.98, 1.02]
            cfg['rand_friction_range'] = [0.98, 1.02]
            cfg['max_episode_steps'] = int(cfg.get('max_episode_steps', 3000))
        elif stage == 2:
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.95, 1.05]
            cfg['rand_friction_range'] = [0.95, 1.05]
            cfg['action_smoothing'] = True
            cfg['action_smoothing_tau'] = 0.2
            cfg['max_episode_steps'] = int(cfg.get('max_episode_steps', 4000))
        else:
            cfg['domain_rand'] = True
            cfg['rand_mass_range'] = [0.9, 1.1]
            cfg['rand_friction_range'] = [0.9, 1.1]
            cfg['action_smoothing'] = True
            cfg['action_symmetry'] = True
            # Do NOT change observation-related options mid-training; keep initial obs dimension fixed
            cfg['max_episode_steps'] = int(cfg.get('max_episode_steps', 5000))

    def reset(self, seed: Optional[int] = None):
        return super().reset(seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        done = bool(terminated or truncated)
        if done:
            # success criterion: standing height close to target and decent duration
            height = info.get('height', 0.0)
            height_ok = abs(height - self.base_target_height) < 0.1
            long_enough = self.current_step >= min(300, self.max_episode_steps // 2)
            success = bool(height_ok and long_enough and not terminated)

            self.success_buffer.append(1 if success else 0)
            if len(self.success_buffer) > self.advance_after:
                self.success_buffer = self.success_buffer[-self.advance_after:]

            # advance if success rate met
            if len(self.success_buffer) == self.advance_after and np.mean(self.success_buffer) >= self.stage_success_threshold:
                if self.stage < self.max_stage:
                    self.stage += 1
                    cfg = self.cfg.copy()
                    self._apply_stage_settings(cfg, self.stage)
                    # Update local settings live (simple subset)
                    self.domain_rand = cfg.get('domain_rand', self.domain_rand)
                    self.rand_mass_range = cfg.get('rand_mass_range', self.rand_mass_range)
                    self.rand_friction_range = cfg.get('rand_friction_range', self.rand_friction_range)
                    self.max_episode_steps = cfg.get('max_episode_steps', self.max_episode_steps)
                    self.enable_action_smoothing = cfg.get('action_smoothing', self.enable_action_smoothing)
                    self.action_smoothing_tau = cfg.get('action_smoothing_tau', self.action_smoothing_tau)
                    self.enable_action_symmetry = cfg.get('action_symmetry', self.enable_action_symmetry)
                    # Do not modify observation processing flags after initialization
                    self.cfg = cfg
                    info['curriculum_stage_advanced'] = self.stage

            info['curriculum_stage'] = self.stage

        return obs, reward, terminated, truncated, info


def make_standing_curriculum_env(render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    return StandingCurriculumEnv(render_mode=render_mode, config=config)


