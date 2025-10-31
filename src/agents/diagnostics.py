"""
Diagnostics callback and helpers for humanoid standing training.
Logs detailed metrics and produces periodic debug artifacts.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DiagnosticsCallback(BaseCallback):
    def __init__(self, log_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = int(log_freq)

        # rolling buffers
        self._heights: List[float] = []
        self._actions_abs: List[float] = []
        self._rewards: List[float] = []

    def _on_step(self) -> bool:
        t = self.num_timesteps
        infos = self.locals.get('infos', [])
        actions = self.locals.get('actions', None)
        rewards = self.locals.get('rewards', None)

        for info in infos:
            if isinstance(info, dict) and 'height' in info:
                self._heights.append(float(info['height']))
        if actions is not None:
            try:
                self._actions_abs.append(float(np.abs(actions).mean()))
            except Exception:
                pass
        if rewards is not None:
            try:
                self._rewards.append(float(np.mean(rewards)))
            except Exception:
                pass

        if self.log_freq > 0 and (t % self.log_freq == 0):
            metrics: Dict[str, Any] = {'global_step': t}

            if self._heights:
                arr = np.asarray(self._heights[-5000:])
                metrics.update({
                    'diag/height_mean': float(arr.mean()),
                    'diag/height_std': float(arr.std()),
                    'diag/height_p10': float(np.percentile(arr, 10)),
                    'diag/height_p90': float(np.percentile(arr, 90)),
                })
            if self._actions_abs:
                arr = np.asarray(self._actions_abs[-5000:])
                metrics.update({
                    'diag/action_mag_mean': float(arr.mean()),
                    'diag/action_mag_p90': float(np.percentile(arr, 90)),
                })
            if self._rewards:
                arr = np.asarray(self._rewards[-5000:])
                metrics.update({
                    'diag/reward_mean_recent': float(arr.mean()),
                })

            # Try wandb if available
            try:
                import wandb
                if wandb.run:
                    wandb.log(metrics, step=t)
            except Exception:
                pass

            if self.verbose:
                print(f"[Diagnostics] step={t:,} | "
                      f"h={metrics.get('diag/height_mean','nan'):.3f}±{metrics.get('diag/height_std','nan'):.3f} | "
                      f"act={metrics.get('diag/action_mag_mean','nan'):.3f} | "
                      f"rew={metrics.get('diag/reward_mean_recent','nan'):.1f}")

        return True


