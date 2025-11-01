"""
Diagnostics callback and helpers for humanoid standing training.
Logs detailed metrics and produces periodic debug artifacts.

ENHANCED: Now tracks height distribution bins and action magnitudes for debugging.
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
                    'diag/height_min': float(arr.min()),
                    'diag/height_max': float(arr.max()),
                })
                
                # NEW: Height distribution bins to diagnose preference issue
                heights_below_1_0 = np.sum(arr < 1.0) / len(arr) * 100
                heights_1_0_to_1_2 = np.sum((arr >= 1.0) & (arr < 1.2)) / len(arr) * 100
                heights_1_2_to_1_35 = np.sum((arr >= 1.2) & (arr < 1.35)) / len(arr) * 100
                heights_1_35_to_1_45 = np.sum((arr >= 1.35) & (arr < 1.45)) / len(arr) * 100
                heights_above_1_45 = np.sum(arr >= 1.45) / len(arr) * 100
                
                metrics.update({
                    'diag/height_pct_below_1.0': float(heights_below_1_0),
                    'diag/height_pct_1.0_to_1.2': float(heights_1_0_to_1_2),
                    'diag/height_pct_1.2_to_1.35': float(heights_1_2_to_1_35),
                    'diag/height_pct_1.35_to_1.45': float(heights_1_35_to_1_45),
                    'diag/height_pct_above_1.45': float(heights_above_1_45),
                })
                
            if self._actions_abs:
                arr = np.asarray(self._actions_abs[-5000:])
                metrics.update({
                    'diag/action_mag_mean': float(arr.mean()),
                    'diag/action_mag_std': float(arr.std()),
                    'diag/action_mag_p10': float(np.percentile(arr, 10)),
                    'diag/action_mag_p90': float(np.percentile(arr, 90)),
                    'diag/action_mag_max': float(arr.max()),
                })
                
            if self._rewards:
                arr = np.asarray(self._rewards[-5000:])
                metrics.update({
                    'diag/reward_mean_recent': float(arr.mean()),
                    'diag/reward_std_recent': float(arr.std()),
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
                
                # Print height distribution
                if self._heights:
                    print(f"  Height distribution: "
                          f"<1.0={metrics.get('diag/height_pct_below_1.0',0):.1f}% | "
                          f"1.0-1.2={metrics.get('diag/height_pct_1.0_to_1.2',0):.1f}% | "
                          f"1.2-1.35={metrics.get('diag/height_pct_1.2_to_1.35',0):.1f}% | "
                          f"1.35-1.45={metrics.get('diag/height_pct_1.35_to_1.45',0):.1f}% | "
                          f">1.45={metrics.get('diag/height_pct_above_1.45',0):.1f}%")

        return True


