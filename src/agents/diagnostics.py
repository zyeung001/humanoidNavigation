"""
Diagnostics callback and helpers for humanoid standing/walking training.
Logs detailed metrics and produces periodic debug artifacts.

ENHANCED: Now tracks height distribution bins, action magnitudes, and velocity tracking for walking.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DiagnosticsCallback(BaseCallback):
    """
    Unified diagnostics callback for both standing and walking tasks.
    Tracks height, actions, rewards, and walking-specific velocity metrics.
    """
    def __init__(self, log_freq: int = 10000, verbose: int = 0, task: str = "standing"):
        super().__init__(verbose)
        self.log_freq = int(log_freq)
        self.task = task

        # Rolling buffers
        self._heights: List[float] = []
        self._actions_abs: List[float] = []
        self._rewards: List[float] = []
        
        # Walking-specific buffers
        self._velocity_errors: List[float] = []
        self._commanded_speeds: List[float] = []
        self._actual_speeds: List[float] = []
        self._xy_positions: List[tuple] = []

    def _on_step(self) -> bool:
        t = self.num_timesteps
        infos = self.locals.get('infos', [])
        actions = self.locals.get('actions', None)
        rewards = self.locals.get('rewards', None)

        for info in infos:
            if isinstance(info, dict):
                # Height (common to both tasks)
                if 'height' in info:
                    self._heights.append(float(info['height']))
                
                # Walking-specific metrics
                if 'velocity_error' in info:
                    self._velocity_errors.append(float(info['velocity_error']))
                if 'commanded_speed' in info:
                    self._commanded_speeds.append(float(info['commanded_speed']))
                if 'actual_speed' in info:
                    self._actual_speeds.append(float(info['actual_speed']))
                if 'x_position' in info and 'y_position' in info:
                    self._xy_positions.append((float(info['x_position']), float(info['y_position'])))
                    
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

            # Height metrics
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
                
                # Height distribution bins
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
                
            # Action metrics
            if self._actions_abs:
                arr = np.asarray(self._actions_abs[-5000:])
                metrics.update({
                    'diag/action_mag_mean': float(arr.mean()),
                    'diag/action_mag_std': float(arr.std()),
                    'diag/action_mag_p10': float(np.percentile(arr, 10)),
                    'diag/action_mag_p90': float(np.percentile(arr, 90)),
                    'diag/action_mag_max': float(arr.max()),
                })
                
            # Reward metrics
            if self._rewards:
                arr = np.asarray(self._rewards[-5000:])
                metrics.update({
                    'diag/reward_mean_recent': float(arr.mean()),
                    'diag/reward_std_recent': float(arr.std()),
                })
            
            # ========== WALKING-SPECIFIC METRICS ==========
            if self._velocity_errors:
                arr = np.asarray(self._velocity_errors[-5000:])
                metrics.update({
                    'diag/velocity_error_mean': float(arr.mean()),
                    'diag/velocity_error_std': float(arr.std()),
                    'diag/velocity_error_p10': float(np.percentile(arr, 10)),
                    'diag/velocity_error_p50': float(np.percentile(arr, 50)),
                    'diag/velocity_error_p90': float(np.percentile(arr, 90)),
                    'diag/velocity_error_max': float(arr.max()),
                })
                
                # Velocity error distribution bins
                vel_err_below_0_1 = np.sum(arr < 0.1) / len(arr) * 100
                vel_err_0_1_to_0_3 = np.sum((arr >= 0.1) & (arr < 0.3)) / len(arr) * 100
                vel_err_0_3_to_0_5 = np.sum((arr >= 0.3) & (arr < 0.5)) / len(arr) * 100
                vel_err_0_5_to_1_0 = np.sum((arr >= 0.5) & (arr < 1.0)) / len(arr) * 100
                vel_err_above_1_0 = np.sum(arr >= 1.0) / len(arr) * 100
                
                metrics.update({
                    'diag/vel_err_pct_below_0.1': float(vel_err_below_0_1),
                    'diag/vel_err_pct_0.1_to_0.3': float(vel_err_0_1_to_0_3),
                    'diag/vel_err_pct_0.3_to_0.5': float(vel_err_0_3_to_0_5),
                    'diag/vel_err_pct_0.5_to_1.0': float(vel_err_0_5_to_1_0),
                    'diag/vel_err_pct_above_1.0': float(vel_err_above_1_0),
                })
            
            if self._commanded_speeds:
                arr = np.asarray(self._commanded_speeds[-5000:])
                metrics.update({
                    'diag/commanded_speed_mean': float(arr.mean()),
                    'diag/commanded_speed_max': float(arr.max()),
                })
            
            if self._actual_speeds:
                arr = np.asarray(self._actual_speeds[-5000:])
                metrics.update({
                    'diag/actual_speed_mean': float(arr.mean()),
                    'diag/actual_speed_std': float(arr.std()),
                    'diag/actual_speed_max': float(arr.max()),
                })
                
                # Speed tracking ratio (actual / commanded)
                if self._commanded_speeds:
                    cmd_arr = np.asarray(self._commanded_speeds[-5000:])
                    # Avoid division by zero for standing commands
                    valid_mask = cmd_arr > 0.1
                    if np.any(valid_mask):
                        speed_ratio = arr[valid_mask] / cmd_arr[valid_mask]
                        metrics.update({
                            'diag/speed_tracking_ratio': float(np.mean(speed_ratio)),
                        })
            
            # XY drift metric
            if self._xy_positions:
                xy = np.array(self._xy_positions[-1000:])
                distances_from_origin = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
                metrics.update({
                    'diag/xy_drift_mean': float(np.mean(distances_from_origin)),
                    'diag/xy_drift_max': float(np.max(distances_from_origin)),
                })

            # Try wandb if available
            try:
                import wandb
                if wandb.run:
                    wandb.log(metrics, step=t)
            except Exception:
                pass

            if self.verbose:
                # Print diagnostics summary
                print(f"[Diagnostics] step={t:,} | "
                      f"h={metrics.get('diag/height_mean','nan'):.3f}±{metrics.get('diag/height_std','nan'):.3f} | "
                      f"act={metrics.get('diag/action_mag_mean','nan'):.3f} | "
                      f"rew={metrics.get('diag/reward_mean_recent','nan'):.1f}")
                
                # Print height distribution
                if self._heights:
                    print(f"  Height dist: "
                          f"<1.0={metrics.get('diag/height_pct_below_1.0',0):.1f}% | "
                          f"1.0-1.2={metrics.get('diag/height_pct_1.0_to_1.2',0):.1f}% | "
                          f"1.2-1.35={metrics.get('diag/height_pct_1.2_to_1.35',0):.1f}% | "
                          f"1.35-1.45={metrics.get('diag/height_pct_1.35_to_1.45',0):.1f}% | "
                          f">1.45={metrics.get('diag/height_pct_above_1.45',0):.1f}%")
                
                # Print walking-specific metrics
                if self._velocity_errors:
                    print(f"  Vel tracking: "
                          f"err={metrics.get('diag/velocity_error_mean', 0):.4f}±{metrics.get('diag/velocity_error_std', 0):.4f} m/s | "
                          f"p90={metrics.get('diag/velocity_error_p90', 0):.4f} m/s")
                    print(f"  Vel err dist: "
                          f"<0.1={metrics.get('diag/vel_err_pct_below_0.1', 0):.1f}% | "
                          f"0.1-0.3={metrics.get('diag/vel_err_pct_0.1_to_0.3', 0):.1f}% | "
                          f"0.3-0.5={metrics.get('diag/vel_err_pct_0.3_to_0.5', 0):.1f}% | "
                          f"0.5-1.0={metrics.get('diag/vel_err_pct_0.5_to_1.0', 0):.1f}% | "
                          f">1.0={metrics.get('diag/vel_err_pct_above_1.0', 0):.1f}%")
                    
                    if self._commanded_speeds and self._actual_speeds:
                        print(f"  Speed: cmd={metrics.get('diag/commanded_speed_mean', 0):.2f} m/s | "
                              f"actual={metrics.get('diag/actual_speed_mean', 0):.2f} m/s | "
                              f"ratio={metrics.get('diag/speed_tracking_ratio', 0):.2f}")

        return True


class WalkingDiagnosticsCallback(DiagnosticsCallback):
    """
    Specialized diagnostics callback for walking task.
    Inherits from DiagnosticsCallback with walking-specific defaults.
    """
    def __init__(self, log_freq: int = 10000, verbose: int = 0):
        super().__init__(log_freq=log_freq, verbose=verbose, task="walking")
