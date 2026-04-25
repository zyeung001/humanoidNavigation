"""
JSONL metrics logger for training runs.

Writes one JSON record per log interval combining:
- PPO internals from SB3 logger (approx_kl, clip_fraction, explained_variance,
  policy_gradient_loss, entropy_loss, std, learning_rate, ...)
- Rollout metrics (ep_rew_mean, ep_len_mean, fps, ...)
- Env metrics from infos (velocity_error, jerk_penalty, height, action_magnitude,
  reward components, curriculum stage, termination causes, behavior ratios)

The output file is a JSONL (one JSON object per line) at a configurable path,
defaulting to <model_dir>/metrics/training_metrics.jsonl. Use jq, pandas, or
plain Python to query offline without depending on WandB.
"""

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _to_jsonable(value):
    """Coerce numpy scalars/arrays to plain Python for json.dump."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class JsonlOutputFormat:
    """
    SB3 logger output format that captures train/* and rollout/* into a buffer.

    Mirrors the pattern used by WandBOutputFormat: attaches to
    Logger.output_formats so SB3's logger.dump() forwards everything.
    The captured values are flushed to disk by JsonlMetricsCallback alongside
    env metrics, so each line is one self-contained snapshot.
    """

    def __init__(self):
        self.buffer: Dict[str, float] = {}

    def write(self, key_values, key_excluded, step):
        for key, value in key_values.items():
            if key in key_excluded and "json" in key_excluded[key]:
                continue
            if not isinstance(value, (int, float, np.floating, np.integer)):
                continue
            if key.startswith("train/"):
                self.buffer[f"ppo/{key[6:]}"] = _to_jsonable(value)
            elif key.startswith("rollout/") or key.startswith("time/"):
                self.buffer[key] = _to_jsonable(value)

    def drain(self) -> Dict[str, float]:
        out = self.buffer
        self.buffer = {}
        return out

    def close(self):
        pass


class JsonlMetricsCallback(BaseCallback):
    """
    Collects training metrics and writes one JSONL record per `log_freq` steps.

    Captures the same env-info keys as the WandB callbacks (velocity errors,
    jerk, height, action magnitude, episode stats, reward/* components,
    curriculum/stage, behavior/*, termination causes), plus PPO internals
    via JsonlOutputFormat attached to SB3's logger.

    Args:
        output_path: JSONL file path. Parents are created if missing.
        log_freq: Step interval between flushes.
        buffer_size: Rolling window size for env metric aggregation.
        flush_each_write: fsync after each line so partial runs are recoverable.
    """

    def __init__(
        self,
        output_path: str,
        log_freq: int = 5000,
        buffer_size: int = 1000,
        flush_each_write: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.log_freq = int(log_freq)
        self.buffer_size = int(buffer_size)
        self.flush_each_write = flush_each_write

        # Env metric buffers (rolling)
        self.velocity_errors: deque = deque(maxlen=buffer_size)
        self.velocity_errors_x: deque = deque(maxlen=buffer_size)
        self.velocity_errors_y: deque = deque(maxlen=buffer_size)
        self.jerk_penalties: deque = deque(maxlen=buffer_size)
        self.action_magnitudes: deque = deque(maxlen=buffer_size)
        self.heights: deque = deque(maxlen=buffer_size)

        # Episode tracking (smaller window — episodes are sparse)
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)

        # Reward components: dict of name -> deque
        self.reward_components: Dict[str, deque] = {}

        # Curriculum + behavior + termination
        self.current_stage = 0
        self.curriculum_stages: deque = deque(maxlen=100)
        self.termination_causes: deque = deque(maxlen=buffer_size)
        self.is_standing_buffer: deque = deque(maxlen=buffer_size)
        self.standing_penalty_buffer: deque = deque(maxlen=buffer_size)
        self.speed_ratio_buffer: deque = deque(maxlen=buffer_size)

        self._sb3_format: Optional[JsonlOutputFormat] = None
        self._fh = None
        self._start_wall = time.time()

    def _on_training_start(self) -> None:
        # Append mode so resumed runs extend the same file.
        self._fh = open(self.output_path, "a", buffering=1, encoding="utf-8")

        # Attach SB3 logger output format for PPO / rollout metrics.
        if hasattr(self.model.logger, "output_formats"):
            self._sb3_format = JsonlOutputFormat()
            self.model.logger.output_formats.append(self._sb3_format)

        header = {
            "event": "run_start",
            "wall_time": datetime.now().isoformat(),
            "timesteps": int(self.num_timesteps),
            "output_path": str(self.output_path),
        }
        self._write(header)
        if self.verbose:
            print(f"[JsonlMetrics] Logging to {self.output_path}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        for info in infos:
            self._collect_env_info(info)

        if self.log_freq > 0 and self.num_timesteps % self.log_freq == 0:
            self._flush_record()

        return True

    def _on_training_end(self) -> None:
        self._flush_record()  # final snapshot
        footer = {
            "event": "run_end",
            "wall_time": datetime.now().isoformat(),
            "timesteps": int(self.num_timesteps),
            "wall_seconds": time.time() - self._start_wall,
        }
        self._write(footer)
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    # ------------------------------------------------------------------ helpers

    def _collect_env_info(self, info: dict) -> None:
        if "velocity_error" in info:
            self.velocity_errors.append(info["velocity_error"])
        if "velocity_error_x" in info:
            self.velocity_errors_x.append(info["velocity_error_x"])
        elif "commanded_vx" in info and "x_velocity" in info:
            self.velocity_errors_x.append(abs(info["commanded_vx"] - info["x_velocity"]))
        if "velocity_error_y" in info:
            self.velocity_errors_y.append(info["velocity_error_y"])
        elif "commanded_vy" in info and "y_velocity" in info:
            self.velocity_errors_y.append(abs(info["commanded_vy"] - info["y_velocity"]))

        if "jerk_penalty" in info:
            self.jerk_penalties.append(info["jerk_penalty"])
        if "action_magnitude" in info:
            self.action_magnitudes.append(info["action_magnitude"])
        if "height" in info:
            self.heights.append(info["height"])

        for key, value in info.items():
            if key.startswith("reward/") and isinstance(value, (int, float, np.floating, np.integer)):
                name = key[len("reward/"):]
                buf = self.reward_components.get(name)
                if buf is None:
                    buf = deque(maxlen=self.buffer_size)
                    self.reward_components[name] = buf
                buf.append(float(value))

        if "termination_cause" in info:
            self.termination_causes.append(info["termination_cause"])
        if "behavior/is_standing" in info:
            self.is_standing_buffer.append(1 if info["behavior/is_standing"] else 0)
        if "behavior/standing_penalty_applied" in info:
            self.standing_penalty_buffer.append(1 if info["behavior/standing_penalty_applied"] else 0)
        if "behavior/speed_ratio" in info:
            self.speed_ratio_buffer.append(info["behavior/speed_ratio"])

        if "episode" in info:
            ep = info["episode"]
            self.episode_rewards.append(float(ep["r"]))
            self.episode_lengths.append(float(ep["l"]))
            if "curriculum_stage" in info:
                stage = int(info["curriculum_stage"])
                self.curriculum_stages.append(stage)
                self.current_stage = stage

    def _flush_record(self) -> None:
        record: Dict = {
            "event": "metrics",
            "timesteps": int(self.num_timesteps),
            "wall_time": datetime.now().isoformat(),
            "wall_seconds": time.time() - self._start_wall,
        }

        if self.velocity_errors:
            record["env/velocity_error_mean"] = float(np.mean(self.velocity_errors))
            record["env/velocity_error_std"] = float(np.std(self.velocity_errors))
        if self.velocity_errors_x:
            record["env/velocity_error_x_mean"] = float(np.mean(self.velocity_errors_x))
        if self.velocity_errors_y:
            record["env/velocity_error_y_mean"] = float(np.mean(self.velocity_errors_y))
        if self.jerk_penalties:
            record["env/jerk_penalty_mean"] = float(np.mean(self.jerk_penalties))
            record["env/jerk_penalty_max"] = float(np.max(self.jerk_penalties))
        if self.action_magnitudes:
            record["env/action_magnitude_mean"] = float(np.mean(self.action_magnitudes))
        if self.heights:
            record["env/height_mean"] = float(np.mean(self.heights))
            record["env/height_std"] = float(np.std(self.heights))

        if self.episode_rewards:
            record["episode/reward_mean"] = float(np.mean(self.episode_rewards))
            record["episode/reward_std"] = float(np.std(self.episode_rewards))
        if self.episode_lengths:
            record["episode/length_mean"] = float(np.mean(self.episode_lengths))
            record["episode/length_std"] = float(np.std(self.episode_lengths))

        for name, buf in self.reward_components.items():
            if buf:
                record[f"reward/{name}_mean"] = float(np.mean(buf))
                record[f"reward/{name}_std"] = float(np.std(buf))

        if self.curriculum_stages:
            record["curriculum/stage"] = int(self.current_stage)
            record["curriculum/avg_stage"] = float(np.mean(self.curriculum_stages))

        if self.termination_causes:
            causes = list(self.termination_causes)
            total = len(causes)
            counts: Dict[str, int] = {}
            for cause in causes:
                counts[str(cause)] = counts.get(str(cause), 0) + 1
            for cause, count in counts.items():
                record[f"termination/{cause}_ratio"] = count / total

        if self.is_standing_buffer:
            standing_ratio = float(np.mean(self.is_standing_buffer))
            record["behavior/standing_ratio"] = standing_ratio
            record["behavior/standing_exploit_ratio"] = standing_ratio
        if self.standing_penalty_buffer:
            record["behavior/standing_penalty_ratio"] = float(np.mean(self.standing_penalty_buffer))
        if self.speed_ratio_buffer:
            record["behavior/speed_ratio_mean"] = float(np.mean(self.speed_ratio_buffer))
            record["behavior/command_effectiveness"] = float(
                np.clip(np.mean(self.speed_ratio_buffer), 0, 1)
            )

        if self._sb3_format is not None:
            record.update(self._sb3_format.drain())

        self._write(record)

    def _write(self, record: Dict) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(record, default=_to_jsonable) + "\n")
        if self.flush_each_write:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except (OSError, ValueError):
                pass


def default_metrics_path(model_dir: str, run_name: Optional[str] = None) -> str:
    """Build a default JSONL path under <model_dir>/metrics/."""
    base = Path(model_dir) / "metrics"
    base.mkdir(parents=True, exist_ok=True)
    if run_name:
        return str(base / f"{run_name}.jsonl")
    return str(base / "training_metrics.jsonl")
