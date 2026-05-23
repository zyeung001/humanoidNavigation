# src/utils/ — Visualization + WandB Utilities

## `__init__.py` — Platform Helpers

Two functions exported at package level:

| Function | Purpose |
|----------|---------|
| `configure_mujoco_gl()` | Sets `MUJOCO_GL=egl` on Linux (headless GPU). No-op on macOS/Windows. Call before importing MuJoCo envs. |
| `get_subprocess_start_method()` | Returns `"spawn"` (Windows), `"fork"` (macOS), or `"forkserver"` (Linux) for `SubprocVecEnv`. |

---

## `visualization.py`

### `setup_display()`

Calls `configure_mujoco_gl()` and prints the resulting `MUJOCO_GL` value. Use at the top of scripts that render frames.

### `test_environment(env) → bool`

Smoke-tests a gymnasium env: resets, renders one frame, steps once, renders again. Checks that the frame is a 3-channel `np.ndarray`. Returns `True` on success. Useful for verifying render modes before recording video.

---

## `wandb_callbacks.py`

### VelocityTrackingWandBCallback(BaseCallback)

Comprehensive WandB callback for walking training. Uses rolling `deque` buffers (default 1000 steps).

**Logs at `log_freq` steps:**
- `train/velocity_error` — episode-end errors only (not step-wise, avoids fall spikes)
- `train/velocity_error_x/y` — per-axis errors
- `train/jerk_penalty`, `train/action_magnitude`
- `train/height_mean/std`
- `episode/reward_mean/std`, `episode/length_mean/std`
- `curriculum/stage`, `curriculum/avg_stage`

Logs curriculum `stage_history` table on training end.

### CurriculumWandBCallback(BaseCallback)

Focused on curriculum progression. Tracks episode-end velocity errors per stage (not step-wise). Logs `curriculum/stage_advanced` and `curriculum/advancement_timestep` on every stage transition. Per-stage metrics use last 20 episodes for stable estimates; tolerance shrinks with stage (`0.7 - 0.05 × stage`).

### VideoRecordingCallback(BaseCallback)

Records evaluation rollouts and uploads to WandB as mp4. Runs every `video_freq` steps. Resets the eval env on episode termination to avoid short clips.

### `init_wandb_run(project, name, config, tags) → bool`

Helper to initialize a WandB run with `reinit=True`. Returns `False` gracefully if WandB is not installed.

### `finish_wandb_run()`

Calls `wandb.finish()`. Use at the end of training scripts to flush all pending logs.

---

## `plot_velocity_commands.py`

### `simulate_and_plot(duration, dt, seed, save_path)`

Simulates `VelocityCommandGenerator` for `duration` seconds at `dt` timestep and produces a matplotlib figure showing step-like command changes (`vx`, `vy`, `yaw_rate`) over time. Demonstrates the uniform command sampling pattern and zero-velocity braking periods. Saves to `save_path` if provided.
