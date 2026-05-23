# src/agents/ — High-Level Agent Wrappers + Diagnostics

## `standing_agent.py`

### StandingAgent

High-level PPO agent for standing task. Wraps environment creation, model construction, training, evaluation, and checkpoint management in one object. Configured entirely from the `standing` section of `config/training_config.yaml`.

Key methods:

| Method | Purpose |
|--------|---------|
| `create_environment()` | Builds `SubprocVecEnv` (n_envs > 1) or `DummyVecEnv`, wraps in `VecNormalize` |
| `create_model()` | Creates PPO with architecture from config; handles string→`torch.nn` activation mapping |
| `train(total_timesteps)` | Runs `.learn()` with `StandingCallback` + `EarlyStoppingCallback` |
| `evaluate(n_episodes)` | Deterministic rollouts; reports height error, stability, reward |
| `analyze_standing_performance(n_episodes)` | Debug mode — prints per-episode breakdown and early-termination rate |
| `load_model(path)` | Loads model + restores matching `VecNormalize` stats |

### StandingCallback(BaseCallback)

Unified SB3 callback for standing training. Handles:
- **WandB logging** every `log_freq` steps: height distribution, XY drift, action magnitudes, policy `log_std`
- **Evaluation** every `eval_freq` steps via `eval_env_fn()`; saves best model by mean reward
- **Video recording** every `video_freq` steps (WandB only)
- **Checkpointing** every `save_freq` steps; co-saves `VecNormalize` `.pkl` alongside model `.zip`

VecNormalize is saved before every evaluation run to keep stats in sync with the model.

### EarlyStoppingCallback(BaseCallback)

Stops training when standing is mastered. Checks every `check_freq` steps across 5 evaluation episodes. Success requires all of: `mean_reward > target`, `height_error < target`, `height_stability < target`, `mean_length > min_episode_length`. Requires `patience` consecutive checks before stopping.

### `create_standing_agent(config_path, device, use_wandb)`

Convenience function for quick setup. Loads `config/training_config.yaml`, extracts the `standing` section, auto-detects CUDA, and returns a `StandingAgent`.

---

## `diagnostics.py`

### DiagnosticsCallback(BaseCallback)

Unified diagnostics for both tasks. Logs to WandB every `log_freq` steps using rolling buffers (last 5000 steps).

**Standing metrics logged (`diag/` prefix):**
- Height: mean, std, p10, p90, min, max
- Height distribution bins: `<1.0`, `1.0–1.2`, `1.2–1.35`, `1.35–1.45`, `>1.45`
- Action magnitude: mean, std, p10, p90, max
- Reward: mean, std

**Walking metrics added (`diag/` prefix):**
- Velocity error: mean, std, p10, p50, p90, max
- Velocity error distribution bins: `<0.1`, `0.1–0.3`, `0.3–0.5`, `0.5–1.0`, `>1.0`
- Commanded speed mean/max, actual speed mean/std/max
- Speed tracking ratio (`actual/commanded`, valid commands only)
- XY drift: mean and max distance from origin

Instantiate with `task="walking"` to enable walking-specific metric collection.

### WalkingDiagnosticsCallback(DiagnosticsCallback)

Thin subclass of `DiagnosticsCallback` with `task="walking"` default. Preferred entry point for walking training scripts.
