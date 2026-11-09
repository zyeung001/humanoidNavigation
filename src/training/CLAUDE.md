# src/training/ — Transfer Learning, Model Management, Callbacks

## Standing training pipeline (used by `scripts/train_standing.py`)

| File | What |
|------|------|
| `experiment.py` | `load_experiment(path)`: a plain config, or an experiment file = `experiment: {base, name, hypothesis, resume}` + `overrides:`. Chains (an experiment's base may be an experiment). `changes_flat()` is what the run changed. `RunDir`: `runs/<name>/{config.yaml, checkpoints/, best/, final/, metrics.jsonl, evals.jsonl}` |
| `run_logger.py` | `MetricsSink` (one dict -> run JSONL + W&B; `log_video()` uploads a video FILE, no moviepy), `start_wandb()`, `TrainingLogger` callback (`train/` `reward/` `ppo/` `health/` `perf/`) and the console line. Module docstring = how to read the dashboard |
| `robustness_eval.py` | `RobustnessEvalCallback`: every N steps (and at start) checkpoint -> score on `src/robustness` SUITE -> `eval/*`, copy best to `runs/<name>/best/`, optional adaptive training mix |

- `eval/score` is the only number comparable across runs with different rewards/obs/nets.
- Experiments never read or write the base config's `final_model_path`/`vecnormalize_path`
  (they go to the run dir), and never implicitly load its vecnorm.
- A run's `config.yaml` is the effective config (`experiment.resolved: true`); loading it does
  not re-merge and does not auto-resume.

## `transfer_utils.py` — Standing → Walking Transfer

The core transfer pipeline. Called by `scripts/train_walking.py` when `--from-standing` is passed.

### Pipeline: `transfer_standing_to_walking()`

```
1. VecNormalizeExtender.extend()     — Grow obs stats from 1484 → 1493 dims
2. WarmupCollector.collect()         — Run 10k random steps to populate stats
3. PolicyTransfer.transfer()         — Copy weights + initialize command dims
4. Return (walking_model, walking_vecnorm)
```

### VecNormalizeExtender

Extends VecNormalize observation statistics to accommodate the 9-dim command block.

**Dimension structure:**
```
Standing: (365 + 6) × 4 frames = 1484 dims (body portion)
Walking:  1484 body + 9 command block = 1493 dims

Command block: [vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual,
                err_vx, err_vy, err_speed, err_angle]
```

Body portion copies standing stats exactly. Command block gets identity stats (mean=0, var=1) because commands are pre-normalized to [-1, 1].

Reward normalization reset: `ret_rms.var = 100.0` for gentler initial scaling.

### PolicyTransfer

Transfers policy weights with proper initialization for new command dimensions.

**Initialization strategies:**
- `INIT_ZERO` — Conservative, policy initially ignores commands
- `INIT_XAVIER` — Recommended default, `√(2/(fan_in+fan_out))`
- `INIT_KAIMING` — He initialization, `√(2/fan_in)`
- `INIT_SMALL_NOISE` — Small random (0.1 scale)
- `INIT_FROM_VELOCITY` — Copy velocity feature weights as template

**Critical operations:**
- Value function re-initialized (standing value estimates corrupt walking gradients)
- `log_std` reset (standing models can have `log_std ≈ 9.0`, std ≈ 8000)
- Command weight boosted by `command_weight_scale` (default 5.0×) for visibility

### WarmupCollector

Runs random actions for 10k steps to populate VecNormalize statistics before training. Pins command block to identity stats after collection to prevent variance collapse.

## `model_manager.py` — Checkpoint Organization

### ModelManager

Organizes model saves into structured directories:
```
models/{task}/
├── latest/          — Most recent checkpoint
├── best/            — Best by metric (e.g., velocity_error)
├── checkpoints/
│   └── stage_{n}/   — Organized by curriculum stage
└── final/           — Production model
models/configs/
└── run_*.yaml       — Archived training configs
```

Key methods: `save_latest()`, `save_checkpoint()`, `save_best()`, `save_final()`, `archive_config()`.
Max 5 checkpoints per stage (configurable).

## `callbacks.py` — Training Callbacks

### Safety Callbacks (in training scripts)

| Callback | Purpose | Frequency |
|----------|---------|-----------|
| `LogStdClampCallback` | Clamp log_std to [-2, 0.5/1.0] | Every 500-2000 steps |
| `CommandStatsProtectorCallback` | Re-pin command stats to identity | Every 10k steps |
| `EntropyScheduleCallback` | Linear entropy coef decay | Every update |
| `SaveVecNormCallback` | Save VecNormalize .pkl | Every 100k steps |

### WandB Callbacks

| Callback | Logs |
|----------|------|
| `VelocityTrackingWandBCallback` | Velocity errors, heights, episode stats |
| `CurriculumWandBCallback` | Stage transitions, per-stage metrics |
| `RewardBreakdownWandBCallback` | Individual reward components, standing ratio |
| `VideoRecordingCallback` | Evaluation videos |
