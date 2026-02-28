# src/core/ — Rewards + Command Generation

## `rewards.py` — Reward Calculator

### RewardWeights (dataclass)

Default weights (overridden by config YAML):

| Weight | Default | Purpose |
|--------|---------|---------|
| `tracking` | 25.0 | Velocity matching (DOMINANT) |
| `direction_bonus` | 8.0 | Moving in commanded direction |
| `height` | 3.0 | Height maintenance |
| `upright` | 2.0 | Orientation stability |
| `alive` | 0.5 | Survival bonus |
| `action_penalty` | 0.003 | Effort minimization |
| `jerk_penalty` | 0.008 | Action smoothness |

### RewardCalculator

Computes rewards using **Gaussian kernels** for smooth gradients:
```
reward = weight × exp(-bandwidth × ||error||²)
```

Key bandwidths:
- Velocity tracking: 2.0 (configurable via `reward_tracking_bandwidth` in config)
- Height: 10.0

Reward range: approximately [-8, +39] per step.

### AdaptiveRewardCalculator

Adjusts weights by curriculum stage:
- Stage 0-1: Survival focus (tracking=8, height=6, upright=4)
- Stage 2-3: Balance (tracking=10, height=5, upright=3)
- Stage 4+: Precision (tracking=12, direction=6)

### RewardMetrics (dataclass)

Container for 17 individual reward components, used for WandB logging and debugging.

## `command_generator.py` — Velocity Commands

### VelocityCommandGenerator

Generates target velocity commands `[vx, vy, yaw_rate]` at randomized intervals.

| Parameter | Default | Notes |
|-----------|---------|-------|
| vx range | [-0.5, 1.5] m/s | Forward-biased |
| vy range | [-0.5, 0.5] m/s | Lateral |
| yaw_rate range | [-1.0, 1.0] rad/s | Turning |
| Switch interval | 2-5 seconds | Randomized |
| Stop probability | 15% | Zero velocity command |

Commands persist for the full switch interval (not every step). `update_curriculum_ranges()` adjusts ranges per curriculum stage.

### VelocityCommandGeneratorWithSmoothing

Adds exponential smoothing to command transitions. `smoothing_tau` controls transition speed (0=no smoothing, 1=instant).
