# Humanoid Navigation

Reinforcement learning for training humanoid robots to stand, walk, and navigate using MuJoCo physics simulation and PPO.

## Overview

Two-stage training pipeline with curriculum learning and transfer:

1. **Standing** — Balance at 1.40m target height via 6-stage curriculum (1.29m ±0.04m achieved, 90%+ success)
2. **Walking** — Command-conditioned velocity tracking via 3-stage speed curriculum (0-1.5 m/s, any direction)
3. **Navigation** — Goal-directed locomotion with obstacle avoidance (future work)

Standing policy weights transfer to walking via `transfer_standing_to_walking()`, which extends the observation space from 1484 to 1493 dims and re-initializes the value function.

## Architecture

```
Standing (5M steps)           Transfer Learning           Walking (30M steps)
┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ StandingCurriculumEnv│    │ VecNormalizeExtender  │    │ WalkingCurriculumEnv │
│ 6 stages: 0.80→1.40m│ ──→│ PolicyTransfer        │──→ │ 3 stages: 0.15→0.8  │
│ Height targeting     │    │ WarmupCollector        │    │ Velocity tracking    │
└─────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

## Project Structure

```
humanoidNavigation/
├── config/                  # Training configuration (single YAML)
│   └── training_config.yaml
├── models/                  # Trained weights + VecNormalize stats
├── scripts/                 # Training, evaluation, debug entry points
│   ├── train_standing.py
│   ├── train_walking.py
│   ├── evaluate.py
│   ├── record_video.py
│   └── debug/               # Diagnostic and analysis tools
└── src/                     # Source library
    ├── agents/              # High-level agent wrappers + diagnostics
    ├── core/                # Reward calculator + velocity command generator
    ├── environments/        # Gym wrappers + curriculum learning
    ├── training/            # Transfer learning, model management, callbacks
    └── utils/               # Visualization + WandB utilities
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

**Standing (from scratch):**
```bash
python scripts/train_standing.py --timesteps 5000000
```

**Walking (from standing model — recommended):**
```bash
python scripts/train_walking.py --from-standing \
    --model models/best_standing_model.zip \
    --timesteps 30000000
```

**Resume training:**
```bash
python scripts/train_walking.py \
    --model models/walking/final/final_walking_model.zip \
    --vecnorm models/walking/final/vecnorm_walking.pkl \
    --timesteps 30000000
```

### Evaluation & Video Recording

```bash
# Evaluate standing
python scripts/evaluate.py --task standing \
    --model models/best_standing_model.zip

# Evaluate walking + record video
python scripts/evaluate.py --task walking \
    --model models/walking/final/final_walking_model.zip --record

# Walking with specific velocity command
python scripts/evaluate.py --task walking \
    --model models/walking/final/final_walking_model.zip \
    --vx 1.0 --vy 0.0 --record
```

### Debug / Analysis

```bash
# Standalone standing test
python scripts/debug/test_standing.py --model models/best_standing_model.zip

# Standalone walking test
python scripts/debug/test_walking.py \
    --model models/walking/final/final_walking_model.zip

# Diagnose transfer learning
python scripts/debug/diagnose_transfer.py
```

## Key Capabilities

### Standing Controller
- Maintains height at 1.29m ±0.04m (target: 1.40m)
- 90%+ success rate over 1000-1500+ step episodes
- Perfect upright orientation (q_w > 0.98)
- Minimal XY drift
- Robust to mass/friction perturbations (±5%)

### Walking Controller
- Command-conditioned on desired velocity (vx, vy) in world frame
- Supports speeds from 0.0 to 1.5 m/s
- Any direction (forward, backward, sideways, diagonal)
- Stable height during locomotion (1.20-1.35m)
- Velocity tracking error < 0.4 m/s at max speed
- Robust to push perturbations (20-80 N)

## Configuration

All training hyperparameters live in `config/training_config.yaml`, organized under `standing:` and `walking:` top-level keys. Covers:

- Network architecture (`[512, 512, 256]` policy and value networks, SiLU activation)
- Learning rate schedules (linear decay)
- Reward function weights and Gaussian kernel bandwidths
- Curriculum stages and advancement criteria
- Domain randomization ranges
- VecNormalize clip values
- WandB logging settings

## Requirements

See `requirements.txt`. Key dependencies:

| Package | Purpose |
|---------|---------|
| `gymnasium[mujoco]` | MuJoCo physics simulation |
| `stable-baselines3` | PPO implementation |
| `torch` | Neural network backend |
| `opencv-python` | Video recording |
| `numpy` | Numerical computing |
| `matplotlib` | Visualization |
| `pyyaml` | Configuration parsing |

Optional: `wandb` for experiment tracking (enabled via config).
