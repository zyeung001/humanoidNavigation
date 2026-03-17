# Humanoid Navigation

Reinforcement learning for training humanoid robots to stand, walk, and navigate using MuJoCo physics simulation and PPO.

## Overview

Two-stage training pipeline with curriculum learning and transfer:

1. **Standing** — Balance at 1.40m target height via 6-stage curriculum (1.29m ±0.04m achieved, 90%+ success)
2. **Walking** — Command-conditioned velocity tracking via 3-stage speed curriculum (0-0.6 m/s)
3. **Maze Navigation** — Frozen walking policy + pure pursuit controller navigates procedurally generated mazes

Standing policy weights transfer to walking via `transfer_standing_to_walking()`, which extends the observation space from 1484 to 1493 dims and re-initializes the value function.

## Architecture

```
Standing (5M steps)       Transfer Learning       Walking (30M steps)        Maze Navigation
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│StandingCurriculumEnv│   │VecNormalizeExtender │   │WalkingCurriculumEnv│   │ NavigationController│
│6 stages: 0.80→1.40m│──→│PolicyTransfer       │──→│3 stages: 0.15→0.6 │──→│ Pure pursuit + A*  │
│Height targeting     │   │WarmupCollector      │   │Velocity tracking   │   │ Frozen walking PPO │
└────────────────────┘   └────────────────────┘   └────────────────────┘   └────────────────────┘
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
│   ├── run_maze_nav.py      # Maze navigation demo
│   ├── record_video.py
│   └── debug/               # Diagnostic and analysis tools
└── src/                     # Source library
    ├── agents/              # High-level agent wrappers + diagnostics
    ├── core/                # Reward calculator + velocity command generator
    ├── environments/        # Gym wrappers + curriculum learning
    ├── maze/                # Maze generation, solving, and navigation
    │   ├── maze_generator.py    # DFS and Prim's maze generation
    │   ├── maze_maps.py         # Predefined maze layouts
    │   ├── maze_mjcf.py         # MuJoCo XML generation with walls
    │   ├── solver.py            # A* pathfinding
    │   └── navigation_controller.py  # Pure pursuit waypoint following
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

### Maze Navigation

```bash
# Run corridor maze with video recording
python scripts/run_maze_nav.py --maze-type corridor \
    --model models/walking/best/model.zip --record

# Smaller maze for faster runs
python scripts/run_maze_nav.py --maze-type corridor \
    --model models/walking/best/model.zip --record --cell-size 1.0

# Open arena
python scripts/run_maze_nav.py --maze-type open \
    --model models/walking/best/model.zip --record --speed 0.3

# Random 3x3 maze
python scripts/run_maze_nav.py --maze-type dfs_3x3 \
    --model models/walking/best/model.zip --record
```

**Available maze types:** `corridor`, `open`, `l_maze`, `u_maze`, `medium`, `open_arena`, `corridor_gen`, `dfs_3x3`, `dfs_5x5`, `prims_3x3`, `prims_5x5`

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--maze-type` | `corridor` | Maze layout |
| `--model` | — | Path to walking model `.zip` |
| `--vecnorm` | auto-detect | Path to VecNormalize `.pkl` |
| `--speed` | `0.3` | Target walking speed (m/s) |
| `--cell-size` | `2.0` | Maze cell size in meters |
| `--max-steps` | `5000` | Max simulation steps |
| `--record` | off | Save video to `data/videos/maze_nav.mp4` |
| `--record-interval` | `3` | Record every Nth frame |

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
- Command-conditioned on desired velocity (vx, vy, yaw_rate) in world frame
- 4-stage speed curriculum: 0.15 → 0.30 → 0.45 → 0.60 m/s
- Stable height during locomotion (~1.27m)
- Three-phase transfer warmup: VF warmup → gradual ramp → permanent policy scaling
- Arm posture penalty (curriculum-gated) for sim-to-real transfer

### Maze Navigation
- Frozen walking policy steered by pure pursuit controller
- A* pathfinding on procedural or predefined maze grids
- Supports DFS, Prim's, and hand-designed maze layouts
- Configurable cell size, wall thickness, and walking speed
- Auto-detects VecNormalize stats and reverses path direction when needed
- Dual-view video: third-person chase cam + top-down map overlay

## Configuration

All training hyperparameters live in `config/training_config.yaml`, organized under `standing:`, `walking:`, and `maze:` top-level keys. Covers:

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
