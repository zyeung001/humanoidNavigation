# Humanoid Navigation

**End-to-end reinforcement learning for humanoid locomotion and autonomous maze navigation.**

Train a simulated humanoid to stand, walk with velocity commands, and navigate procedurally generated mazes — all in MuJoCo with PPO.

## Training Pipeline

```
                    Transfer Learning
Standing (5M steps) ──────────────────> Walking (30M steps) ──────────> Maze Navigation
                                                                        (zero-shot)

┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ StandingCurriculumEnv│  │ WalkingCurriculumEnv │  │ NavigationController│
│                     │  │                     │  │                     │
│ 6-stage curriculum  │  │ 3-stage curriculum  │  │ A* pathfinding      │
│ 0.80m → 1.40m      │──│ 0.15 → 0.80 m/s    │──│ Pure pursuit control│
│ Height balancing    │  │ Velocity tracking   │  │ Frozen walking PPO  │
│ Domain randomization│  │ Push perturbations  │  │ Procedural mazes    │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

**Key ideas:**
- **Curriculum learning** — progressive difficulty prevents reward hacking and collapse
- **Transfer learning** — standing policy bootstraps walking (obs space extended 1484 → 1493 dims, value function re-initialized)
- **Zero-shot navigation** — walking policy is frozen; a classical planner (A* + pure pursuit) issues velocity commands

## Results

| Capability | Performance |
|------------|-------------|
| Standing balance | 1.29m ± 0.04m height, 90%+ success over 1500-step episodes |
| Walking speed range | 0 – 0.80 m/s, command-conditioned (vx, vy, yaw) |
| Orientation stability | q_w > 0.98 (near-perfect upright) |
| Robustness | Survives ±5% mass/friction perturbation, 20–80N push forces |
| Maze navigation | Solves corridor, L-maze, U-maze, and procedural DFS/Prim's mazes |

## Quick Start

### Installation

```bash
make setup
# or: pip install -r requirements.txt
```

Requires Python 3.10+ and a working MuJoCo installation.

### Train

```bash
# Standing controller (from scratch)
make train-standing ARGS="--timesteps 5000000"

# Walking controller (transfer from standing — recommended)
make train-walking ARGS="--from-standing --model models/best_standing_model.zip --timesteps 30000000"

# Resume walking training (paths created by ModelManager during training)
make train-walking ARGS="--model models/walking/final/final_walking_model.zip --vecnorm models/walking/final/vecnorm_walking.pkl"
```

### Evaluate

```bash
# Evaluate + record video
make evaluate ARGS="--task walking --model models/walking/final/final_walking_model.zip --record"

# Walking with specific velocity command
make evaluate ARGS="--task walking --model models/walking/final/final_walking_model.zip --vx 1.0 --vy 0.0 --record"
```

### Maze Navigation

```bash
# Corridor maze with video
python scripts/run_maze_nav.py --maze-type corridor --model models/walking/best/model.zip --record

# Random 5x5 maze
python scripts/run_maze_nav.py --maze-type dfs_5x5 --model models/walking/best/model.zip --record

# Open arena
python scripts/run_maze_nav.py --maze-type open --model models/walking/best/model.zip --speed 0.3 --record
```

Available maze types: `corridor`, `open`, `l_maze`, `u_maze`, `medium`, `open_arena`, `dfs_3x3`, `dfs_5x5`, `prims_3x3`, `prims_5x5`

| Flag | Default | Description |
|------|---------|-------------|
| `--maze-type` | `corridor` | Maze layout |
| `--model` | required | Path to walking model `.zip` |
| `--vecnorm` | auto-detect | Path to VecNormalize `.pkl` |
| `--speed` | `0.3` | Target walking speed (m/s) |
| `--cell-size` | `2.0` | Maze cell size in meters |
| `--max-steps` | `5000` | Max simulation steps |
| `--record` | off | Save video to `data/videos/maze_nav.mp4` |

## Project Structure

```
humanoidNavigation/
├── config/
│   └── training_config.yaml        # All hyperparameters (standing, walking, maze)
├── models/                         # Trained weights + VecNormalize stats
│   ├── best_standing_model.zip     # Standing model
│   ├── vecnorm.pkl                 # Standing normalization stats
│   └── walking/final/              # Walking model (more dirs created during training)
├── scripts/
│   ├── train_standing.py           # Standing training entry point
│   ├── train_walking.py            # Walking training entry point
│   ├── evaluate.py                 # Unified evaluation + video recording
│   ├── run_maze_nav.py             # Maze navigation demo
│   ├── record_video.py             # Legacy video recording
│   └── debug/                      # Diagnostic and analysis tools
├── src/
│   ├── core/                       # Reward calculator + velocity command generator
│   │   ├── rewards.py              # Gaussian kernel rewards (RewardCalculator)
│   │   └── command_generator.py    # Velocity command sampling with smoothing
│   ├── environments/               # Gym wrappers + curriculum learning
│   │   ├── standing_env.py         # Standing balance environment
│   │   ├── standing_curriculum.py  # 6-stage height curriculum
│   │   ├── walking_env.py          # Command-conditioned walking environment
│   │   └── walking_curriculum.py   # 3-stage speed curriculum
│   ├── maze/                       # Maze generation, solving, and navigation
│   │   ├── maze_generator.py       # DFS and Prim's procedural generation
│   │   ├── maze_maps.py            # Predefined maze layouts
│   │   ├── maze_mjcf.py            # Grid → MuJoCo XML with physical walls
│   │   ├── solver.py               # A* pathfinding
│   │   ├── navigation_controller.py # Pure pursuit waypoint following
│   │   └── maze_renderer.py        # Top-down minimap overlay
│   ├── training/                   # Transfer learning + model management
│   │   ├── transfer_utils.py       # Standing → walking policy transfer
│   │   ├── model_manager.py        # Checkpoint organization
│   │   ├── callbacks.py            # WandB logging callbacks
│   │   └── schedules.py            # LR and clip range decay schedules
│   ├── agents/                     # High-level agent wrapper + diagnostics
│   └── utils/                      # Visualization utilities
├── Makefile                        # Build targets (setup, train, evaluate, lint)
└── requirements.txt
```

## Technical Details

### Observation Space

| Component | Dims | Notes |
|-----------|------|-------|
| Humanoid-v5 base | 365 | Joint positions, velocities, contact forces |
| COM features | +6 | Center of mass position + velocity |
| History stacking | x4 | Temporal context for velocity estimation |
| Command block | +9 | Walking only: cmd, actual, error vectors |
| **Standing total** | **1484** | (365 + 6) x 4 |
| **Walking total** | **1493** | 1484 + 9 |

### Network Architecture

- Policy: `[512, 512, 256]` MLP with SiLU activation
- Value: `[512, 512, 256]` MLP (separate network)
- Orthogonal initialization
- Action space: 17 continuous joint torques

### Reward Design

Walking uses Gaussian kernel rewards for smooth gradients:

```
reward = weight * exp(-bandwidth * ||error||^2)
```

Dominant signal is velocity tracking (weight=25.0, bandwidth=2.0). Additional terms for height maintenance, upright orientation, action smoothness, and arm posture.

### Transfer Learning

Standing → walking transfer extends observation statistics and re-initializes critical components:

1. `VecNormalizeExtender` — grows obs normalization from 1484 → 1493 dims
2. `PolicyTransfer` — copies weights, Xavier-initializes command dimensions
3. `WarmupCollector` — 10k random steps to populate normalization statistics
4. Value function re-initialized (standing values corrupt walking gradients)
5. `log_std` reset (standing models can have std ~8000)

### Curriculum Learning

**Standing (6 stages):** Progressive height targets from 0.80m to 1.40m with increasing episode length requirements and domain randomization.

**Walking (3 stages):** Progressive speed from 0.15 to 0.80 m/s with increasing push perturbation strength and direction diversity. Anti-exploit mechanism requires actual movement to advance.

## Configuration

All hyperparameters in `config/training_config.yaml` under `standing:`, `walking:`, and `maze:` keys. Covers network architecture, learning rate schedules, reward weights, curriculum stages, and domain randomization.

## Requirements

| Package | Purpose |
|---------|---------|
| `gymnasium[mujoco]` | MuJoCo physics simulation |
| `stable-baselines3` | PPO implementation |
| `torch` | Neural network backend |
| `opencv-python` | Video recording |
| `numpy` | Array computation |
| `matplotlib` | Visualization |
| `pyyaml` | Configuration |

Optional: `wandb` for experiment tracking.
