# Humanoid Walking

Reinforcement learning for humanoid velocity tracking using MuJoCo physics simulation.

## Overview

This project trains a humanoid robot to follow velocity commands `[vx, vy, yaw_rate]` using the **3-Prompt Design**:

1. **VelocityCommandGenerator** (Prompt 1): Generates target commands with uniform sampling at 2-5 second intervals, 15% stop probability
2. **Plotting Script** (Prompt 2): Visualizes generated commands over 60 seconds
3. **Reward Function** (Prompt 3): `R_total = R_tracking + R_upright + R_effort`
   - `R_tracking = exp(-β * ||v_target - v_agent||²)` (Gaussian kernel, β=5.0)
   - `R_upright = +10.0` if upright, else 0 (binary survival)
   - `R_effort = -0.01 * ||action||²` (action penalty)

## Project Structure

```
humanoidNavigation/
├── config/
│   └── training_config.yaml    # Walking training configuration
├── data/
│   ├── checkpoints/            # Training checkpoints
│   └── videos/                 # Recorded videos
├── docs/
│   └── VELOCITY_TRACKING_DESIGN.md
├── scripts/
│   ├── train_walking.py        # Main training script
│   ├── evaluate.py             # Evaluation script
│   ├── record_video.py         # Video recording
│   └── debug/
│       ├── test_walking.py     # Walking tests
│       └── walking_plot.py     # Walking visualization
├── src/
│   ├── core/
│   │   ├── command_generator.py   # VelocityCommandGenerator (Prompt 1)
│   │   └── rewards.py             # RewardCalculator
│   ├── environments/
│   │   ├── walking_env.py         # Walking environment (Prompt 3)
│   │   └── walking_curriculum.py  # Curriculum wrapper
│   ├── training/
│   │   ├── callbacks.py           # WandB callbacks
│   │   └── model_manager.py       # Checkpoint management
│   ├── utils/
│   │   └── plot_velocity_commands.py  # Plotting (Prompt 2)
│   └── visualization/
│       ├── plotting.py
│       └── rendering.py
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Basic training
python scripts/train_walking.py

# With custom parameters
python scripts/train_walking.py --timesteps 5000000 --n-envs 8

# Debug mode (single process, no multiprocessing)
python scripts/train_walking.py --debug

# Resume from checkpoint
python scripts/train_walking.py --model models/walking/latest/model.zip
```

### Recording Videos

```bash
# Record with velocity command
python scripts/record_video.py \
    --model models/walking/final/model.zip \
    --vx 1.0 --vy 0.5 \
    --output walking_demo.mp4

# Different speeds
python scripts/record_video.py --model path/to/model.zip --vx 0.0   # Stand still
python scripts/record_video.py --model path/to/model.zip --vx 2.0   # Fast walk
```

### Evaluation

```bash
python scripts/evaluate.py \
    --model models/walking/final/model.zip \
    --episodes 10 \
    --vx 1.0 --vy 0.0
```

### Plot Command Generator (Prompt 2)

```bash
cd src/utils
python plot_velocity_commands.py
```

## Configuration

All parameters in `config/training_config.yaml`:

```yaml
walking:
  # Command generator ranges (Prompt 1)
  cmd_vx_min: -0.5
  cmd_vx_max: 1.5
  cmd_vy_min: -0.5
  cmd_vy_max: 0.5
  cmd_yaw_min: -1.0
  cmd_yaw_max: 1.0
  stop_probability: 0.15  # 15% stop command
  
  # Reward function (Prompt 3)
  tracking_beta: 5.0  # exp(-β * error²)
  
  # Training
  n_envs: 12  # 4-8 for Colab, 12-32 for local
  total_timesteps: 10_000_000
```

## Key Features

- **Velocity Tracking**: Follow any velocity command `[vx, vy, yaw_rate]`
- **Command Switching**: Random command changes every 2-5 seconds
- **Braking Practice**: 15% probability of stop command
- **Curriculum Learning**: Progressive speed increases
- **Robust Training**: Push perturbations, domain randomization
- **WandB Logging**: Track velocity error, curriculum stage

## Requirements

- Python 3.8+
- `gymnasium[mujoco]` - MuJoCo simulation
- `stable-baselines3` - PPO algorithm
- `torch` - Neural networks
- `opencv-python` - Video recording
- `matplotlib` - Visualization
- `wandb` (optional) - Logging
