# Humanoid Navigation

Humanoid Navigation - A reinforcement learning project for training humanoid robots to walk, stand, and navigate using MuJoCo physics simulation.

## Overview

This project focuses on developing AI agents that can control humanoid robots to perform various locomotion tasks:

- **Standing**: Maintaining perfect balance and upright posture (1.29m ±0.04m, 90%+ success)
- **Walking**: Command-conditioned locomotion at any desired world-frame velocity (0-3 m/s)
- **Navigation**: Moving towards goals while avoiding obstacles (future work)

## Features

- Physics-based simulation using MuJoCo
- Reinforcement learning training pipeline (PPO)
- Multiple locomotion behaviors (walking, standing)
- Configurable training environments with curriculum learning
- Real-time visualization and monitoring
- Performance metrics and evaluation tools
- Command-conditioned velocity control for walking

## Project Structure

```
humanoidNavigation/
├── config/              # Training configurations
├── data/                # Checkpoints, logs, videos
├── docs/                # Documentation
├── models/              # Saved models and VecNormalize stats
├── scripts/             # Training, testing, and recording scripts
└── src/                 # Source code
    ├── agents/          # Training agents and diagnostics
    ├── environments/    # Environment wrappers
    └── utils/           # Utility functions
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

**Standing Task:**
```bash
python scripts/train_standing.py --timesteps 10000000
```

**Walking Task:**
```bash
python scripts/train_walking.py --timesteps 15000000
```

### Recording Videos

**Standing:**
```bash
python scripts/record_video.py \
    --task standing \
    --model models/final_standing_model.zip \
    --vecnorm models/vecnorm.pkl \
    --episodes 1 --steps 2000
```

**Walking (with velocity command):**
```bash
python scripts/record_video.py \
    --task walking \
    --model models/final_walking_model.zip \
    --vecnorm models/vecnorm_walking.pkl \
    --vx_target 1.0 --vy_target 0.0 \
    --episodes 1 --steps 2000
```

### Testing

```bash
# Test standing
python scripts/test/test_standing.py \
    --model models/final_standing_model.zip \
    --vecnorm models/vecnorm.pkl

# Test walking
python scripts/test/test_walking.py \
    --model models/final_walking_model.zip \
    --vecnorm models/vecnorm_walking.pkl \
    --episodes 5 --steps 2000
```

## Key Capabilities

### Standing Controller
- Maintains height at 1.29m ±0.04m
- 90%+ success rate
- 1000-1500+ step episodes
- Perfect upright orientation (w > 0.98)
- Minimal XY drift

### Walking Controller
- Command-conditioned on desired velocity (vx, vy)
- Supports speeds from 0.0 to 3.0 m/s
- Any direction (forward, backward, sideways, diagonal)
- Stable height (1.20-1.35m during locomotion)
- Velocity tracking error < 0.4 m/s at max speed
- Can stand perfectly when command = (0,0)

## Requirements

See `requirements.txt` for full list. Key dependencies:
- `gymnasium[mujoco]` - MuJoCo physics simulation
- `stable-baselines3` - PPO implementation
- `torch` - Neural network backend
- `opencv-python` - Video recording
- `matplotlib` - Visualization
- `pyyaml` - Configuration files

## Configuration

All training parameters are configured in `config/training_config.yaml`:
- Network architecture
- Learning rate schedules
- Reward function weights
- Curriculum learning stages
- Domain randomization settings


