# src/__init__.py
"""
Humanoid Walking - RL Training for Humanoid Velocity Tracking

Following the 3-Prompt Design:
    Prompt 1: VelocityCommandGenerator - generates [vx, vy, yaw_rate] commands
    Prompt 2: Plotting script - visualizes commands over 60 seconds
    Prompt 3: Reward function - R_total = R_tracking + R_upright + R_effort

Package Structure:
    src/
    ├── core/           - VelocityCommandGenerator, RewardCalculator
    ├── environments/   - WalkingEnv with command tracking
    ├── training/       - ModelManager, WandB callbacks
    └── visualization/  - Plotting utilities

Quick imports:
    from src.environments import make_walking_env
    from src.core import VelocityCommandGenerator
    from src.training import ModelManager
"""

__version__ = "1.0.0"

__all__ = [
    'core',
    'environments',
    'training',
    'visualization',
]
