# src/__init__.py
"""
Humanoid Navigation - RL Training for Humanoid Locomotion

Package Structure:
    src/
    ├── core/           - Fundamental components (rewards, command generator)
    ├── environments/   - Gym environments (standing, walking)
    ├── training/       - Training utilities (callbacks, model manager)
    ├── visualization/  - Plotting and rendering
    └── agents/         - High-level agent classes

Quick imports:
    from src.environments import make_walking_env, make_standing_env
    from src.core import VelocityCommandGenerator, RewardCalculator
    from src.training import ModelManager, VelocityTrackingWandBCallback
"""

__version__ = "0.2.0"

# Re-export common items for convenience
from .environments import (
    WalkingEnv,
    WalkingCurriculumEnv,
    StandingEnv,
    StandingCurriculumEnv,
    make_walking_env,
    make_walking_curriculum_env,
    make_standing_env,
    make_standing_curriculum_env,
)

__all__ = [
    # Environments
    'WalkingEnv',
    'WalkingCurriculumEnv', 
    'StandingEnv',
    'StandingCurriculumEnv',
    'make_walking_env',
    'make_walking_curriculum_env',
    'make_standing_env',
    'make_standing_curriculum_env',
]

