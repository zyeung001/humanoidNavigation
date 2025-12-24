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

# Lazy imports to avoid circular dependency issues
# Users should import directly from submodules:
#   from src.environments import make_walking_env
#   from src.core import RewardCalculator

__all__ = [
    'environments',
    'core',
    'training',
    'visualization',
    'agents',
]

