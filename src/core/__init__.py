# src/core/__init__.py
"""
Core building blocks for humanoid training.

Contains fundamental components used across environments and training:
- VelocityCommandGenerator: Target velocity command generation
- RewardCalculator: Modular reward computation
"""

from .command_generator import (
    VelocityCommandGenerator,
    VelocityCommandGeneratorWithSmoothing
)
from .rewards import (
    RewardCalculator,
    AdaptiveRewardCalculator,
    RewardWeights,
    RewardMetrics
)

__all__ = [
    'VelocityCommandGenerator',
    'VelocityCommandGeneratorWithSmoothing',
    'RewardCalculator',
    'AdaptiveRewardCalculator',
    'RewardWeights',
    'RewardMetrics',
]

