# src/utils/__init__.py
"""
Utility modules (backwards compatibility layer).

NOTE: This module provides backwards compatibility. 
New code should import from the reorganized locations:

    # New way (preferred)
    from src.core import VelocityCommandGenerator, RewardCalculator
    from src.training import ModelManager, VelocityTrackingWandBCallback
    from src.visualization import setup_display, simulate_and_plot_commands

    # Old way (still works)
    from src.utils import VelocityCommandGenerator, RewardCalculator
"""

# Re-export from new locations for backwards compatibility
from .velocity_command_generator import (
    VelocityCommandGenerator,
    VelocityCommandGeneratorWithSmoothing
)
from .reward_calculator import (
    RewardCalculator,
    AdaptiveRewardCalculator,
    RewardWeights,
    RewardMetrics
)
from .model_manager import ModelManager
from .wandb_callbacks import (
    VelocityTrackingWandBCallback,
    CurriculumWandBCallback,
    VideoRecordingCallback,
    init_wandb_run,
    finish_wandb_run
)
from .visualization import setup_display, test_environment

__all__ = [
    # Command generation
    'VelocityCommandGenerator',
    'VelocityCommandGeneratorWithSmoothing',
    # Reward calculation
    'RewardCalculator',
    'AdaptiveRewardCalculator',
    'RewardWeights',
    'RewardMetrics',
    # Model management
    'ModelManager',
    # WandB logging
    'VelocityTrackingWandBCallback',
    'CurriculumWandBCallback',
    'VideoRecordingCallback',
    'init_wandb_run',
    'finish_wandb_run',
    # Visualization
    'setup_display',
    'test_environment',
]
