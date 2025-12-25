# src/utils/__init__.py
"""
Utility modules for humanoid training.

Contains:
- visualization: Display and rendering utilities
- plot_velocity_commands: Velocity command visualization (Prompt 2)
- wandb_callbacks: WandB logging for training metrics
"""

from .visualization import setup_display, test_environment
from .plot_velocity_commands import simulate_and_plot
from .wandb_callbacks import (
    VelocityTrackingWandBCallback,
    CurriculumWandBCallback,
    VideoRecordingCallback,
    init_wandb_run,
    finish_wandb_run
)

__all__ = [
    # Visualization
    'setup_display',
    'test_environment',
    # Plotting
    'simulate_and_plot',
    # WandB logging
    'VelocityTrackingWandBCallback',
    'CurriculumWandBCallback',
    'VideoRecordingCallback',
    'init_wandb_run',
    'finish_wandb_run',
]
