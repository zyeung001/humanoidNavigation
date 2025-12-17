# src/training/__init__.py
"""
Training utilities for humanoid RL.

Contains:
- ModelManager: Checkpoint and weight storage management
- Callbacks: WandB logging, metrics tracking, curriculum logging
"""

from .model_manager import ModelManager
from .callbacks import (
    VelocityTrackingWandBCallback,
    CurriculumWandBCallback,
    VideoRecordingCallback,
    init_wandb_run,
    finish_wandb_run
)

__all__ = [
    'ModelManager',
    'VelocityTrackingWandBCallback',
    'CurriculumWandBCallback',
    'VideoRecordingCallback',
    'init_wandb_run',
    'finish_wandb_run',
]

