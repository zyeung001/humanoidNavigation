# src/training/__init__.py
"""
Training utilities for humanoid RL.

Contains:
- ModelManager: Checkpoint and weight storage management
- Callbacks: WandB logging, metrics tracking, curriculum logging
- TransferUtils: Standing → Walking transfer learning utilities
"""

from .model_manager import ModelManager
from .callbacks import (
    VelocityTrackingWandBCallback,
    CurriculumWandBCallback,
    VideoRecordingCallback,
    init_wandb_run,
    finish_wandb_run
)
from .transfer_utils import (
    transfer_standing_to_walking,
    VecNormalizeExtender,
    PolicyTransfer,
    WarmupCollector,
)

__all__ = [
    'ModelManager',
    'VelocityTrackingWandBCallback',
    'CurriculumWandBCallback',
    'VideoRecordingCallback',
    'init_wandb_run',
    'finish_wandb_run',
    # Transfer learning
    'transfer_standing_to_walking',
    'VecNormalizeExtender',
    'PolicyTransfer',
    'WarmupCollector',
]

