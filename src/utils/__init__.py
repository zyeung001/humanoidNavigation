# src/utils/__init__.py
"""
Utility modules for walking training.

Main modules are in:
- src.core: VelocityCommandGenerator, RewardCalculator
- src.training: ModelManager, callbacks
- src.visualization: plotting utilities
"""

# Keep plot_velocity_commands for Prompt 2
from .plot_velocity_commands import simulate_and_plot

__all__ = [
    'simulate_and_plot',
]
