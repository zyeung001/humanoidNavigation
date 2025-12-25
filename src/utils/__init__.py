# src/utils/__init__.py
"""
Utility modules for humanoid training.

Contains:
- visualization: Display and rendering utilities
- plot_velocity_commands: Velocity command visualization (Prompt 2)
"""

from .visualization import setup_display, test_environment
from .plot_velocity_commands import simulate_and_plot

__all__ = [
    # Visualization
    'setup_display',
    'test_environment',
    # Plotting
    'simulate_and_plot',
]
