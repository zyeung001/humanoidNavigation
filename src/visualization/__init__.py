# src/visualization/__init__.py
"""
Visualization and rendering utilities.

Contains:
- Plotting: Command visualization, reward analysis
- Rendering: Display setup, video utilities
"""

from .rendering import setup_display, test_environment
from .plotting import (
    simulate_and_plot_commands,
    plot_reward_components,
    plot_episode_metrics
)

__all__ = [
    'setup_display',
    'test_environment',
    'simulate_and_plot_commands',
    'plot_reward_components',
    'plot_episode_metrics',
]

