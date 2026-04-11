# src/utils/__init__.py
"""
Utility modules for humanoid training.

Contains:
- visualization: Display and rendering utilities
- plot_velocity_commands: Velocity command visualization
"""

import os
import platform


def configure_mujoco_gl():
    """Set MUJOCO_GL based on platform. Call before importing MuJoCo envs.

    - Linux: EGL (headless GPU rendering, required for Colab/servers)
    - macOS: leave unset (defaults to GLFW native window)
    - Windows: leave unset (auto-detects)
    """
    if platform.system() == "Linux":
        os.environ.setdefault("MUJOCO_GL", "egl")


def get_subprocess_start_method():
    """Return the best multiprocessing start method for SubprocVecEnv.

    - Windows: 'spawn' (only option)
    - macOS: 'fork' (forkserver not supported)
    - Linux: 'forkserver' (safer than fork with CUDA)
    """
    system = platform.system()
    if system == "Windows":
        return "spawn"
    elif system == "Darwin":
        return "fork"
    return "forkserver"


from .visualization import setup_display, test_environment
from .plot_velocity_commands import simulate_and_plot

__all__ = [
    # Platform helpers
    'configure_mujoco_gl',
    'get_subprocess_start_method',
    # Visualization
    'setup_display',
    'test_environment',
    # Plotting
    'simulate_and_plot',
]
