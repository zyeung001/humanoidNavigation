# src/maze/__init__.py
"""
Procedural maze generation with MuJoCo wall geometry.

Provides maze generation algorithms, MJCF XML generation for MuJoCo simulation,
predefined maze maps, and rendering utilities.
"""

from .maze_generator import (
    generate_maze_dfs,
    generate_maze_prims,
    open_arena,
    corridor,
)
from .maze_mjcf import MazeMJCFGenerator
from .maze_renderer import MazeRenderer
from .solver import solve as solve_maze
from .navigation_controller import NavigationController
from .maze_maps import (
    CORRIDOR,
    L_MAZE,
    U_MAZE,
    OPEN,
    MEDIUM_MAZE,
)

__all__ = [
    "generate_maze_dfs",
    "generate_maze_prims",
    "open_arena",
    "corridor",
    "MazeMJCFGenerator",
    "MazeRenderer",
    "CORRIDOR",
    "L_MAZE",
    "U_MAZE",
    "OPEN",
    "MEDIUM_MAZE",
    "solve_maze",
    "NavigationController",
]
