# maze_generator.py
"""
Procedural maze generation algorithms.

Two algorithms producing 2D binary grids (1=wall, 0=open):
- DFS Recursive Backtracker: Long corridors, simpler paths
- Randomized Prim's: More branching, shorter dead ends

Grid convention: (2*rows+1) x (2*cols+1)
  - Even indices are walls/borders
  - Odd indices are cells
"""

import numpy as np


def generate_maze_dfs(rows, cols, seed=None):
    """Generate a maze using DFS recursive backtracker.

    Produces long corridors with fewer branches — good for early curriculum stages.

    Args:
        rows: Number of cell rows.
        cols: Number of cell columns.
        seed: Random seed for reproducibility.

    Returns:
        2D numpy array of shape (2*rows+1, 2*cols+1). 1=wall, 0=open.
    """
    rng = np.random.RandomState(seed)
    h, w = 2 * rows + 1, 2 * cols + 1
    grid = np.ones((h, w), dtype=np.int32)

    # Start at cell (0, 0) -> grid position (1, 1)
    stack = [(0, 0)]
    visited = set()
    visited.add((0, 0))
    grid[1, 1] = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                neighbors.append((nr, nc))

        if neighbors:
            nr, nc = neighbors[rng.randint(len(neighbors))]
            # Remove wall between current and neighbor
            wr, wc = 1 + 2 * r + (nr - r), 1 + 2 * c + (nc - c)
            grid[wr, wc] = 0
            grid[1 + 2 * nr, 1 + 2 * nc] = 0
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    return grid


def generate_maze_prims(rows, cols, seed=None):
    """Generate a maze using randomized Prim's algorithm.

    Produces more branching with shorter dead ends — harder navigation.

    Args:
        rows: Number of cell rows.
        cols: Number of cell columns.
        seed: Random seed for reproducibility.

    Returns:
        2D numpy array of shape (2*rows+1, 2*cols+1). 1=wall, 0=open.
    """
    rng = np.random.RandomState(seed)
    h, w = 2 * rows + 1, 2 * cols + 1
    grid = np.ones((h, w), dtype=np.int32)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Start from cell (0, 0)
    grid[1, 1] = 0
    in_maze = set()
    in_maze.add((0, 0))

    # Frontier: walls adjacent to maze cells
    frontier = []
    for dr, dc in directions:
        nr, nc = dr, dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.append((nr, nc, 0, 0))  # (new_cell_r, new_cell_c, from_r, from_c)

    while frontier:
        idx = rng.randint(len(frontier))
        nr, nc, fr, fc = frontier.pop(idx)

        if (nr, nc) in in_maze:
            continue

        # Carve path
        in_maze.add((nr, nc))
        grid[1 + 2 * nr, 1 + 2 * nc] = 0
        # Remove wall between
        wr, wc = 1 + 2 * fr + (nr - fr), 1 + 2 * fc + (nc - fc)
        grid[wr, wc] = 0

        # Add new frontier cells
        for dr, dc in directions:
            nnr, nnc = nr + dr, nc + dc
            if 0 <= nnr < rows and 0 <= nnc < cols and (nnr, nnc) not in in_maze:
                frontier.append((nnr, nnc, nr, nc))

    return grid


def open_arena(rows, cols):
    """Create an open arena with no internal walls.

    Useful for curriculum Stage 0 (goal-seeking with no obstacles).

    Args:
        rows: Number of cell rows.
        cols: Number of cell columns.

    Returns:
        2D numpy array with only border walls.
    """
    h, w = 2 * rows + 1, 2 * cols + 1
    grid = np.zeros((h, w), dtype=np.int32)
    # Border walls
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    return grid


def corridor(length, width=1):
    """Create a straight corridor maze.

    Useful for curriculum Stage 1.

    Args:
        length: Number of cells long.
        width: Number of cells wide (default 1).

    Returns:
        2D numpy array representing the corridor.
    """
    return open_arena(width, length)
