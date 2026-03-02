# solver.py
"""
A* maze solver with path simplification.

Finds shortest path through a 2D grid using A* with 8-connected neighbors,
then simplifies the path using Ramer-Douglas-Peucker algorithm.
"""

import heapq
import math

import numpy as np


def solve(grid, start_cell, goal_cell, cell_size=2.0):
    """Find shortest path through a maze grid using A*.

    Args:
        grid: 2D numpy array (0=open, 1=wall).
        start_cell: (row, col) start position.
        goal_cell: (row, col) goal position.
        cell_size: World-space size of each grid cell for coordinate conversion.

    Returns:
        List of (x, y) world-coordinate waypoints, or None if no path exists.
    """
    path = _astar(grid, start_cell, goal_cell)
    if path is None:
        return None

    # Convert grid path to world coordinates
    rows, cols = grid.shape
    offset_x = -(cols * cell_size) / 2.0
    offset_y = -(rows * cell_size) / 2.0

    world_path = []
    for r, c in path:
        x = offset_x + c * cell_size + cell_size / 2.0
        y = offset_y + (rows - 1 - r) * cell_size + cell_size / 2.0
        world_path.append((x, y))

    # Simplify path
    simplified = _rdp_simplify(world_path, epsilon=cell_size * 0.3)
    return simplified


def _astar(grid, start, goal):
    """A* pathfinding with 8-connected neighbors.

    Args:
        grid: 2D numpy array (0=open, 1=wall).
        start: (row, col) tuple.
        goal: (row, col) tuple.

    Returns:
        List of (row, col) tuples from start to goal, or None.
    """
    rows, cols = grid.shape
    sr, sc = start
    gr, gc = goal

    if grid[sr, sc] != 0 or grid[gr, gc] != 0:
        return None

    # 8-connected directions with costs
    directions = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),  # cardinal
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),             # diagonal
        (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
    ]

    def heuristic(r, c):
        # Chebyshev distance (consistent with 8-connected)
        dr = abs(r - gr)
        dc = abs(c - gc)
        return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)

    # Priority queue: (f_score, counter, row, col)
    counter = 0
    open_set = [(heuristic(sr, sc), counter, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0.0}

    while open_set:
        _, _, r, c = heapq.heappop(open_set)

        if (r, c) == (gr, gc):
            # Reconstruct path
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            path.reverse()
            return path

        for dr, dc, cost in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                # For diagonal moves, check that both adjacent cells are open
                if dr != 0 and dc != 0:
                    if grid[r + dr, c] != 0 or grid[r, c + dc] != 0:
                        continue

                new_g = g_score[(r, c)] + cost
                if new_g < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = new_g
                    f = new_g + heuristic(nr, nc)
                    came_from[(nr, nc)] = (r, c)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, nr, nc))

    return None


def _rdp_simplify(points, epsilon):
    """Ramer-Douglas-Peucker path simplification.

    Reduces a polyline to fewer points while preserving shape within epsilon tolerance.

    Args:
        points: List of (x, y) tuples.
        epsilon: Maximum perpendicular distance tolerance.

    Returns:
        Simplified list of (x, y) tuples.
    """
    if len(points) <= 2:
        return list(points)

    # Find the point farthest from the line between first and last
    start = np.array(points[0])
    end = np.array(points[-1])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    max_dist = 0.0
    max_idx = 0

    for i in range(1, len(points) - 1):
        pt = np.array(points[i])
        if line_len < 1e-10:
            dist = np.linalg.norm(pt - start)
        else:
            # Perpendicular distance from point to line
            dist = abs(np.cross(line_vec, start - pt)) / line_len
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        left = _rdp_simplify(points[:max_idx + 1], epsilon)
        right = _rdp_simplify(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]
