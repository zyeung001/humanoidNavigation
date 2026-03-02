# test_maze_solver.py
"""Tests for A* solver and navigation controller."""

import math

import numpy as np

from src.maze.solver import solve, _astar, _rdp_simplify
from src.maze.navigation_controller import NavigationController


# --- A* Solver Tests ---

def test_astar_simple_open_grid():
    """A* finds path in a fully open grid."""
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    path = _astar(grid, (1, 1), (3, 3))
    assert path is not None
    assert path[0] == (1, 1)
    assert path[-1] == (3, 3)


def test_astar_no_path():
    """A* returns None for an unsolvable maze."""
    grid = np.zeros((5, 5), dtype=np.int32)
    # Wall blocking all paths
    grid[2, :] = 1
    path = _astar(grid, (1, 1), (3, 3))
    assert path is None


def test_astar_start_on_wall():
    """A* returns None if start is on a wall."""
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[1, 1] = 1
    path = _astar(grid, (1, 1), (3, 3))
    assert path is None


def test_astar_same_start_goal():
    """A* returns path of length 1 when start equals goal."""
    grid = np.zeros((5, 5), dtype=np.int32)
    path = _astar(grid, (2, 2), (2, 2))
    assert path is not None
    assert len(path) == 1
    assert path[0] == (2, 2)


def test_solve_returns_world_coords():
    """solve() returns world-coordinate waypoints."""
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    waypoints = solve(grid, (1, 1), (3, 3), cell_size=2.0)
    assert waypoints is not None
    assert len(waypoints) >= 2
    # Check waypoints are tuples of floats
    for wp in waypoints:
        assert len(wp) == 2
        assert isinstance(wp[0], float)
        assert isinstance(wp[1], float)


# --- Path Simplification Tests ---

def test_rdp_straight_line():
    """RDP preserves endpoints of a straight line."""
    points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    simplified = _rdp_simplify(points, epsilon=0.1)
    assert len(simplified) == 2
    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]


def test_rdp_preserves_corners():
    """RDP keeps corner points in an L-shaped path."""
    points = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
    simplified = _rdp_simplify(points, epsilon=0.1)
    assert len(simplified) == 3  # All points kept


def test_rdp_reduces_waypoints():
    """RDP reduces waypoints on a jagged path."""
    # Create a zigzag with small deviations
    points = [(float(i), float(i) + 0.01 * (-1) ** i) for i in range(20)]
    simplified = _rdp_simplify(points, epsilon=0.1)
    assert len(simplified) < len(points)


# --- Navigation Controller Tests ---

def test_nav_controller_basic():
    """Controller produces valid commands."""
    waypoints = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
    nav = NavigationController(waypoints, target_speed=0.3)

    vx, vy, yaw = nav.get_command((0.0, 0.0), 0.0)
    assert -0.5 <= vx <= 0.5
    assert -0.5 <= vy <= 0.5
    assert -1.0 <= yaw <= 1.0


def test_nav_controller_goal_detection():
    """Controller detects when goal is reached."""
    waypoints = [(0.0, 0.0), (1.0, 0.0)]
    nav = NavigationController(waypoints, target_speed=0.3, goal_threshold=0.5)

    # Position near the goal
    vx, vy, yaw = nav.get_command((0.9, 0.0), 0.0)
    assert nav.goal_reached is True
    assert vx == 0.0 and vy == 0.0 and yaw == 0.0


def test_nav_controller_yaw_rate_clamped():
    """Yaw rate is clamped to max_yaw_rate."""
    waypoints = [(0.0, 0.0), (0.0, 5.0)]  # Goal is to the left
    nav = NavigationController(waypoints, max_yaw_rate=1.0, kp_yaw=10.0)

    _, _, yaw = nav.get_command((0.0, 0.0), 0.0)
    assert abs(yaw) <= 1.0 + 1e-6


def test_nav_controller_reset():
    """Controller reset clears goal_reached state."""
    waypoints = [(0.0, 0.0), (0.5, 0.0)]
    nav = NavigationController(waypoints, goal_threshold=1.0)

    nav.get_command((0.3, 0.0), 0.0)
    assert nav.goal_reached is True

    nav.reset()
    assert nav.goal_reached is False
    assert nav.current_waypoint_idx == 0


def test_nav_controller_empty_waypoints():
    """Controller handles empty waypoints gracefully."""
    nav = NavigationController([], target_speed=0.3)
    vx, vy, yaw = nav.get_command((0.0, 0.0), 0.0)
    assert vx == 0.0 and vy == 0.0 and yaw == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
