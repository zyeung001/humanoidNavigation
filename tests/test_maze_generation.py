# test_maze_generation.py
"""Tests for maze generation, MJCF output, maps, and rendering."""

import numpy as np
import xml.etree.ElementTree as ET

from src.maze.maze_generator import generate_maze_dfs, generate_maze_prims, open_arena, corridor
from src.maze.maze_maps import CORRIDOR, L_MAZE, U_MAZE, OPEN, MEDIUM_MAZE
from src.maze.maze_mjcf import MazeMJCFGenerator
from src.maze.maze_renderer import MazeRenderer


# --- Maze Generator Tests ---

def test_dfs_grid_dimensions():
    for rows, cols in [(3, 3), (5, 5), (4, 7), (1, 1)]:
        grid = generate_maze_dfs(rows, cols, seed=42)
        assert grid.shape == (2 * rows + 1, 2 * cols + 1), f"DFS grid shape mismatch for {rows}x{cols}"


def test_prims_grid_dimensions():
    for rows, cols in [(3, 3), (5, 5), (4, 7), (1, 1)]:
        grid = generate_maze_prims(rows, cols, seed=42)
        assert grid.shape == (2 * rows + 1, 2 * cols + 1), f"Prims grid shape mismatch for {rows}x{cols}"


def test_dfs_cells_are_open():
    grid = generate_maze_dfs(3, 3, seed=42)
    # All cell positions (odd, odd) should be open
    for r in range(3):
        for c in range(3):
            assert grid[2 * r + 1, 2 * c + 1] == 0, f"DFS cell ({r},{c}) should be open"


def test_prims_cells_are_open():
    grid = generate_maze_prims(3, 3, seed=42)
    for r in range(3):
        for c in range(3):
            assert grid[2 * r + 1, 2 * c + 1] == 0, f"Prims cell ({r},{c}) should be open"


def test_dfs_borders_are_walls():
    grid = generate_maze_dfs(3, 3, seed=42)
    assert np.all(grid[0, :] == 1), "Top border should be walls"
    assert np.all(grid[-1, :] == 1), "Bottom border should be walls"
    assert np.all(grid[:, 0] == 1), "Left border should be walls"
    assert np.all(grid[:, -1] == 1), "Right border should be walls"


def test_open_arena():
    grid = open_arena(3, 3)
    assert grid.shape == (7, 7)
    # Borders are walls
    assert np.all(grid[0, :] == 1)
    assert np.all(grid[-1, :] == 1)
    # Interior is open
    assert np.all(grid[1:-1, 1:-1] == 0)


def test_corridor():
    grid = corridor(5, 1)
    assert grid.shape == (3, 11)
    # Interior row should be open
    assert np.all(grid[1, 1:-1] == 0)


def test_seeded_reproducibility():
    g1 = generate_maze_dfs(5, 5, seed=123)
    g2 = generate_maze_dfs(5, 5, seed=123)
    assert np.array_equal(g1, g2), "Same seed should produce same maze"


# --- Predefined Maps Tests ---

def test_predefined_map_shapes():
    assert OPEN.shape == (11, 11)
    assert CORRIDOR.shape == (3, 11)
    assert L_MAZE.shape == (7, 7)
    assert U_MAZE.shape == (6, 7)
    assert MEDIUM_MAZE.shape == (7, 7)


def test_predefined_maps_have_open_cells():
    for name, maze in [("OPEN", OPEN), ("CORRIDOR", CORRIDOR), ("L_MAZE", L_MAZE),
                        ("U_MAZE", U_MAZE), ("MEDIUM_MAZE", MEDIUM_MAZE)]:
        assert np.any(maze == 0), f"{name} should have at least one open cell"


def test_predefined_maps_have_walls():
    for name, maze in [("OPEN", OPEN), ("CORRIDOR", CORRIDOR), ("L_MAZE", L_MAZE),
                        ("U_MAZE", U_MAZE), ("MEDIUM_MAZE", MEDIUM_MAZE)]:
        assert np.any(maze == 1), f"{name} should have at least one wall"


# --- MJCF Generator Tests ---

def test_mjcf_generates_valid_xml():
    grid = open_arena(2, 2)
    gen = MazeMJCFGenerator(cell_size=2.0, wall_height=2.5)
    xml_path = gen.generate(grid)
    # Should parse without error
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assert root.tag == "mujoco"


def test_mjcf_wall_count():
    grid = open_arena(2, 2)
    gen = MazeMJCFGenerator()
    xml_path = gen.generate(grid)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    wall_geoms = root.findall(".//*[@name]")
    maze_walls = [g for g in wall_geoms if g.get("name", "").startswith("maze_wall_")]
    expected_walls = int(np.sum(grid == 1))
    assert len(maze_walls) == expected_walls


def test_mjcf_has_topdown_camera():
    grid = open_arena(2, 2)
    gen = MazeMJCFGenerator()
    xml_path = gen.generate(grid)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cameras = root.findall(".//*[@name='maze_topdown']")
    assert len(cameras) == 1


def test_grid_to_world():
    gen = MazeMJCFGenerator(cell_size=2.0)
    grid = np.zeros((5, 7))  # 5 rows, 7 cols
    x, y = gen.grid_to_world(0, 0, grid.shape)
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_sample_start_goal():
    grid = open_arena(3, 3)
    gen = MazeMJCFGenerator()
    start, goal = gen.sample_start_goal(grid, min_distance_cells=2, seed=42)
    assert grid[start[0], start[1]] == 0, "Start must be on open cell"
    assert grid[goal[0], goal[1]] == 0, "Goal must be on open cell"
    assert start != goal, "Start and goal must be different"


# --- Renderer Tests ---

def test_renderer_schematic():
    grid = open_arena(3, 3)
    renderer = MazeRenderer(grid, cell_size=2.0)
    fig = renderer.render_schematic(agent_pos=(1, 1), goal_pos=(5, 5))
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_renderer_occupancy():
    grid = open_arena(2, 2)
    renderer = MazeRenderer(grid, cell_size=2.0)
    occ = renderer.render_occupancy(resolution=0.5)
    assert occ.ndim == 2
    assert np.any(occ == 1)  # Has walls
    assert np.any(occ == 0)  # Has open space


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
