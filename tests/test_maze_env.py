# test_maze_env.py
"""Tests for maze navigation environment and curriculum."""

import numpy as np

from src.environments.maze_env import MazeNavigationEnv
from src.environments.maze_curriculum import MazeCurriculum, MAZE_STAGES


# --- MazeNavigationEnv Tests ---

def test_maze_env_obs_dim_constant():
    """Verify expected observation dimension constant."""
    assert MazeNavigationEnv.NAV_OBS_DIM == 6
    assert MazeNavigationEnv.WALKING_OBS_DIM == 1495
    assert MazeNavigationEnv.TOTAL_OBS_DIM == 1501


def test_maze_env_nav_obs_shape():
    """Navigation observation should be a 6-dim float array."""
    env = MazeNavigationEnv.__new__(MazeNavigationEnv)
    env.goal_pos = np.array([5.0, 5.0])
    env.max_goal_dist = 20.0
    env.max_wall_dist = 10.0
    env.cell_size = 2.0
    env.grid = np.zeros((11, 11), dtype=np.int32)
    env.grid[0, :] = 1
    env.grid[-1, :] = 1
    env.grid[:, 0] = 1
    env.grid[:, -1] = 1

    # Mock methods
    env._get_agent_pos = lambda: np.array([0.0, 0.0])
    env._get_agent_heading = lambda: 0.0

    nav_obs = env._compute_nav_obs()
    assert nav_obs.shape == (6,)
    assert nav_obs.dtype == np.float32


def test_maze_env_nav_reward_range():
    """Navigation reward should be in a reasonable range."""
    env = MazeNavigationEnv.__new__(MazeNavigationEnv)
    env.prev_dist_to_goal = 5.0
    env.walking_reward_scale = 0.5

    class MockEnv:
        pass

    env.env = MockEnv()
    env._get_base_env = lambda: MockEnv()

    # Test proximity component
    reward = env._compute_nav_reward(dist_to_goal=0.0)
    # At dist=0: proximity=2.0, progress=5.0*clip(5.0, -0.5, 0.5)=2.5, time=-0.01
    assert reward > 0, "Close to goal should give positive reward"

    reward_far = env._compute_nav_reward(dist_to_goal=10.0)
    # Far from goal: proximity is small, progress is negative
    assert reward_far < reward, "Far from goal should give lower reward"


# --- MazeCurriculum Tests ---

def test_curriculum_stages_defined():
    """All 7 stages should be defined."""
    assert len(MAZE_STAGES) == 7


def test_curriculum_stage_names():
    """Verify stage name progression."""
    expected_names = ["open_arena", "corridor", "l_maze", "u_maze", "dfs_3x3", "dfs_5x5", "prims_5x5"]
    for i, name in enumerate(expected_names):
        assert MAZE_STAGES[i]["name"] == name, f"Stage {i} should be {name}"


def test_curriculum_initial_state():
    """Curriculum should start at stage 0 with 0 success rate."""
    curr = MazeCurriculum(start_stage=0)
    assert curr.current_stage == 0
    assert curr.success_rate == 0.0
    assert curr.total_episodes == 0


def test_curriculum_get_maze_grid():
    """get_maze_grid should return a valid 2D array for each stage."""
    for stage in range(len(MAZE_STAGES)):
        curr = MazeCurriculum(start_stage=stage, seed=42)
        grid = curr.get_maze_grid()
        assert grid.ndim == 2
        assert np.any(grid == 0), f"Stage {stage} grid should have open cells"
        assert np.any(grid == 1), f"Stage {stage} grid should have walls"


def test_curriculum_advancement():
    """Curriculum should advance when success rate exceeds threshold."""
    curr = MazeCurriculum(start_stage=0, window_size=5)

    # Record 5 successful episodes with sufficient length
    for _ in range(5):
        curr.record_episode(goal_reached=True, episode_length=200)

    assert curr.current_stage == 1, "Should advance to stage 1 after consistent success"


def test_curriculum_no_advance_on_failure():
    """Curriculum should not advance when success rate is low."""
    curr = MazeCurriculum(start_stage=0, window_size=10)

    # Record 10 failed episodes
    for _ in range(10):
        curr.record_episode(goal_reached=False, episode_length=50)

    assert curr.current_stage == 0, "Should stay at stage 0 after consistent failure"


def test_curriculum_max_stage_cap():
    """Curriculum should not advance past max_stage."""
    curr = MazeCurriculum(start_stage=5, max_stage=5, window_size=3)

    for _ in range(5):
        curr.record_episode(goal_reached=True, episode_length=2000)

    assert curr.current_stage == 5, "Should not exceed max_stage"


def test_curriculum_info():
    """get_info should return expected keys."""
    curr = MazeCurriculum(start_stage=0)
    info = curr.get_info()
    assert "maze_stage" in info
    assert "maze_stage_name" in info
    assert "maze_success_rate" in info
    assert "maze_total_episodes" in info


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
