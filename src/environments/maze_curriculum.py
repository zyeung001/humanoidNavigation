# maze_curriculum.py
"""
Progressive maze navigation curriculum.

7-stage curriculum from open arena to complex procedural mazes.
Advancement based on goal-reaching success rate.
"""

import numpy as np
from collections import deque

try:
    from src.maze.maze_generator import generate_maze_dfs, generate_maze_prims, open_arena, corridor
    from src.maze.maze_maps import L_MAZE, U_MAZE
    HAS_MAZE = True
except ImportError:
    HAS_MAZE = False


# Curriculum stage definitions
MAZE_STAGES = [
    {
        "name": "open_arena",
        "description": "Open arena — learn goal-seeking with no obstacles",
        "maze_type": "open_arena",
        "rows": 3,
        "cols": 3,
        "min_episode_length": 100,
        "required_success_rate": 0.40,
    },
    {
        "name": "corridor",
        "description": "Corridor — walk between walls to reach end",
        "maze_type": "corridor",
        "rows": 1,
        "cols": 5,
        "min_episode_length": 200,
        "required_success_rate": 0.40,
    },
    {
        "name": "l_maze",
        "description": "L-maze — navigate one corner",
        "maze_type": "l_maze",
        "rows": 3,
        "cols": 3,
        "min_episode_length": 300,
        "required_success_rate": 0.40,
    },
    {
        "name": "u_maze",
        "description": "U-maze — navigate U-turns",
        "maze_type": "u_maze",
        "rows": 2,
        "cols": 3,
        "min_episode_length": 400,
        "required_success_rate": 0.40,
    },
    {
        "name": "dfs_3x3",
        "description": "DFS 3x3 — small procedural maze",
        "maze_type": "dfs",
        "rows": 3,
        "cols": 3,
        "min_episode_length": 500,
        "required_success_rate": 0.40,
    },
    {
        "name": "dfs_5x5",
        "description": "DFS 5x5 — medium procedural maze",
        "maze_type": "dfs",
        "rows": 5,
        "cols": 5,
        "min_episode_length": 800,
        "required_success_rate": 0.40,
    },
    {
        "name": "prims_5x5",
        "description": "Prim's 5x5 — complex branching maze",
        "maze_type": "prims",
        "rows": 5,
        "cols": 5,
        "min_episode_length": 1000,
        "required_success_rate": 0.40,
    },
]


class MazeCurriculum:
    """Manages progressive maze difficulty for navigation training.

    Tracks success rate over a window of recent episodes and advances
    to harder maze types when the agent consistently reaches goals.
    """

    def __init__(
        self,
        start_stage=0,
        max_stage=6,
        window_size=30,
        seed=None,
    ):
        """
        Args:
            start_stage: Initial curriculum stage (0-6).
            max_stage: Maximum stage to advance to.
            window_size: Number of episodes for success rate calculation.
            seed: Random seed for procedural maze generation.
        """
        self.current_stage = min(start_stage, len(MAZE_STAGES) - 1)
        self.max_stage = min(max_stage, len(MAZE_STAGES) - 1)
        self.window_size = window_size
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        self.episode_results = deque(maxlen=window_size)
        self.total_episodes = 0
        self.stage_episodes = 0

    @property
    def stage_config(self):
        """Get current stage configuration."""
        return MAZE_STAGES[self.current_stage]

    @property
    def success_rate(self):
        """Current success rate over the episode window."""
        if len(self.episode_results) == 0:
            return 0.0
        return sum(self.episode_results) / len(self.episode_results)

    def get_maze_grid(self):
        """Generate a maze grid for the current curriculum stage.

        Returns:
            2D numpy array (1=wall, 0=open).
        """
        cfg = self.stage_config
        maze_type = cfg["maze_type"]

        if not HAS_MAZE:
            # Fallback: simple open grid
            h, w = 2 * cfg["rows"] + 1, 2 * cfg["cols"] + 1
            grid = np.zeros((h, w), dtype=np.int32)
            grid[0, :] = 1
            grid[-1, :] = 1
            grid[:, 0] = 1
            grid[:, -1] = 1
            return grid

        if maze_type == "open_arena":
            return open_arena(cfg["rows"], cfg["cols"])
        elif maze_type == "corridor":
            return corridor(cfg["cols"], cfg["rows"])
        elif maze_type == "l_maze":
            return L_MAZE.copy()
        elif maze_type == "u_maze":
            return U_MAZE.copy()
        elif maze_type == "dfs":
            seed = self._rng.randint(0, 100000)
            return generate_maze_dfs(cfg["rows"], cfg["cols"], seed=seed)
        elif maze_type == "prims":
            seed = self._rng.randint(0, 100000)
            return generate_maze_prims(cfg["rows"], cfg["cols"], seed=seed)
        else:
            return open_arena(cfg["rows"], cfg["cols"])

    def record_episode(self, goal_reached, episode_length):
        """Record an episode result and check for stage advancement.

        Args:
            goal_reached: Whether the agent reached the goal.
            episode_length: Length of the episode in steps.

        Returns:
            True if the stage was advanced, False otherwise.
        """
        self.total_episodes += 1
        self.stage_episodes += 1

        # Count as success only if goal reached AND survived long enough
        min_len = self.stage_config["min_episode_length"]
        success = goal_reached and episode_length >= min_len
        self.episode_results.append(1 if success else 0)

        # Check advancement
        if self._should_advance():
            return self._advance()
        return False

    def _should_advance(self):
        """Check if advancement criteria are met."""
        if self.current_stage >= self.max_stage:
            return False
        if len(self.episode_results) < self.window_size:
            return False
        required = self.stage_config["required_success_rate"]
        return self.success_rate >= required

    def _advance(self):
        """Advance to the next curriculum stage."""
        old_stage = self.current_stage
        self.current_stage = min(self.current_stage + 1, self.max_stage)
        self.episode_results.clear()
        self.stage_episodes = 0

        if self.current_stage != old_stage:
            print(f"[MazeCurriculum] Advanced: Stage {old_stage} → {self.current_stage} "
                  f"({self.stage_config['name']})")
            return True
        return False

    def get_info(self):
        """Get curriculum status info dict."""
        return {
            "maze_stage": self.current_stage,
            "maze_stage_name": self.stage_config["name"],
            "maze_success_rate": self.success_rate,
            "maze_total_episodes": self.total_episodes,
            "maze_stage_episodes": self.stage_episodes,
        }
