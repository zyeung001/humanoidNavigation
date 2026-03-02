# maze_env.py
"""
Maze navigation environment wrapping WalkingEnv.

Adds maze geometry, navigation observations, goal-conditioned rewards,
and a hierarchical controller interface for maze navigation training.
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

try:
    from src.maze.maze_mjcf import MazeMJCFGenerator
    from src.maze.maze_maps import OPEN
    HAS_MAZE = True
except ImportError:
    HAS_MAZE = False


class MazeNavigationEnv(gym.Wrapper):
    """
    Maze navigation environment that wraps a walking-capable humanoid.

    Extends the walking observation space with 6 navigation features and
    adds goal-conditioned rewards on top of locomotion rewards.

    Observation space: walking_obs (1493) + nav_features (6) = 1499 dims
    Navigation features (in agent-local frame):
        - goal_dx, goal_dy: Vector to goal (2)
        - goal_dist: Euclidean distance to goal, normalized (1)
        - goal_angle: Angle to goal relative to heading (1)
        - wall_front: Raycast distance to nearest wall ahead (1)
        - wall_left: Raycast distance to nearest wall left (1)
    """

    NAV_OBS_DIM = 6
    WALKING_OBS_DIM = 1493
    TOTAL_OBS_DIM = WALKING_OBS_DIM + NAV_OBS_DIM  # 1499

    def __init__(
        self,
        walking_env,
        grid=None,
        cell_size=2.0,
        wall_height=2.5,
        goal_threshold=0.5,
        max_episode_steps=10000,
        max_goal_dist=20.0,
        max_wall_dist=10.0,
        walking_reward_scale=0.5,
        config=None,
    ):
        """
        Args:
            walking_env: A WalkingEnv (or WalkingCurriculumEnv) instance.
            grid: 2D numpy array (1=wall, 0=open). Defaults to OPEN arena.
            cell_size: World-space cell size in meters.
            wall_height: Height of maze walls.
            goal_threshold: Distance to goal to consider it reached.
            max_episode_steps: Maximum steps per episode.
            max_goal_dist: Normalization constant for goal distance.
            max_wall_dist: Maximum raycast distance for wall sensing.
            walking_reward_scale: Scale factor for walking reward component.
            config: Additional configuration dict.
        """
        super().__init__(walking_env)
        self.cfg = config or {}
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.goal_threshold = goal_threshold
        self.max_episode_steps = max_episode_steps
        self.max_goal_dist = max_goal_dist
        self.max_wall_dist = max_wall_dist
        self.walking_reward_scale = walking_reward_scale

        # Maze grid
        if grid is not None:
            self.grid = grid
        elif HAS_MAZE:
            self.grid = OPEN.copy()
        else:
            self.grid = np.zeros((11, 11), dtype=np.int32)
            self.grid[0, :] = 1
            self.grid[-1, :] = 1
            self.grid[:, 0] = 1
            self.grid[:, -1] = 1

        # MJCF generator
        if HAS_MAZE:
            self.mjcf_gen = MazeMJCFGenerator(cell_size=cell_size, wall_height=wall_height)
        else:
            self.mjcf_gen = None

        # Goal tracking
        self.start_pos = np.zeros(2)
        self.goal_pos = np.zeros(2)
        self.prev_dist_to_goal = 0.0
        self.current_step = 0
        self._goal_reached = False

        # Override observation space to include nav features
        walking_obs_shape = walking_env.observation_space.shape[0]
        total_dim = walking_obs_shape + self.NAV_OBS_DIM
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

    def reset(self, **kwargs):
        """Reset environment with new start/goal positions."""
        obs, info = self.env.reset(**kwargs)

        self.current_step = 0
        self._goal_reached = False

        # Sample start and goal positions
        if self.mjcf_gen is not None:
            start_cell, goal_cell = self.mjcf_gen.sample_start_goal(
                self.grid, min_distance_cells=3, seed=None
            )
            sx, sy = self.mjcf_gen.grid_to_world(start_cell[0], start_cell[1], self.grid.shape)
            gx, gy = self.mjcf_gen.grid_to_world(goal_cell[0], goal_cell[1], self.grid.shape)
            self.start_pos = np.array([sx, sy])
            self.goal_pos = np.array([gx, gy])

            # Teleport humanoid to start position
            base_env = self.env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            if hasattr(base_env, 'unwrapped') and hasattr(base_env.unwrapped, 'data'):
                base_env = base_env.unwrapped
            if hasattr(base_env, 'data'):
                base_env.data.qpos[0] = sx
                base_env.data.qpos[1] = sy
        else:
            self.start_pos = np.zeros(2)
            self.goal_pos = np.array([5.0, 5.0])

        self.prev_dist_to_goal = np.linalg.norm(self._get_agent_pos() - self.goal_pos)

        # Augment observation with nav features
        nav_obs = self._compute_nav_obs()
        augmented_obs = np.concatenate([obs, nav_obs])

        return augmented_obs, info

    def step(self, action):
        """Step the environment and compute maze navigation rewards."""
        obs, walking_reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        # Compute navigation features
        nav_obs = self._compute_nav_obs()
        augmented_obs = np.concatenate([obs, nav_obs])

        # Compute navigation reward
        agent_pos = self._get_agent_pos()
        dist_to_goal = np.linalg.norm(agent_pos - self.goal_pos)

        nav_reward = self._compute_nav_reward(dist_to_goal)

        # Combined reward
        total_reward = self.walking_reward_scale * walking_reward + nav_reward

        # Check goal reached
        if dist_to_goal < self.goal_threshold:
            self._goal_reached = True
            total_reward += 50.0  # Goal bonus
            terminated = True

        # Time limit
        if self.current_step >= self.max_episode_steps:
            truncated = True

        self.prev_dist_to_goal = dist_to_goal

        # Add nav info
        info["nav_dist_to_goal"] = dist_to_goal
        info["nav_goal_reached"] = self._goal_reached
        info["nav_reward"] = nav_reward

        return augmented_obs, total_reward, terminated, truncated, info

    def _compute_nav_obs(self):
        """Compute 6-dim navigation observation in agent-local frame."""
        agent_pos = self._get_agent_pos()
        heading = self._get_agent_heading()

        # Goal vector in world frame
        goal_vec = self.goal_pos - agent_pos
        goal_dist = np.linalg.norm(goal_vec)

        # Rotate to agent-local frame
        cos_h, sin_h = np.cos(-heading), np.sin(-heading)
        goal_dx = cos_h * goal_vec[0] - sin_h * goal_vec[1]
        goal_dy = sin_h * goal_vec[0] + cos_h * goal_vec[1]

        # Normalized distance
        goal_dist_norm = min(goal_dist / self.max_goal_dist, 1.0)

        # Angle to goal relative to heading
        goal_angle = np.arctan2(goal_dy, goal_dx)

        # Wall raycasts (simplified — use grid-based approximation)
        wall_front = self._raycast_wall(agent_pos, heading)
        wall_left = self._raycast_wall(agent_pos, heading + np.pi / 2)

        return np.array([
            goal_dx / self.max_goal_dist,
            goal_dy / self.max_goal_dist,
            goal_dist_norm,
            goal_angle / np.pi,  # Normalize to [-1, 1]
            wall_front / self.max_wall_dist,
            wall_left / self.max_wall_dist,
        ], dtype=np.float32)

    def _compute_nav_reward(self, dist_to_goal):
        """Compute navigation reward components."""
        # Goal proximity (dense)
        proximity = 2.0 * np.exp(-1.0 * dist_to_goal)

        # Progress (dense, clipped)
        progress = self.prev_dist_to_goal - dist_to_goal
        progress_reward = 5.0 * np.clip(progress, -0.5, 0.5)

        # Time penalty
        time_penalty = -0.01

        # Wall collision penalty
        wall_penalty = 0.0
        base_env = self._get_base_env()
        if hasattr(base_env, 'data') and hasattr(base_env.data, 'ncon'):
            # Check contacts with maze walls
            for i in range(base_env.data.ncon):
                contact = base_env.data.contact[i]
                geom1_name = base_env.model.geom(contact.geom1).name if contact.geom1 < base_env.model.ngeom else ""
                geom2_name = base_env.model.geom(contact.geom2).name if contact.geom2 < base_env.model.ngeom else ""
                if "maze_wall" in geom1_name or "maze_wall" in geom2_name:
                    wall_penalty = -0.5
                    break

        return proximity + progress_reward + time_penalty + wall_penalty

    def _get_agent_pos(self):
        """Get agent (x, y) world position from qpos."""
        base_env = self._get_base_env()
        if hasattr(base_env, 'data'):
            return np.array([base_env.data.qpos[0], base_env.data.qpos[1]])
        return np.zeros(2)

    def _get_agent_heading(self):
        """Get agent heading from quaternion."""
        base_env = self._get_base_env()
        if hasattr(base_env, 'data'):
            quat = base_env.data.qpos[3:7]
            return np.arctan2(
                2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1 - 2 * (quat[2] ** 2 + quat[3] ** 2)
            )
        return 0.0

    def _get_base_env(self):
        """Unwrap to get the base MuJoCo environment."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        if hasattr(env, 'unwrapped'):
            return env.unwrapped
        return env

    def _raycast_wall(self, position, angle):
        """Simplified grid-based wall distance estimation.

        Walks along a ray in the grid until hitting a wall or max distance.
        """
        step_size = self.cell_size * 0.25
        rows, cols = self.grid.shape
        offset_x = -(cols * self.cell_size) / 2.0
        offset_y = -(rows * self.cell_size) / 2.0

        dx = np.cos(angle) * step_size
        dy = np.sin(angle) * step_size

        x, y = position[0], position[1]
        dist = 0.0

        for _ in range(int(self.max_wall_dist / step_size)):
            x += dx
            y += dy
            dist += step_size

            # Convert world to grid
            gc = int((x - offset_x) / self.cell_size)
            gr = rows - 1 - int((y - offset_y) / self.cell_size)

            if 0 <= gr < rows and 0 <= gc < cols:
                if self.grid[gr, gc] == 1:
                    return dist
            else:
                return dist  # Out of bounds

        return self.max_wall_dist

    @property
    def goal_reached(self):
        return self._goal_reached
