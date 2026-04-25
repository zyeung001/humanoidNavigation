# nav_rebuild_env.py
"""
Phase 0 navigation rebuild environment.

End-to-end goal-conditioned navigation: maze geometry, A* path computed at
reset, body-frame waypoint observations, progress-along-path reward.

Replaces the walking_policy + nav_controller decomposition. Single policy
takes goal in, emits torques out. No velocity-command interface.

See NAVIGATION_REBUILD_PLAN.md (origin/plan/navigation-rebuild) for the
full motivation and phase plan.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from src.maze.maze_maps import CORRIDOR
from src.maze.maze_mjcf import MazeMJCFGenerator
from src.maze.solver import solve as astar_solve


class NavRebuildEnv(gym.Wrapper):
    """
    End-to-end navigation environment for goal-conditioned humanoid locomotion.

    Reset:
      1. Sample (start, goal) cells from open grid cells (min separation).
      2. Run A* on the grid → world-coord polyline path.
      3. Teleport humanoid to start cell, reset state.

    Observation per step:
      [ humanoid_obs (~365) , COM features (6) , waypoint_block (6) ]

    Waypoint block: body-frame (dx, dy) to the next 3 waypoints along the
    path. If fewer than 3 remain, repeat the goal.

    Reward:
      + progress along path arc-length      (forward motion toward goal)
      + waypoint-reached bonus (monotonic)  (sparse, only first crossing)
      + goal-reached bonus (terminal)
      - time penalty (small, every step)
      - collision penalty (terminal)        (any contact with maze wall)
      - fall penalty (terminal)             (height too low or torso flipped)

    Forbidden / not present (per plan):
      - survival reward
      - Euclidean progress
      - mid-episode A* replan
      - velocity-command interface
    """

    WAYPOINT_LOOKAHEAD = 3
    WAYPOINT_OBS_DIM = 2 * WAYPOINT_LOOKAHEAD  # (dx, dy) per waypoint

    def __init__(
        self,
        grid: Optional[np.ndarray] = None,
        cell_size: float = 2.0,
        wall_height: float = 2.5,
        wall_thickness: float = 0.15,
        goal_threshold: float = 0.5,
        waypoint_threshold: float = 0.6,
        max_episode_steps: int = 2000,
        min_start_goal_cells: int = 3,
        # Open-arena mode (Phase 1): no walls, random goal point.
        open_arena: bool = False,
        open_arena_size: float = 30.0,
        open_arena_goal_dist: float = 5.0,
        # Observation structure — mirrors walking_env so warm-start surgery
        # can transfer body weights at columns [0:1424] directly.
        history_len: int = 4,
        # Reward weights — initial guesses, tuned in Phase 0 verification.
        progress_weight: float = 5.0,
        waypoint_bonus: float = 5.0,
        goal_bonus: float = 50.0,
        time_penalty: float = 0.005,
        collision_penalty: float = 25.0,
        fall_penalty: float = 25.0,
        # Termination thresholds (mirrors walking_env defaults).
        fall_height_threshold: float = 0.7,
        config: Optional[dict] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        cfg = config or {}
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.goal_threshold = goal_threshold
        self.waypoint_threshold = waypoint_threshold
        self.max_episode_steps = max_episode_steps
        self.min_start_goal_cells = min_start_goal_cells
        self.history_len = max(1, int(history_len))

        self.open_arena = bool(open_arena)
        self.open_arena_size = float(open_arena_size)
        self.open_arena_goal_dist = float(open_arena_goal_dist)
        if self.open_arena:
            # No walls; we still pass a tiny stub grid into the maze generator
            # so the floor is enlarged. The grid here is unused for path
            # planning — paths are straight lines start→goal.
            self.grid = np.array([[0]], dtype=np.int32)
        else:
            self.grid = grid.copy() if grid is not None else CORRIDOR.copy()

        self.progress_weight = progress_weight
        self.waypoint_bonus = waypoint_bonus
        self.goal_bonus = goal_bonus
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty
        self.fall_penalty = fall_penalty
        self.fall_height_threshold = fall_height_threshold

        self.cfg = cfg
        self._rng = np.random.RandomState(seed)

        # Base humanoid env. In open-arena mode skip maze MJCF and use the
        # default humanoid XML directly.
        if self.open_arena:
            self.mjcf_gen = None
            self.maze_xml_path = None
            env = gym.make(
                "Humanoid-v5",
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
            )
        else:
            self.mjcf_gen = MazeMJCFGenerator(
                cell_size=cell_size,
                wall_height=wall_height,
                wall_thickness=wall_thickness,
            )
            self.maze_xml_path = self.mjcf_gen.generate(self.grid)
            env = gym.make(
                "Humanoid-v5",
                xml_file=self.maze_xml_path,
                render_mode=render_mode,
                exclude_current_positions_from_observation=False,
            )
        super().__init__(env)

        # Cache wall geom IDs for fast collision lookup (empty in open arena).
        self._wall_geom_ids = self._collect_wall_geom_ids()

        # Episode state
        self.path: list = []                  # list of (x, y) waypoints (start..goal)
        self.cumulative_arc: np.ndarray = np.zeros(0)
        self.path_total_length: float = 0.0
        self.current_segment_idx: int = 0
        self.next_waypoint_idx: int = 1
        self.prev_progress: float = 0.0
        self.start_pos = np.zeros(2)
        self.goal_pos = np.zeros(2)
        self.current_step = 0
        self._termination_cause: Optional[str] = None

        # Per-frame body dim = humanoid raw obs + 6 COM features. Mirrors
        # walking_env so warm-start can keep the same column layout.
        probe_obs, _ = self.env.reset()
        self._humanoid_obs_dim = int(np.asarray(probe_obs).shape[0])
        self._com_dim = 6
        self._body_dim_per_frame = self._humanoid_obs_dim + self._com_dim

        self._stacked_body_dim = self._body_dim_per_frame * self.history_len
        total_dim = self._stacked_body_dim + self.WAYPOINT_OBS_DIM
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )
        self._frozen_obs_dim = total_dim
        self._obs_history: list = []

    # ------------------------------------------------------------------
    # Public API: gym.Env
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        obs, info = self.env.reset(seed=seed)

        self.current_step = 0
        self._termination_cause = None
        self._obs_history = []

        # Sample start/goal and build the path.
        if self.open_arena:
            self.path = self._sample_open_arena_path()
        else:
            self.path = self._sample_start_goal_path()

        sx, sy = self.path[0]
        gx, gy = self.path[-1]
        self.start_pos = np.array([sx, sy])
        self.goal_pos = np.array([gx, gy])

        # Teleport humanoid to start. qpos[0:2] are world (x, y) when
        # exclude_current_positions_from_observation=False.
        base = self.env.unwrapped
        base.data.qpos[0] = sx
        base.data.qpos[1] = sy
        # Random initial heading so policy learns to handle any start orientation.
        if self.open_arena:
            yaw = float(self._rng.uniform(-np.pi, np.pi))
            half = yaw * 0.5
            base.data.qpos[3] = np.cos(half)
            base.data.qpos[4] = 0.0
            base.data.qpos[5] = 0.0
            base.data.qpos[6] = np.sin(half)
        # Re-fetch obs after teleport so initial obs reflects new pose.
        obs = base._get_obs()

        # Path arc-length bookkeeping.
        seg_lengths = [
            np.hypot(self.path[i + 1][0] - self.path[i][0],
                     self.path[i + 1][1] - self.path[i][1])
            for i in range(len(self.path) - 1)
        ]
        self.cumulative_arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.path_total_length = float(self.cumulative_arc[-1])
        self.current_segment_idx = 0
        self.next_waypoint_idx = 1
        # Initial progress = projection of start pos onto path = 0.
        self.prev_progress = 0.0

        info["nav_path"] = list(self.path)
        info["nav_path_length"] = self.path_total_length
        info["nav_start"] = (sx, sy)
        info["nav_goal"] = (gx, gy)

        augmented = self._augment_obs(obs)
        return augmented.astype(np.float32), info

    def step(self, action):
        obs, _base_reward, _base_term, _base_trunc, info = self.env.step(action)
        self.current_step += 1

        agent_pos = self._agent_xy()
        height = float(self.env.unwrapped.data.qpos[2])

        # ---------- progress along path arc-length ----------
        new_progress, new_segment_idx = self._project_progress(agent_pos)
        progress_delta = new_progress - self.prev_progress
        # Clip to a sane per-step bound to suppress jumps from teleporting
        # near the next segment (rare; keeps gradient bounded).
        progress_delta = float(np.clip(progress_delta, -0.5, 0.5))
        progress_reward = self.progress_weight * progress_delta
        self.prev_progress = new_progress
        self.current_segment_idx = max(self.current_segment_idx, new_segment_idx)

        # ---------- waypoint-reached bonus (monotonic) ----------
        wp_bonus = 0.0
        while (self.next_waypoint_idx < len(self.path) - 1
               and self._dist_to_waypoint(agent_pos, self.next_waypoint_idx)
               < self.waypoint_threshold):
            wp_bonus += self.waypoint_bonus
            self.next_waypoint_idx += 1

        # ---------- time penalty ----------
        time_pen = -self.time_penalty

        # ---------- terminations ----------
        terminated = False
        terminal_bonus = 0.0
        terminal_penalty = 0.0

        # Goal check
        dist_to_goal = float(np.linalg.norm(agent_pos - self.goal_pos))
        if dist_to_goal < self.goal_threshold:
            terminated = True
            self._termination_cause = "goal"
            terminal_bonus = self.goal_bonus

        # Collision check (only if not already terminating for goal)
        if not terminated and self._wall_contact_active():
            terminated = True
            self._termination_cause = "collision"
            terminal_penalty = -self.collision_penalty

        # Fall check
        if not terminated:
            quat = self.env.unwrapped.data.qpos[3:7]
            up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
            if height < self.fall_height_threshold or up_z < 0.0:
                terminated = True
                self._termination_cause = "fall"
                terminal_penalty = -self.fall_penalty

        truncated = self.current_step >= self.max_episode_steps and not terminated

        total_reward = (
            progress_reward
            + wp_bonus
            + time_pen
            + terminal_bonus
            + terminal_penalty
        )

        # Diagnostics for Phase 0 verification.
        info.update({
            "nav_progress_arc": new_progress,
            "nav_progress_delta": progress_delta,
            "nav_dist_to_goal": dist_to_goal,
            "nav_next_waypoint": self.next_waypoint_idx,
            "nav_height": height,
            "reward/progress": progress_reward,
            "reward/waypoint": wp_bonus,
            "reward/time": time_pen,
            "reward/terminal_bonus": terminal_bonus,
            "reward/terminal_penalty": terminal_penalty,
            "reward/total": total_reward,
        })
        if self._termination_cause is not None:
            info["termination_cause"] = self._termination_cause

        augmented = self._augment_obs(obs)
        return augmented.astype(np.float32), total_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_open_arena_path(self) -> list:
        """Open-arena path: agent at origin, random goal at fixed distance.

        Phase 1 setup per the rebuild plan: no walls, single random goal
        ~5m away. Path is the straight line [start, goal].
        """
        start = (0.0, 0.0)
        angle = float(self._rng.uniform(-np.pi, np.pi))
        d = self.open_arena_goal_dist
        goal = (d * np.cos(angle), d * np.sin(angle))
        return [start, goal]

    def _sample_start_goal_path(self) -> list:
        """Sample (start, goal) free cells with an A* path between them."""
        free_cells = list(zip(*np.where(self.grid == 0)))
        if len(free_cells) < 2:
            raise ValueError("Maze grid has fewer than 2 free cells.")

        for _ in range(200):
            si, gi = self._rng.choice(len(free_cells), size=2, replace=False)
            si, gi = int(si), int(gi)
            s = free_cells[si]
            g = free_cells[gi]
            if abs(s[0] - g[0]) + abs(s[1] - g[1]) < self.min_start_goal_cells:
                continue
            path = astar_solve(self.grid, s, g, cell_size=self.cell_size)
            if path is not None and len(path) >= 2:
                return path

        # Fallback: pick the two cells with greatest grid distance.
        best = (free_cells[0], free_cells[-1])
        best_d = 0
        for i, a in enumerate(free_cells):
            for b in free_cells[i + 1:]:
                d = abs(a[0] - b[0]) + abs(a[1] - b[1])
                if d > best_d:
                    best_d = d
                    best = (a, b)
        path = astar_solve(self.grid, best[0], best[1], cell_size=self.cell_size)
        if path is None:
            raise RuntimeError("No A* path exists in this grid.")
        return path

    def _project_progress(self, pos: np.ndarray) -> tuple[float, int]:
        """Project agent position onto the path; return (arc_length, segment_idx).

        Searches segments from current_segment_idx onward to prevent regress
        through earlier segments (e.g., looping back). Within each segment,
        clamps the projection parameter to [0, 1].
        """
        best_arc = self.cumulative_arc[self.current_segment_idx]
        best_seg = self.current_segment_idx
        best_d = np.inf

        for i in range(self.current_segment_idx, len(self.path) - 1):
            ax, ay = self.path[i]
            bx, by = self.path[i + 1]
            ab = np.array([bx - ax, by - ay])
            ab_len = np.linalg.norm(ab)
            if ab_len < 1e-9:
                continue
            ap = pos - np.array([ax, ay])
            t = float(np.clip(np.dot(ap, ab) / (ab_len ** 2), 0.0, 1.0))
            proj = np.array([ax + t * (bx - ax), ay + t * (by - ay)])
            d = float(np.linalg.norm(pos - proj))
            arc = self.cumulative_arc[i] + t * ab_len
            if d < best_d:
                best_d = d
                best_arc = arc
                best_seg = i
            # Early exit: if we just past midpoint of segment, later segments
            # likely farther; but path geometry can fool this — be safe and
            # keep scanning all forward segments.

        return best_arc, best_seg

    def _dist_to_waypoint(self, pos: np.ndarray, idx: int) -> float:
        wx, wy = self.path[idx]
        return float(np.hypot(pos[0] - wx, pos[1] - wy))

    def _agent_xy(self) -> np.ndarray:
        d = self.env.unwrapped.data
        return np.array([d.qpos[0], d.qpos[1]])

    def _agent_heading(self) -> float:
        quat = self.env.unwrapped.data.qpos[3:7]
        return float(np.arctan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        ))

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Build [stacked_body (history×(humanoid+COM)), waypoint_block (6)].

        Body columns [0:stacked_body_dim] match walking_env's layout exactly
        (raw humanoid obs + COM features per frame, history-stacked oldest→newest)
        so the warm-start adapter can keep walking weights at those columns.
        """
        base = np.asarray(base_obs, dtype=np.float32).ravel()

        # Per-frame body feature: raw humanoid obs + COM (pos, vel) = 356 dims
        # for Humanoid-v5.
        d = self.env.unwrapped.data
        try:
            com_pos = np.asarray(d.subtree_com[0], dtype=np.float32)
        except (AttributeError, IndexError):
            com_pos = np.zeros(3, dtype=np.float32)
        com_vel = np.asarray(d.qvel[:3], dtype=np.float32)
        body_frame = np.concatenate([base, com_pos, com_vel]).astype(np.float32)

        # Pad/truncate frame to match probed body_dim_per_frame (defensive).
        if body_frame.shape[0] != self._body_dim_per_frame:
            if body_frame.shape[0] < self._body_dim_per_frame:
                body_frame = np.concatenate([
                    body_frame,
                    np.zeros(self._body_dim_per_frame - body_frame.shape[0],
                             dtype=np.float32),
                ])
            else:
                body_frame = body_frame[:self._body_dim_per_frame]

        # History stack: keep last history_len frames, pad oldest with zeros
        # if the buffer is not full yet.
        self._obs_history.append(body_frame)
        if len(self._obs_history) > self.history_len:
            self._obs_history = self._obs_history[-self.history_len:]
        if len(self._obs_history) < self.history_len:
            pad = [np.zeros_like(body_frame)
                   for _ in range(self.history_len - len(self._obs_history))]
            frames = pad + self._obs_history
        else:
            frames = self._obs_history
        stacked_body = np.concatenate(frames, axis=0).astype(np.float32)

        # Waypoint block (body-frame dx, dy to next K waypoints).
        wp_block = self._waypoint_block()

        feat = np.concatenate([stacked_body, wp_block]).astype(np.float32)
        if feat.shape[0] != self._frozen_obs_dim:
            if feat.shape[0] < self._frozen_obs_dim:
                feat = np.concatenate([
                    feat,
                    np.zeros(self._frozen_obs_dim - feat.shape[0], dtype=np.float32),
                ])
            else:
                feat = feat[:self._frozen_obs_dim]
        return feat

    def _waypoint_block(self) -> np.ndarray:
        """Body-frame (dx, dy) to next K waypoints. Repeats goal if path runs out."""
        agent_pos = self._agent_xy()
        heading = self._agent_heading()
        cos_h = np.cos(-heading)
        sin_h = np.sin(-heading)

        out = np.zeros(self.WAYPOINT_OBS_DIM, dtype=np.float32)
        for k in range(self.WAYPOINT_LOOKAHEAD):
            idx = min(self.next_waypoint_idx + k, len(self.path) - 1)
            wx, wy = self.path[idx]
            dx_w = wx - agent_pos[0]
            dy_w = wy - agent_pos[1]
            # Rotate world→body (rotation by -heading).
            dx_b = cos_h * dx_w - sin_h * dy_w
            dy_b = sin_h * dx_w + cos_h * dy_w
            out[2 * k] = dx_b
            out[2 * k + 1] = dy_b
        return out

    def _collect_wall_geom_ids(self) -> set:
        """Return the set of geom IDs whose name starts with 'maze_wall'."""
        model = self.env.unwrapped.model
        ids = set()
        for gid in range(model.ngeom):
            name = model.geom(gid).name or ""
            if name.startswith("maze_wall"):
                ids.add(gid)
        return ids

    def _wall_contact_active(self) -> bool:
        """True if any active contact involves a maze wall geom."""
        if not self._wall_geom_ids:
            return False
        data = self.env.unwrapped.data
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 in self._wall_geom_ids or c.geom2 in self._wall_geom_ids:
                return True
        return False
