# maze_renderer.py
"""
Top-down maze rendering utilities.

Three rendering modes:
- MuJoCo camera: Realistic view via mujoco.Renderer + maze_topdown camera
- Matplotlib schematic: 2D grid plot with agent/goal/path overlay
- Occupancy grid: Binary numpy array at configurable resolution
"""

import numpy as np


class MazeRenderer:
    """Render maze visualizations in multiple modes."""

    def __init__(self, grid, cell_size=2.0):
        """
        Args:
            grid: 2D numpy array (1=wall, 0=open).
            cell_size: World-space size of each grid cell.
        """
        self.grid = grid
        self.cell_size = cell_size

    def render_mujoco_topdown(self, mj_model, mj_data, width=640, height=640):
        """Render bird's-eye view using MuJoCo's built-in renderer.

        Args:
            mj_model: MuJoCo model.
            mj_data: MuJoCo data.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            RGB numpy array of shape (height, width, 3).
        """
        import mujoco

        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        renderer.update_scene(mj_data, camera="maze_topdown")
        img = renderer.render()
        renderer.close()
        return img

    def render_schematic(self, agent_pos=None, goal_pos=None, path=None, figsize=(8, 8)):
        """Render a 2D matplotlib schematic of the maze.

        Args:
            agent_pos: (row, col) grid position of the agent, or None.
            goal_pos: (row, col) grid position of the goal, or None.
            path: List of (row, col) grid positions for path overlay, or None.
            figsize: Matplotlib figure size.

        Returns:
            Matplotlib figure object.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(self.grid, cmap="binary", origin="upper", interpolation="nearest")

        if path is not None:
            path_arr = np.array(path)
            ax.plot(path_arr[:, 1], path_arr[:, 0], "b-", linewidth=2, alpha=0.7, label="Path")

        if agent_pos is not None:
            ax.plot(agent_pos[1], agent_pos[0], "go", markersize=12, label="Agent")

        if goal_pos is not None:
            ax.plot(goal_pos[1], goal_pos[0], "r*", markersize=15, label="Goal")

        ax.set_title("Maze Layout")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        if agent_pos is not None or goal_pos is not None or path is not None:
            ax.legend(loc="upper right")
        ax.set_aspect("equal")
        plt.tight_layout()
        return fig

    def render_occupancy(self, resolution=0.1):
        """Generate a high-resolution binary occupancy grid.

        Useful for path planning and sim-to-real transfer.

        Args:
            resolution: Size of each pixel in meters.

        Returns:
            2D numpy array where 1=occupied, 0=free.
        """
        rows, cols = self.grid.shape
        world_h = rows * self.cell_size
        world_w = cols * self.cell_size
        occ_h = int(world_h / resolution)
        occ_w = int(world_w / resolution)

        occupancy = np.zeros((occ_h, occ_w), dtype=np.int32)

        for r in range(rows):
            for c in range(cols):
                if self.grid[r, c] == 1:
                    # Map grid cell to occupancy pixels
                    pr_start = int(r * self.cell_size / resolution)
                    pr_end = int((r + 1) * self.cell_size / resolution)
                    pc_start = int(c * self.cell_size / resolution)
                    pc_end = int((c + 1) * self.cell_size / resolution)
                    occupancy[pr_start:pr_end, pc_start:pc_end] = 1

        return occupancy
