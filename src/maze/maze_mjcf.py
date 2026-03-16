# maze_mjcf.py
"""
MJCF XML generation for MuJoCo maze environments.

Converts a 2D binary grid into physical MuJoCo walls by modifying
the base Humanoid-v5 XML with wall geoms, enlarged floor, and a
top-down camera.
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Locate the base humanoid XML from Gymnasium
try:
    import gymnasium
    _GYM_PATH = Path(gymnasium.__file__).parent / "envs" / "mujoco" / "assets" / "humanoid.xml"
except ImportError:
    _GYM_PATH = None


class MazeMJCFGenerator:
    """Generate MuJoCo MJCF XML files with maze wall geometry."""

    def __init__(self, cell_size=2.0, wall_height=0.5, wall_rgba=(0.6, 0.6, 0.6, 1.0)):
        """
        Args:
            cell_size: World-space size of each grid cell in meters.
            wall_height: Height of maze walls in meters.
            wall_rgba: RGBA color for wall geoms.
        """
        self.cell_size = cell_size
        self.wall_height = wall_height
        self.wall_rgba = wall_rgba

    def generate(self, grid, base_xml_path=None):
        """Convert a 2D grid to an MJCF XML string.

        Args:
            grid: 2D numpy array (1=wall, 0=open).
            base_xml_path: Path to base humanoid XML. Uses Gymnasium default if None.

        Returns:
            Path to a temporary XML file ready for gym.make(xml_file=...).
        """
        if base_xml_path is None:
            if _GYM_PATH is None or not _GYM_PATH.exists():
                raise FileNotFoundError("Cannot find base humanoid.xml. Install gymnasium[mujoco].")
            base_xml_path = _GYM_PATH

        tree = ET.parse(str(base_xml_path))
        root = tree.getroot()

        # Set offscreen framebuffer size for mujoco.Renderer
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        gl = visual.find("global")
        if gl is None:
            gl = ET.SubElement(visual, "global")
        gl.set("offwidth", "1280")
        gl.set("offheight", "960")

        rows, cols = grid.shape
        half_size = self.cell_size / 2.0

        # Center the grid at the world origin
        offset_x = -(cols * self.cell_size) / 2.0
        offset_y = -(rows * self.cell_size) / 2.0

        # Enlarge the floor plane
        worldbody = root.find("worldbody")
        floor = worldbody.find(".//geom[@type='plane']")
        if floor is None:
            floor = worldbody.find(".//geom[@name='floor']")
        if floor is not None:
            floor_extent = max(rows, cols) * self.cell_size
            floor.set("size", f"{floor_extent} {floor_extent} 0.1")

        # Add maze walls
        wall_half_h = self.wall_height / 2.0
        rgba_str = " ".join(f"{c:.2f}" for c in self.wall_rgba)

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:
                    x = offset_x + c * self.cell_size + half_size
                    y = offset_y + (rows - 1 - r) * self.cell_size + half_size  # flip y
                    z = wall_half_h

                    wall = ET.SubElement(worldbody, "geom")
                    wall.set("type", "box")
                    wall.set("pos", f"{x:.3f} {y:.3f} {z:.3f}")
                    wall.set("size", f"{half_size:.3f} {half_size:.3f} {wall_half_h:.3f}")
                    wall.set("rgba", rgba_str)
                    wall.set("contype", "1")
                    wall.set("conaffinity", "1")
                    wall.set("condim", "3")
                    wall.set("friction", "1.0 0.005 0.0001")
                    wall.set("name", f"maze_wall_{r}_{c}")

        # Add top-down fixed camera
        cam_z = max(rows, cols) * self.cell_size * 1.2
        cam = ET.SubElement(worldbody, "camera")
        cam.set("name", "maze_topdown")
        cam.set("mode", "fixed")
        cam.set("pos", f"0 0 {cam_z:.1f}")
        cam.set("euler", "0 0 0")
        cam.set("fovy", "90")

        # Override the default tracking camera: GTA-style third-person
        torso = worldbody.find(".//body[@name='torso']")
        if torso is not None:
            # Remove existing track camera
            for existing_cam in torso.findall("camera[@name='track']"):
                torso.remove(existing_cam)
            # GTA-style: directly behind and above, looking down over shoulder
            track_cam = ET.SubElement(torso, "camera")
            track_cam.set("name", "track")
            track_cam.set("mode", "track")
            track_cam.set("pos", "0 -3.0 2.5")
            track_cam.set("xyaxes", "1 0 0 0 0.4 1")

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
        tree.write(tmp.name, xml_declaration=True)
        return tmp.name

    def grid_to_world(self, grid_row, grid_col, grid_shape):
        """Convert grid cell indices to world coordinates.

        Args:
            grid_row: Row index in the grid.
            grid_col: Column index in the grid.
            grid_shape: (rows, cols) shape of the grid.

        Returns:
            (x, y) world coordinates.
        """
        rows, cols = grid_shape
        offset_x = -(cols * self.cell_size) / 2.0
        offset_y = -(rows * self.cell_size) / 2.0

        x = offset_x + grid_col * self.cell_size + self.cell_size / 2.0
        y = offset_y + (rows - 1 - grid_row) * self.cell_size + self.cell_size / 2.0
        return x, y

    def sample_start_goal(self, grid, min_distance_cells=3, seed=None):
        """Sample well-separated start and goal positions from free cells.

        Args:
            grid: 2D binary grid.
            min_distance_cells: Minimum Manhattan distance between start and goal.
            seed: Random seed.

        Returns:
            ((start_row, start_col), (goal_row, goal_col)) grid indices.
        """
        rng = np.random.RandomState(seed)
        free_cells = list(zip(*np.where(grid == 0)))

        if len(free_cells) < 2:
            raise ValueError("Grid has fewer than 2 free cells.")

        for _ in range(1000):
            idx_s, idx_g = rng.choice(len(free_cells), size=2, replace=False)
            sr, sc = free_cells[idx_s]
            gr, gc = free_cells[idx_g]
            dist = abs(sr - gr) + abs(sc - gc)
            if dist >= min_distance_cells:
                return (sr, sc), (gr, gc)

        # Fallback: pick the two most distant free cells
        idx_s = rng.randint(len(free_cells))
        sr, sc = free_cells[idx_s]
        best_dist = 0
        best_goal = free_cells[0]
        for gr, gc in free_cells:
            d = abs(sr - gr) + abs(sc - gc)
            if d > best_dist:
                best_dist = d
                best_goal = (gr, gc)

        return (sr, sc), best_goal
