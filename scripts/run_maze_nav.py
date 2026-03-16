#!/usr/bin/env python3
# run_maze_nav.py
"""
Demo script for maze navigation with a frozen walking policy.

Loads a maze, solves it with A*, then uses pure pursuit + frozen PPO
to navigate the humanoid through the maze.

Usage:
    python scripts/run_maze_nav.py --maze-type corridor --model models/walking/best/model.zip
    python scripts/run_maze_nav.py --maze-type dfs_3x3 --model models/walking/best/model.zip --record
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2


def render_minimap(grid, cell_size, agent_xy, goal_xy, waypoints, map_size=300):
    """Render a top-down minimap with agent position, goal, and path.

    Args:
        grid: 2D numpy array (1=wall, 0=open).
        cell_size: World-space cell size.
        agent_xy: (x, y) world position of agent.
        goal_xy: (x, y) world position of goal.
        waypoints: List of (x, y) world-coordinate waypoints.
        map_size: Pixel size of the square minimap.

    Returns:
        RGB numpy array of shape (map_size, map_size, 3).
    """
    rows, cols = grid.shape
    # Pixel size per cell
    px_per_cell = map_size / max(rows, cols)

    # Draw walls
    img = np.ones((map_size, map_size, 3), dtype=np.uint8) * 240  # light gray bg
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                y1 = int(r * px_per_cell)
                y2 = int((r + 1) * px_per_cell)
                x1 = int(c * px_per_cell)
                x2 = int((c + 1) * px_per_cell)
                cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 60), -1)

    # Helper: world coords to pixel coords on minimap
    offset_x = -(cols * cell_size) / 2.0
    offset_y = -(rows * cell_size) / 2.0

    def world_to_px(wx, wy):
        gc = (wx - offset_x) / cell_size
        gr = rows - 1 - (wy - offset_y) / cell_size
        px = int(gc * px_per_cell)
        py = int(gr * px_per_cell)
        return px, py

    # Draw path
    if waypoints and len(waypoints) >= 2:
        for i in range(len(waypoints) - 1):
            p1 = world_to_px(waypoints[i][0], waypoints[i][1])
            p2 = world_to_px(waypoints[i + 1][0], waypoints[i + 1][1])
            cv2.line(img, p1, p2, (200, 150, 50), 2)

    # Draw goal
    gx, gy = world_to_px(goal_xy[0], goal_xy[1])
    cv2.circle(img, (gx, gy), 8, (0, 0, 255), -1)
    cv2.putText(img, "G", (gx - 5, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw agent
    ax, ay = world_to_px(agent_xy[0], agent_xy[1])
    cv2.circle(img, (ax, ay), 6, (0, 200, 0), -1)

    # Border
    cv2.rectangle(img, (0, 0), (map_size - 1, map_size - 1), (100, 100, 100), 2)

    # Label
    cv2.putText(img, "MAP", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)

    return img


def composite_frame(mujoco_frame, minimap):
    """Place minimap on the right side of the MuJoCo frame.

    Returns:
        Composite RGB numpy array.
    """
    h, w = mujoco_frame.shape[:2]
    map_h = minimap.shape[0]

    # Resize minimap to match mujoco frame height
    scale = h / map_h
    new_map_w = int(minimap.shape[1] * scale)
    minimap_resized = cv2.resize(minimap, (new_map_w, h), interpolation=cv2.INTER_NEAREST)

    # Concatenate side by side
    return np.concatenate([mujoco_frame, minimap_resized], axis=1)


def get_maze_grid(maze_type, seed=42):
    """Get a maze grid by type name."""
    from src.maze.maze_generator import generate_maze_dfs, generate_maze_prims, open_arena, corridor
    from src.maze.maze_maps import CORRIDOR, L_MAZE, U_MAZE, OPEN, MEDIUM_MAZE

    maze_types = {
        "open": lambda: OPEN.copy(),
        "corridor": lambda: CORRIDOR.copy(),
        "l_maze": lambda: L_MAZE.copy(),
        "u_maze": lambda: U_MAZE.copy(),
        "medium": lambda: MEDIUM_MAZE.copy(),
        "open_arena": lambda: open_arena(3, 3),
        "corridor_gen": lambda: corridor(5, 1),
        "dfs_3x3": lambda: generate_maze_dfs(3, 3, seed=seed),
        "dfs_5x5": lambda: generate_maze_dfs(5, 5, seed=seed),
        "prims_3x3": lambda: generate_maze_prims(3, 3, seed=seed),
        "prims_5x5": lambda: generate_maze_prims(5, 5, seed=seed),
    }

    if maze_type not in maze_types:
        raise ValueError(f"Unknown maze type '{maze_type}'. Choose from: {list(maze_types.keys())}")

    return maze_types[maze_type]()


def main():
    parser = argparse.ArgumentParser(description="Run maze navigation demo")
    parser.add_argument("--maze-type", type=str, default="corridor", help="Maze type")
    parser.add_argument("--model", type=str, default=None, help="Path to walking model .zip")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to VecNormalize .pkl")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for maze generation")
    parser.add_argument("--render", action="store_true", help="Show visualization")
    parser.add_argument("--record", action="store_true", help="Record video")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max simulation steps")
    parser.add_argument("--speed", type=float, default=0.3, help="Target walking speed (m/s)")
    args = parser.parse_args()

    # Generate maze
    print(f"Generating maze: {args.maze_type}")
    grid = get_maze_grid(args.maze_type, seed=args.seed)
    print(f"  Grid shape: {grid.shape}")

    # Solve maze
    from src.maze.solver import solve
    from src.maze.maze_mjcf import MazeMJCFGenerator

    mjcf_gen = MazeMJCFGenerator(cell_size=2.0)
    start_cell, goal_cell = mjcf_gen.sample_start_goal(grid, seed=args.seed)
    print(f"  Start: {start_cell}, Goal: {goal_cell}")

    waypoints = solve(grid, start_cell, goal_cell, cell_size=2.0)
    if waypoints is None:
        print("ERROR: No path found!")
        return

    print(f"  Path: {len(waypoints)} waypoints")

    # Show schematic if no model provided
    if args.model is None:
        print("\nNo model provided — showing maze schematic only.")
        from src.maze.maze_renderer import MazeRenderer
        renderer = MazeRenderer(grid, cell_size=2.0)
        fig = renderer.render_schematic(
            agent_pos=start_cell,
            goal_pos=goal_cell,
        )
        if args.render:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            output = "data/maze_schematic.png"
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150)
            print(f"  Saved schematic to {output}")
            import matplotlib.pyplot as plt
            plt.close(fig)
        return

    # Full navigation loop with walking policy
    print(f"\nLoading walking model: {args.model}")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.environments import make_walking_env
    from src.maze.navigation_controller import NavigationController

    # Generate maze MJCF with physical walls
    print("  Generating maze MJCF with walls...")
    maze_xml_path = mjcf_gen.generate(grid)
    print(f"  Maze XML: {maze_xml_path}")

    # Convert start/goal from grid to world coordinates
    start_x, start_y = mjcf_gen.grid_to_world(start_cell[0], start_cell[1], grid.shape)
    goal_x, goal_y = mjcf_gen.grid_to_world(goal_cell[0], goal_cell[1], grid.shape)
    print(f"  Start world: ({start_x:.2f}, {start_y:.2f})")
    print(f"  Goal world:  ({goal_x:.2f}, {goal_y:.2f})")

    render_mode = "rgb_array" if args.record else ("human" if args.render else None)
    env = make_walking_env(render_mode=render_mode, config={
        "max_episode_steps": args.max_steps,
        "obs_history": 4,
        "obs_include_com": True,
        "obs_feature_norm": True,
        "xml_file": maze_xml_path,
        "random_height_init": False,
    })
    vec_env = DummyVecEnv([lambda: env])

    if args.vecnorm:
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)
    nav = NavigationController(waypoints, target_speed=args.speed)

    obs = vec_env.reset()

    # Teleport humanoid to maze start position
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    base_env = base_env.unwrapped
    base_env.data.qpos[0] = start_x
    base_env.data.qpos[1] = start_y
    print(f"  Humanoid teleported to ({start_x:.2f}, {start_y:.2f})")

    frames = []

    goal_world = (goal_x, goal_y)

    for step in range(args.max_steps):
        # Get humanoid position and heading from underlying env
        base_env = env.unwrapped
        pos_x, pos_y = base_env.data.qpos[0], base_env.data.qpos[1]
        # Approximate heading from quaternion
        quat = base_env.data.qpos[3:7]
        heading = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                             1 - 2 * (quat[2] ** 2 + quat[3] ** 2))

        # Get navigation command
        cmd = nav.get_command((pos_x, pos_y), heading)

        if nav.goal_reached:
            print(f"  Goal reached at step {step}!")
            break

        # Inject command into walking env
        env.fixed_command = (float(cmd[0]), float(cmd[1]), float(cmd[2]))

        # Step the policy
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        if args.record:
            frame = vec_env.render()
            if frame is not None:
                minimap = render_minimap(
                    grid, mjcf_gen.cell_size,
                    agent_xy=(pos_x, pos_y),
                    goal_xy=goal_world,
                    waypoints=waypoints,
                )
                frames.append(composite_frame(frame, minimap))

        if done[0]:
            print(f"  Episode terminated at step {step}")
            break

    if not nav.goal_reached:
        print(f"  Did not reach goal within {args.max_steps} steps")

    # Save video if recording
    if args.record and frames:
        output = "data/videos/maze_nav.mp4"
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"  Video saved: {output}")

    vec_env.close()
    print("Done.")


if __name__ == "__main__":
    main()
