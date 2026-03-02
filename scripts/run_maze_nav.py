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

    render_mode = "rgb_array" if args.record else ("human" if args.render else None)
    env = make_walking_env(render_mode=render_mode, config={"max_episode_steps": args.max_steps})
    vec_env = DummyVecEnv([lambda: env])

    if args.vecnorm:
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)
    nav = NavigationController(waypoints, target_speed=args.speed)

    obs = vec_env.reset()
    frames = []

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
                frames.append(frame)

        if done[0]:
            print(f"  Episode terminated at step {step}")
            break

    if not nav.goal_reached:
        print(f"  Did not reach goal within {args.max_steps} steps")

    # Save video if recording
    if args.record and frames:
        try:
            import cv2
            output = "data/videos/maze_nav.mp4"
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"  Video saved: {output}")
        except ImportError:
            print("  OpenCV not available, skipping video save")

    vec_env.close()
    print("Done.")


if __name__ == "__main__":
    main()
