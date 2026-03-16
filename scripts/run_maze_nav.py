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


def ensure_framebuffer(mj_model, width, height):
    """Ensure the MuJoCo model's offscreen framebuffer is large enough."""
    if mj_model.vis.global_.offwidth < width:
        mj_model.vis.global_.offwidth = width
    if mj_model.vis.global_.offheight < height:
        mj_model.vis.global_.offheight = height


def render_topdown_map(mj_model, mj_data, grid, cell_size, goal_xy, waypoints, size=480):
    """Render a MuJoCo top-down camera view with overlaid path and goal markers.

    Uses the maze_topdown camera for a true physics-engine render, then overlays
    the planned path, goal marker, and agent marker using OpenCV.

    Returns:
        RGB numpy array of shape (size, size, 3).
    """
    import mujoco

    ensure_framebuffer(mj_model, size, size)
    renderer = mujoco.Renderer(mj_model, height=size, width=size)
    renderer.update_scene(mj_data, camera="maze_topdown")
    img = renderer.render().copy()
    renderer.close()

    # Compute world-to-pixel mapping for the top-down camera
    # Camera is at height cam_z with fovy=90, so visible half-extent = cam_z * tan(45°) = cam_z
    rows, cols = grid.shape
    cam_z = max(rows, cols) * cell_size * 1.2
    visible_half = cam_z  # fovy=90 → tan(45°) = 1.0

    def world_to_px(wx, wy):
        # Camera is centered at origin, looking straight down
        u = (wx + visible_half) / (2.0 * visible_half) * size
        v = (-wy + visible_half) / (2.0 * visible_half) * size
        return int(np.clip(u, 0, size - 1)), int(np.clip(v, 0, size - 1))

    # Draw path
    if waypoints and len(waypoints) >= 2:
        for i in range(len(waypoints) - 1):
            p1 = world_to_px(waypoints[i][0], waypoints[i][1])
            p2 = world_to_px(waypoints[i + 1][0], waypoints[i + 1][1])
            cv2.line(img, p1, p2, (50, 200, 255), 2, cv2.LINE_AA)

    # Draw goal
    gx, gy = world_to_px(goal_xy[0], goal_xy[1])
    cv2.circle(img, (gx, gy), 10, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (gx, gy), 10, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw agent position
    agent_x = float(mj_data.qpos[0])
    agent_y = float(mj_data.qpos[1])
    ax, ay = world_to_px(agent_x, agent_y)
    cv2.circle(img, (ax, ay), 8, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (ax, ay), 8, (255, 255, 255), 2, cv2.LINE_AA)

    # Labels
    cv2.putText(img, "TOP-DOWN", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return img


def render_third_person(mj_model, mj_data, width=640, height=480):
    """Render the third-person tracking camera view."""
    import mujoco

    ensure_framebuffer(mj_model, width, height)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    renderer.update_scene(mj_data, camera="track")
    img = renderer.render().copy()
    renderer.close()
    return img


def composite_frame(third_person, topdown_map):
    """Place third-person view on left, top-down map on right."""
    h = third_person.shape[0]
    # Resize map to match height
    map_resized = cv2.resize(topdown_map, (h, h), interpolation=cv2.INTER_LINEAR)
    return np.concatenate([third_person, map_resized], axis=1)


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

    # Auto-detect VecNormalize .pkl file next to model if not specified
    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        model_dir = Path(args.model).parent
        candidates = [
            model_dir / "vecnorm.pkl",
            model_dir / "vecnorm_walking.pkl",
            model_dir / (Path(args.model).stem + "_vecnorm.pkl"),
        ]
        # Also check for any .pkl file in the same directory
        for c in candidates:
            if c.exists():
                vecnorm_path = str(c)
                print(f"  Auto-detected VecNormalize: {vecnorm_path}")
                break
        if vecnorm_path is None:
            pkl_files = list(model_dir.glob("*.pkl"))
            if len(pkl_files) == 1:
                vecnorm_path = str(pkl_files[0])
                print(f"  Auto-detected VecNormalize: {vecnorm_path}")
            elif len(pkl_files) > 1:
                print(f"  WARNING: Multiple .pkl files found, specify --vecnorm: {pkl_files}")
        if vecnorm_path is None:
            print("  WARNING: No VecNormalize .pkl found — policy may receive unnormalized observations!")

    # Always use rgb_array when recording (we render manually via mujoco.Renderer)
    render_mode = "rgb_array" if (args.record or args.render) else None
    env = make_walking_env(render_mode=render_mode, config={
        "max_episode_steps": args.max_steps,
        "obs_history": 4,
        "obs_include_com": True,
        "obs_feature_norm": True,
        "xml_file": maze_xml_path,
        "random_height_init": False,
        "use_command_generator": False,
        "max_commanded_speed": 1.0,  # Allow full speed range for nav commands
        "push_enabled": False,
        "domain_rand": False,
        "action_smoothing": True,
        "action_smoothing_tau": 0.08,
    })
    vec_env = DummyVecEnv([lambda: env])

    if vecnorm_path:
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)
    nav = NavigationController(waypoints, target_speed=args.speed)

    # Get the unwrapped MuJoCo env
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    base_env = base_env.unwrapped

    import mujoco

    # Reset env, then teleport to maze start (keep default heading — policy
    # was trained facing +x and can't generalize to rotated quaternions).
    obs = vec_env.reset()
    base_env.data.qpos[0] = start_x
    base_env.data.qpos[1] = start_y
    base_env.data.qvel[:] = 0
    mujoco.mj_forward(base_env.model, base_env.data)

    # Update the walking env's internal state to match teleported position
    env.prev_height = float(base_env.data.qpos[2])
    env.low_height_steps = 0
    env.current_step = 0

    print(f"  Humanoid teleported to ({start_x:.2f}, {start_y:.2f}), height: {base_env.data.qpos[2]:.3f}")

    frames = []

    goal_world = (goal_x, goal_y)

    print(f"\n  {'Step':>5s}  {'Pos':>14s}  {'Hdg':>5s}  {'Cmd(vx,vy,yaw)':>20s}  {'ActVel(vx,vy)':>16s}  {'Height':>6s}  {'WP':>3s}")
    print(f"  {'-'*5}  {'-'*14}  {'-'*5}  {'-'*20}  {'-'*16}  {'-'*6}  {'-'*3}")

    for step in range(args.max_steps):
        # Get humanoid position and heading from underlying env
        base_env = env.unwrapped
        pos_x, pos_y = base_env.data.qpos[0], base_env.data.qpos[1]
        height = base_env.data.qpos[2]
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

        # Display command values periodically
        if step % 50 == 0:
            actual_vx = float(base_env.data.qvel[0])
            actual_vy = float(base_env.data.qvel[1])
            print(f"  {step:5d}  ({pos_x:6.2f},{pos_y:6.2f})  {np.degrees(heading):5.1f}  "
                  f"({cmd[0]:+5.2f},{cmd[1]:+5.2f},{cmd[2]:+5.2f})  "
                  f"({actual_vx:+6.3f},{actual_vy:+6.3f})  {height:6.3f}  "
                  f"{nav.current_waypoint_idx:3d}/{len(nav.waypoints)}")

        # Step the policy
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        if args.record:
            mj_model = base_env.model
            mj_data = base_env.data
            tp = render_third_person(mj_model, mj_data)
            td = render_topdown_map(mj_model, mj_data, grid, mjcf_gen.cell_size,
                                    goal_world, waypoints)
            frames.append(composite_frame(tp, td))

        if done[0]:
            actual_vx = float(base_env.data.qvel[0])
            actual_vy = float(base_env.data.qvel[1])
            print(f"  {step:5d}  ({pos_x:6.2f},{pos_y:6.2f})  {np.degrees(heading):5.1f}  "
                  f"({cmd[0]:+5.2f},{cmd[1]:+5.2f},{cmd[2]:+5.2f})  "
                  f"({actual_vx:+6.3f},{actual_vy:+6.3f})  {height:6.3f}  "
                  f"TERMINATED")
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
