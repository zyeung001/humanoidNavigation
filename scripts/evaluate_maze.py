#!/usr/bin/env python3
# evaluate_maze.py
"""
Evaluate a trained maze navigation model.

Usage:
    python scripts/evaluate_maze.py --model models/maze/final/model.zip \
        --maze-type dfs_3x3 --episodes 10

    python scripts/evaluate_maze.py --model models/maze/final/model.zip \
        --maze-type l_maze --render --record
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    parser = argparse.ArgumentParser(description="Evaluate maze navigation model")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to VecNormalize .pkl")
    parser.add_argument("--maze-type", type=str, default="open", help="Maze type for evaluation")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Show visualization")
    parser.add_argument("--record", action="store_true", help="Record video")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Get maze grid
    from scripts.run_maze_nav import get_maze_grid
    grid = get_maze_grid(args.maze_type, seed=args.seed)
    print(f"Maze: {args.maze_type}, grid shape: {grid.shape}")

    # Create environment
    from src.environments import make_walking_env
    from src.environments.maze_env import MazeNavigationEnv
    from stable_baselines3.common.monitor import Monitor

    render_mode = "rgb_array" if args.record else ("human" if args.render else None)
    walking_env = make_walking_env(render_mode=render_mode, config={
        'max_episode_steps': args.max_steps,
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.08,
    })

    maze_env = MazeNavigationEnv(walking_env, grid=grid, max_episode_steps=args.max_steps)
    vec_env = DummyVecEnv([lambda: Monitor(maze_env)])

    if args.vecnorm and os.path.exists(args.vecnorm):
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model, env=vec_env)

    print(f"\n{'='*60}")
    print("MAZE NAVIGATION EVALUATION")
    print(f"{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Maze: {args.maze_type}")
    print(f"  Episodes: {args.episodes}")
    print(f"{'='*60}\n")

    # Run evaluation
    results = []
    frames = []

    for ep in range(args.episodes):
        obs = vec_env.reset()
        ep_reward = 0
        ep_steps = 0
        goal_reached = False

        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]
            ep_steps += 1

            if args.record:
                frame = vec_env.render()
                if frame is not None:
                    frames.append(frame)

            if done[0]:
                goal_reached = info[0].get("nav_goal_reached", False)
                break

        status = "GOAL" if goal_reached else "FAIL"
        dist = info[0].get("nav_dist_to_goal", -1)
        print(f"  Episode {ep+1:2d}: {status} | steps={ep_steps:5d} | "
              f"reward={ep_reward:7.1f} | dist={dist:.2f}")

        results.append({
            "episode": ep,
            "goal_reached": goal_reached,
            "steps": ep_steps,
            "reward": ep_reward,
            "final_dist": dist,
        })

    # Summary
    n_success = sum(1 for r in results if r["goal_reached"])
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Success rate: {n_success}/{args.episodes} ({100*n_success/args.episodes:.0f}%)")
    print(f"  Avg reward: {np.mean([r['reward'] for r in results]):.1f}")
    print(f"  Avg steps: {np.mean([r['steps'] for r in results]):.0f}")
    print(f"  Avg final dist: {np.mean([r['final_dist'] for r in results]):.2f}")

    # Save video
    if args.record and frames:
        try:
            import cv2
            output = f"data/videos/maze_{args.maze_type}_eval.mp4"
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"\n  Video saved: {output}")
        except ImportError:
            print("\n  OpenCV not available, skipping video save")

    vec_env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
