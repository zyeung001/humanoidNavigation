#!/usr/bin/env python3
# train_maze.py
"""
Training script for humanoid maze navigation.

Two training modes:
1. Hierarchical (recommended): Freeze walking policy, train small navigation policy
2. End-to-end: Single policy controlling both locomotion and navigation

Usage:
    # Hierarchical (default)
    python scripts/train_maze.py --walking-model models/walking/best/model.zip \
        --walking-vecnorm models/walking/best/vecnorm.pkl --timesteps 50000000

    # End-to-end
    python scripts/train_maze.py --mode end-to-end \
        --walking-model models/walking/best/model.zip --timesteps 50000000
"""

import os
import sys
import warnings
import argparse

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.utils import configure_mujoco_gl, get_subprocess_start_method
configure_mujoco_gl()

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.environments.maze_env import MazeNavigationEnv
from src.environments.maze_curriculum import MazeCurriculum
from src.training.model_manager import ModelManager


def load_config(path=None):
    """Load configuration, merging maze defaults with file config."""
    defaults = {
        "cell_size": 2.0,
        "wall_height": 2.5,
        "goal_threshold": 0.5,
        "max_episode_steps": 10000,
        "hierarchical": True,
        "total_timesteps": 50_000_000,
        "n_envs": 8,
        "learning_rate": 0.0003,
        "final_learning_rate": 0.00005,
        "batch_size": 256,
        "n_steps": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "seed": 42,
    }

    if path and os.path.exists(path):
        with open(path, 'r') as f:
            full_config = yaml.safe_load(f)
        if 'maze' in full_config:
            defaults.update(full_config['maze'])

    return defaults


def make_maze_env(walking_model_path, walking_vecnorm_path, config, rank=0, render_mode=None):
    """Create a single maze navigation environment."""
    from src.environments import make_walking_env

    walking_cfg = {
        'max_episode_steps': config.get('max_episode_steps', 10000),
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.08,
    }

    walking_env = make_walking_env(render_mode=render_mode, config=walking_cfg)

    maze_env = MazeNavigationEnv(
        walking_env,
        cell_size=config.get('cell_size', 2.0),
        wall_height=config.get('wall_height', 2.5),
        goal_threshold=config.get('goal_threshold', 0.5),
        max_episode_steps=config.get('max_episode_steps', 10000),
        config=config,
    )

    return Monitor(maze_env)


class MazeCurriculumCallback(BaseCallback):
    """Callback that updates maze curriculum based on episode results."""

    def __init__(self, curriculum, verbose=0):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.episode_count = 0

    def _on_step(self):
        # Check for completed episodes in infos
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_count += 1
                    goal_reached = info.get("nav_goal_reached", False)
                    ep_length = info["episode"]["l"]

                    advanced = self.curriculum.record_episode(goal_reached, ep_length)

                    if self.verbose > 0 and self.episode_count % 50 == 0:
                        cur_info = self.curriculum.get_info()
                        print(f"  [Maze] Episode {self.episode_count}: "
                              f"Stage {cur_info['maze_stage']} ({cur_info['maze_stage_name']}), "
                              f"success_rate={cur_info['maze_success_rate']:.2f}")

                    if advanced:
                        # Update maze grid in all environments
                        new_grid = self.curriculum.get_maze_grid()
                        if hasattr(self.training_env, 'env_method'):
                            try:
                                self.training_env.env_method('set_grid', new_grid)
                            except AttributeError:
                                pass

        return True


def main():
    parser = argparse.ArgumentParser(description="Train maze navigation")
    parser.add_argument("--mode", choices=["hierarchical", "end-to-end"], default="hierarchical")
    parser.add_argument("--walking-model", type=str, required=True, help="Path to walking model")
    parser.add_argument("--walking-vecnorm", type=str, default=None, help="Path to walking VecNormalize")
    parser.add_argument("--config", type=str, default="config/training_config.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-stage", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.timesteps:
        config["total_timesteps"] = args.timesteps
    if args.n_envs:
        config["n_envs"] = args.n_envs

    print(f"\n{'='*60}")
    print(f"MAZE NAVIGATION TRAINING ({args.mode})")
    print(f"{'='*60}")
    print(f"  Walking model: {args.walking_model}")
    print(f"  Timesteps: {config['total_timesteps']:,}")
    print(f"  Environments: {config['n_envs']}")
    print(f"{'='*60}\n")

    # Initialize curriculum
    curriculum = MazeCurriculum(
        start_stage=args.start_stage,
        seed=args.seed,
    )
    print(f"Starting at maze stage {curriculum.current_stage}: {curriculum.stage_config['name']}")

    # Create environments
    n_envs = config["n_envs"]
    start_method = get_subprocess_start_method()

    def make_env(rank):
        def _init():
            return make_maze_env(
                args.walking_model, args.walking_vecnorm, config, rank=rank
            )
        return _init

    if n_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method=start_method)
    else:
        vec_env = DummyVecEnv([make_env(0)])

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Create model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        verbose=1,
        seed=args.seed,
        tensorboard_log="data/logs/maze",
    )

    # Model manager
    manager = ModelManager("maze", base_dir="models")
    manager.archive_config(config)

    # Callbacks
    curriculum_cb = MazeCurriculumCallback(curriculum, verbose=1)
    callbacks = CallbackList([curriculum_cb])

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    manager.save_final(model, vec_env)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
