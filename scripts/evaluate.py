#!/usr/bin/env python3
# evaluate.py
"""
Unified evaluation and video recording script.

Supports both standing and walking tasks.

Usage:
    python scripts/evaluate.py --task walking --model models/walking/final/model.zip
    python scripts/evaluate.py --task walking --model models/walking/final/model.zip --record
    python scripts/evaluate.py --task walking --vx 1.0 --vy 0.0 --record
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_eval_env(task: str, config: dict, render_mode: str = "rgb_array"):
    """Create evaluation environment."""
    if task == "walking":
        from src.environments import make_walking_env
        return make_walking_env(render_mode=render_mode, config=config)
    elif task == "standing":
        from src.environments import make_standing_env
        return make_standing_env(render_mode=render_mode, config=config)
    else:
        raise ValueError(f"Unknown task: {task}")


def run_evaluation(
    model,
    env,
    n_episodes: int = 10,
    max_steps: int = 1000,
    verbose: bool = True
):
    """Run evaluation episodes and return metrics."""
    results = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        heights = []
        velocity_errors = []
        
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0] if hasattr(reward, '__len__') else reward
            episode_length += 1
            
            # Extract info
            if hasattr(info, '__len__') and len(info) > 0:
                info = info[0]
            
            if 'height' in info:
                heights.append(info['height'])
            if 'velocity_error' in info:
                velocity_errors.append(info['velocity_error'])
            
            if done[0] if hasattr(done, '__len__') else done:
                break
        
        result = {
            'episode': ep,
            'reward': episode_reward,
            'length': episode_length,
            'avg_height': np.mean(heights) if heights else 0,
            'avg_velocity_error': np.mean(velocity_errors) if velocity_errors else 0,
            'success': episode_length >= max_steps
        }
        results.append(result)
        
        if verbose:
            status = "OK  " if result['success'] else "FAIL"
            print(f"  Episode {ep+1:2d}: {status} len={result['length']:4d}, "
                  f"reward={result['reward']:7.1f}, height={result['avg_height']:.3f}")
    
    return results


def record_video(
    model,
    env,
    output_path: str,
    n_episodes: int = 1,
    max_steps: int = 500,
    fps: int = 30
):
    """Record evaluation video. Expects a VecEnv (DummyVecEnv, optionally VecNormalize-wrapped)."""
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Cannot record video.")
        return

    frames = []

    for ep in range(n_episodes):
        obs = env.reset()

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if done[0]:
                break

        print(f"  Episode {ep+1}: {len(frames)} frames")
    
    if frames:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"Video saved: {output_path}")
    else:
        print("No frames captured")


def main():
    parser = argparse.ArgumentParser(description="Evaluate humanoid model")
    parser.add_argument('--task', type=str, required=True, choices=['walking', 'standing'])
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--vecnorm', type=str, default=None, help='Path to VecNormalize')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--record', action='store_true', help='Record video')
    parser.add_argument('--output', type=str, default=None, help='Video output path')
    parser.add_argument('--vx', type=float, default=None, help='Walking: commanded vx')
    parser.add_argument('--vy', type=float, default=None, help='Walking: commanded vy')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config (e.g. config/real_humanoid_config.yaml). '
                             'When set, the task section is merged into eval config so xml_file '
                             'and other env settings match training.')
    parser.add_argument('--override', type=str, default=None,
                        help='Optional second YAML whose task section overrides --config '
                             '(matches train_walking.py semantics, e.g. ablation overrides).')
    args = parser.parse_args()

    # Build config: defaults first, then overlay YAML section if provided.
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.2,
        'max_episode_steps': args.max_steps,
    }
    if args.config:
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        section = yaml_cfg.get(args.task, {})
        if not section:
            print(f"WARNING: no '{args.task}' section in {args.config}; using defaults")
        else:
            config.update(section)
            # max_steps CLI flag should still win for eval episode length
            config['max_episode_steps'] = args.max_steps
            print(f"Loaded {args.task} config from {args.config}")
            if 'xml_file' in section:
                print(f"  xml_file: {section['xml_file']}")

    if args.override:
        with open(args.override, 'r') as f:
            ov_cfg = yaml.safe_load(f) or {}
        ov_section = ov_cfg.get(args.task, {}) or {}
        if ov_section:
            config.update(ov_section)
            config['max_episode_steps'] = args.max_steps
            print(f"Applied override from {args.override}")
            if 'xml_file' in ov_section:
                print(f"  xml_file -> {ov_section['xml_file']}")
            if 'heading_source' in ov_section:
                print(f"  heading_source -> {ov_section['heading_source']}")
    
    # Walking-specific
    if args.task == 'walking':
        if args.vx is not None or args.vy is not None:
            config['fixed_command'] = (args.vx or 0.0, args.vy or 0.0)
            print(f"Using fixed command: vx={config['fixed_command'][0]}, vy={config['fixed_command'][1]}")

    print(f"\n{'='*60}")
    print(f"HUMANOID {args.task.upper()} EVALUATION")
    print(f"{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"{'='*60}\n")

    # Create environment
    render_mode = "rgb_array" if args.record else None
    env = make_eval_env(args.task, config, render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize if provided
    if args.vecnorm and os.path.exists(args.vecnorm):
        print(f"Loading VecNormalize from: {args.vecnorm}")
        vec_env = VecNormalize.load(args.vecnorm, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Load model
    model_path = args.model if args.model.endswith('.zip') else f"{args.model}.zip"
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=vec_env)

    # Run evaluation
    print("\nRunning evaluation...")
    results = run_evaluation(model, vec_env, args.episodes, args.max_steps)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    success_rate = sum(1 for r in results if r['success']) / len(results) * 100
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Avg reward: {np.mean([r['reward'] for r in results]):.1f}")
    print(f"  Avg length: {np.mean([r['length'] for r in results]):.0f}")
    print(f"  Avg height: {np.mean([r['avg_height'] for r in results]):.3f}")
    if results[0]['avg_velocity_error'] > 0:
        print(f"  Avg velocity error: {np.mean([r['avg_velocity_error'] for r in results]):.3f}")

    # Record video if requested
    if args.record:
        output_path = args.output or f"data/videos/{args.task}_eval.mp4"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        print("\nRecording video...")

        # Rebuild env with rendering AND the same VecNormalize stats used in eval.
        # Without VecNormalize the policy sees unnormalized obs and falls within ~80 steps.
        render_env = make_eval_env(args.task, config, render_mode="rgb_array")
        render_vec = DummyVecEnv([lambda: render_env])
        if args.vecnorm and os.path.exists(args.vecnorm):
            render_vec = VecNormalize.load(args.vecnorm, render_vec)
            render_vec.training = False
            render_vec.norm_reward = False
        record_video(model, render_vec, output_path, n_episodes=1, max_steps=args.max_steps)
        render_vec.close()

    vec_env.close()
    print("\nEvaluation complete")


if __name__ == "__main__":
    main()

