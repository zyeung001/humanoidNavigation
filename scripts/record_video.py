#!/usr/bin/env python3
# record_video.py
"""
Video recording script for walking model evaluation.

Usage:
    python scripts/record_video.py --model models/walking/final/model.zip
    python scripts/record_video.py --model models/walking/final/model.zip --vx 1.0 --vy 0.5
    python scripts/record_video.py --model models/walking/final/model.zip --output my_video.mp4
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("WARNING: OpenCV not installed. Video recording disabled.")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environments import make_walking_env


def record_walking_video(
    model_path: str,
    output_path: str,
    vecnorm_path: str = None,
    vx: float = 0.5,
    vy: float = 0.0,
    yaw_rate: float = 0.0,
    n_episodes: int = 1,
    max_steps: int = 500,
    fps: int = 30,
    warmup_steps: int = 10,
):
    """Record video of walking model."""
    if not CV2_AVAILABLE:
        print("ERROR: OpenCV required for video recording")
        return
    
    print(f"\n{'='*60}")
    print("WALKING VIDEO RECORDING")
    print(f"{'='*60}")
    print(f"  Model: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Command: vx={vx:.2f}, vy={vy:.2f}, yaw={yaw_rate:.2f}")
    print(f"  Episodes: {n_episodes}, Max steps: {max_steps}")
    print(f"{'='*60}\n")
    
    # Create config
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.25,
        'max_episode_steps': max_steps,
        'fixed_command': (vx, vy),  # Fixed command for inference
    }
    
    # Create environment with rendering
    env = make_walking_env(render_mode="rgb_array", config=config)
    vec_env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize if provided
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize from: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=vec_env)
    
    # Record frames
    all_frames = []
    
    for ep in range(n_episodes):
        print(f"\nRecording episode {ep+1}/{n_episodes}...")
        obs = vec_env.reset()
        episode_frames = []
        episode_reward = 0
        
        # Warmup phase
        for _ in range(warmup_steps):
            action = vec_env.action_space.sample() * 0.01
            obs, _, _, _ = vec_env.step(action)
        
        # Main recording
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            episode_reward += reward[0]
            
            # Capture frame
            try:
                frame = vec_env.render()
                if frame is not None:
                    episode_frames.append(frame)
            except Exception as e:
                print(f"  Warning: Frame capture failed at step {step}: {e}")
            
            if done[0]:
                print(f"  Episode ended at step {step}")
                break
            
            if step % 100 == 0:
                print(f"  Step {step}/{max_steps}, reward={episode_reward:.1f}")
        
        print(f"  Episode {ep+1}: {len(episode_frames)} frames, reward={episode_reward:.1f}")
        all_frames.extend(episode_frames)
    
    vec_env.close()
    
    # Save video
    if all_frames:
        print(f"\nSaving {len(all_frames)} frames to {output_path}...")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        height, width = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in all_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"✓ Video saved: {output_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Duration: {len(all_frames)/fps:.1f}s at {fps}fps")
    else:
        print("ERROR: No frames captured")


def main():
    parser = argparse.ArgumentParser(description="Record walking model video")
    parser.add_argument('--model', type=str, required=True, help='Path to model.zip')
    parser.add_argument('--vecnorm', type=str, default=None, help='Path to vecnormalize.pkl')
    parser.add_argument('--output', type=str, default='data/videos/walking.mp4', help='Output path')
    parser.add_argument('--vx', type=float, default=0.5, help='Commanded forward velocity')
    parser.add_argument('--vy', type=float, default=0.0, help='Commanded lateral velocity')
    parser.add_argument('--yaw', type=float, default=0.0, help='Commanded yaw rate')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    args = parser.parse_args()
    
    # Find vecnorm if not specified
    vecnorm_path = args.vecnorm
    if not vecnorm_path:
        # Try to find it alongside model
        model_dir = Path(args.model).parent
        possible_paths = [
            model_dir / "vecnormalize.pkl",
            model_dir / "vecnorm.pkl",
            model_dir.parent / "vecnormalize.pkl",
        ]
        for p in possible_paths:
            if p.exists():
                vecnorm_path = str(p)
                print(f"Auto-detected VecNormalize: {vecnorm_path}")
                break
    
    record_walking_video(
        model_path=args.model,
        output_path=args.output,
        vecnorm_path=vecnorm_path,
        vx=args.vx,
        vy=args.vy,
        yaw_rate=args.yaw,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
