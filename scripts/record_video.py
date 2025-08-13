"""
Universal video recording script for humanoid agents.
Can record videos of trained models or random policies.
"""

import argparse
import os
import sys
import gymnasium as gym
import numpy as np
import cv2
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.visualization import setup_display, test_environment

def load_model(model_path, env):
    """Load a trained model (placeholder - implement based on your RL library)"""
    if model_path is None or not os.path.exists(model_path):
        print("No valid model provided, using random policy")
        return None
    
    # TODO: Implement model loading based on RL library
    # Example for stable-baselines3:
    # from stable_baselines3 import PPO
    # return PPO.load(model_path)
    
    print(f"Model loading not implemented yet. Using random policy.")
    return None

def get_action(model, observation):
    """Get action from model or random policy"""
    if model is None:
        # Random policy
        return env.action_space.sample()
    else:
        # TODO: Implement based on RL library
        # Example for stable-baselines3:
        # action, _ = model.predict(observation, deterministic=True)
        # return action
        return env.action_space.sample()

def setup_video_writer(output_path, fps, width, height):
    """Setup video writer with codec fallback"""
    codecs_to_try = [
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
    ]
    
    for codec_name, fourcc in codecs_to_try:
        print(f"Trying codec: {codec_name}")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if out.isOpened():
            print(f"Successfully initialized VideoWriter with {codec_name}")
            return out
        else:
            print(f"{codec_name} failed")
            out.release()
    
    print("ERROR: Could not initialize any video codec!")
    return None

def record_frames_only(env, model, args):
    """Fallback: save frames as individual images"""
    frames_dir = Path(args.output).parent / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving frames to {frames_dir}")
    frame_count = 0
    
    for episode in range(args.episodes):
        print(f"Recording episode {episode + 1}/{args.episodes}")
        obs, info = env.reset()
        
        for step in range(args.steps):
            action = get_action(model, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            try:
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame_bgr, (args.width, args.height))
                
                frame_path = frames_dir / f'episode_{episode:02d}_frame_{frame_count:04d}.png'
                cv2.imwrite(str(frame_path), frame_resized)
                frame_count += 1
                
            except Exception as e:
                print(f"Frame save error at step {step}: {e}")
                break
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} complete: {step + 1} steps")
    
    print(f"Frames saved to {frames_dir}")

def record_video(env, model, args):
    """Record video with proper codec handling"""
    # Setup output path
    if args.output is None:
        model_name = "random" if args.random or model is None else "trained"
        args.output = f"data/videos/humanoid_{model_name}_recording.mp4"
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Try to setup video writer
    out = setup_video_writer(args.output, args.fps, args.width, args.height)
    
    if out is None or args.frames_only:
        record_frames_only(env, model, args)
        return
    
    # Record video
    print("Starting video recording...")
    total_steps = 0
    
    for episode in range(args.episodes):
        print(f"Recording episode {episode + 1}/{args.episodes}")
        obs, info = env.reset()
        
        for step in range(args.steps):
            action = get_action(model, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            try:
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame_bgr, (args.width, args.height))
                out.write(frame_resized)
                total_steps += 1
                
            except Exception as e:
                print(f"Rendering error at episode {episode}, step {step}: {e}")
                break
            
            if terminated or truncated:
                break
            
            if total_steps % 100 == 0:
                print(f"Recorded {total_steps} frames...")
        
        print(f"Episode {episode + 1} complete: {step + 1} steps")
    
    out.release()
    print(f"Video saved as '{args.output}'")

def main():
    args = parse_arguments()
    
    print("="*50)
    print("HUMANOID VIDEO RECORDING")
    print("="*50)
    
    if not setup_display():
        print("Failed to setup display. Exiting.")
        return 1
    
    try:
        env = gym.make(args.env, render_mode="rgb_array")
        print(f"Environment '{args.env}' created successfully")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return 1
    
    if not test_environment(env):
        print("Environment test failed. Check your setup.")
        env.close()
        return 1
    
    model = None if args.random else load_model(args.model, env)
    
    try:
        record_video(env, model, args)
    except Exception as e:
        print(f"Recording failed: {e}")
        return 1
    finally:
        env.close()
    
    print("="*50)
    print("RECORDING COMPLETE!")
    print(f"Output: {args.output}")
    print("="*50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())