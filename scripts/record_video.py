"""
Universal video recording script for humanoid agents.
Can record videos of trained models or random policies (Stable-Baselines3).
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.visualization import setup_display, test_environment  # noqa: E402

# Optional: import SB3 algorithms we support
try:
    from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False

def load_model(model_path, env, algo):
    """Load a Stable-Baselines3 model from .zip."""
    if model_path is None or not os.path.exists(model_path):
        print("No valid model provided, using random policy")
        return None

    if not SB3_AVAILABLE:
        print("Stable-Baselines3 not installed. Using random policy.")
        return None

    algo_map = {
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "td3": TD3,
        "ddpg": DDPG,
    }
    algo_cls = algo_map.get(algo.lower())
    if algo_cls is None:
        print(f"Unknown algo '{algo}'. Using random policy.")
        return None

    print(f"Loading SB3 {algo.upper()} model from: {model_path}")
    try:
        # Passing env binds spaces for predict/eval; device auto-detects CPU/GPU
        model = algo_cls.load(model_path, env=env, device="auto", print_system_info=False)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


def get_action(model, observation, env):
    """Get action from SB3 model or random policy."""
    if model is None:
        return env.action_space.sample()
    try:
        action, _ = model.predict(observation, deterministic=True)
        return action
    except Exception as e:
        print(f"Model predict failed ({e}), falling back to random action.")
        return env.action_space.sample()


def setup_video_writer(output_path, fps, width, height):
    """Setup video writer with codec fallback"""
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # prefer mp4 if extension is .mp4
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
    ]

    for codec_name, fourcc in codecs_to_try:
        print(f"Trying codec: {codec_name}")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"Successfully initialized VideoWriter with {codec_name}")
            return out
        print(f"{codec_name} failed")
        out.release()

    print("ERROR: Could not initialize any video codec!")
    return None


def record_frames_only(env, model, args):
    """Fallback: save frames as individual images"""
    frames_dir = Path(args.output or "data/videos/humanoid_frames").parent / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving frames to {frames_dir}")
    frame_count = 0

    for episode in range(args.episodes):
        print(f"Recording episode {episode + 1}/{args.episodes}")
        obs, info = env.reset()

        for step in range(args.steps):
            action = get_action(model, obs, env)
            obs, reward, terminated, truncated, info = env.step(action)

            try:
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame_bgr, (args.width, args.height))

                frame_path = frames_dir / f'episode_{episode:02d}_frame_{frame_count:06d}.png'
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
        model_name = "random" if args.random or model is None else args.algo.lower()
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
            action = get_action(model, obs, env)
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Record Gymnasium env videos with trained SB3 model or random policy.")
    parser.add_argument("--env", type=str, default="Humanoid-v4", help="Gymnasium environment id")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=360, help="Video height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--output", type=str, default=None, help="Output video path (mp4/avi)")
    parser.add_argument("--random", action="store_true", help="Use random actions instead of a trained model")
    parser.add_argument("--frames_only", action="store_true", help="Save individual frames instead of a video")
    parser.add_argument("--model", type=str, default=None, help="Path to Stable-Baselines3 model (.zip)")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo","a2c","sac","td3","ddpg"],
                        help="SB3 algorithm used to train the model")
    return parser.parse_args()

def main():
    args = parse_arguments()

    print("=" * 50)
    print("HUMANOID VIDEO RECORDING")
    print("=" * 50)

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

    model = None if args.random else load_model(args.model, env, args.algo)

    try:
        record_video(env, model, args)
    except Exception as e:
        print(f"Recording failed: {e}")
        return 1
    finally:
        env.close()

    print("=" * 50)
    print("RECORDING COMPLETE!")
    print(f"Output: {args.output}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
