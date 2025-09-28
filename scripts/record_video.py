"""
Universal video recording script for humanoid agents.
Enhanced to work with both standard Gymnasium and custom HumanoidEnv setups.

record_video.py
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
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False

# Try to import custom environment
try:
    from src.environments.humanoid_env import make_humanoid_env
    CUSTOM_ENV_AVAILABLE = True
except Exception:
    CUSTOM_ENV_AVAILABLE = False


def create_environment(env_name, render_mode="rgb_array", task_type=None, vecnorm_path=None):
    """Create environment - handles both standard and custom setups"""
    env = None
    vec_env = None
    
    # Try custom environment first if task_type is specified
    if task_type and CUSTOM_ENV_AVAILABLE:
        try:
            print(f"Creating custom {task_type} environment...")
            env = make_humanoid_env(task_type=task_type, render_mode=render_mode)
            
            # Wrap in VecEnv for SB3 compatibility
            vec_env = DummyVecEnv([lambda: env])
            
            # Load VecNormalize if available
            if vecnorm_path and os.path.exists(vecnorm_path):
                print(f"Loading VecNormalize from {vecnorm_path}")
                try:
                    vec_env = VecNormalize.load(vecnorm_path, vec_env)
                    vec_env.training = False
                    vec_env.norm_reward = False
                    print("VecNormalize loaded successfully")
                except Exception as e:
                    print(f"VecNormalize loading failed: {e}")
            
            return vec_env, True  # Return (env, is_vectorized)
            
        except Exception as e:
            print(f"Custom environment creation failed: {e}")
            print("Falling back to standard Gymnasium environment...")
    
    # Standard Gymnasium environment
    try:
        print(f"Creating standard environment: {env_name}")
        env = gym.make(env_name, render_mode=render_mode)
        return env, False  # Return (env, is_vectorized)
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return None, False


def load_model(model_path, env, algo, is_vectorized=False):
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
        # Load model with or without environment binding
        model = algo_cls.load(model_path, env=env, device="auto", print_system_info=False)
        return model
    except Exception as e:
        print(f"Failed to load model with env: {e}")
        # Try loading without environment binding
        try:
            model = algo_cls.load(model_path, device="auto", print_system_info=False)
            print("Model loaded without environment binding")
            return model
        except Exception as e2:
            print(f"Complete model loading failure: {e2}")
            return None


def get_action(model, observation, env, is_vectorized=False):
    """Get action from SB3 model or random policy."""
    if model is None:
        if is_vectorized:
            return [env.action_space.sample()]
        return env.action_space.sample()
    
    try:
        action, _ = model.predict(observation, deterministic=True)
        return action
    except Exception as e:
        print(f"Model predict failed ({e}), falling back to random action.")
        if is_vectorized:
            return [env.action_space.sample()]
        return env.action_space.sample()


def safe_step(env, action, is_vectorized=False):
    """Handle step with different return formats"""
    result = env.step(action)
    
    if is_vectorized:
        # VecEnv returns obs, reward, done, info
        if len(result) == 4:
            return result
        else:
            # Handle new format if needed
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
    else:
        # Regular env - handle both old and new gym API
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, info
        elif len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        else:
            raise ValueError(f"Unexpected step() return format: {len(result)} values")


def safe_render(env, is_vectorized=False):
    """Handle rendering for different environment types"""
    try:
        if is_vectorized:
            # VecEnv rendering
            frame = env.render(mode='rgb_array')
        else:
            # Regular env rendering
            frame = env.render()
        return frame
    except Exception as e:
        print(f"Render error: {e}")
        return None


def setup_video_writer(output_path, fps, width, height):
    """Setup video writer with codec fallback"""
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
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


def record_frames_only(env, model, args, is_vectorized=False):
    """Fallback: save frames as individual images"""
    frames_dir = Path(args.output or "data/videos/humanoid_frames").parent / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving frames to {frames_dir}")
    frame_count = 0

    for episode in range(args.episodes):
        print(f"Recording episode {episode + 1}/{args.episodes}")
        
        if is_vectorized:
            obs = env.reset()
        else:
            obs, info = env.reset()

        for step in range(args.steps):
            action = get_action(model, obs, env, is_vectorized)
            obs, reward, done, info = safe_step(env, action, is_vectorized)

            try:
                frame = safe_render(env, is_vectorized)
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_resized = cv2.resize(frame_bgr, (args.width, args.height))

                    frame_path = frames_dir / f'episode_{episode:02d}_frame_{frame_count:06d}.png'
                    cv2.imwrite(str(frame_path), frame_resized)
                    frame_count += 1

            except Exception as e:
                print(f"Frame save error at step {step}: {e}")
                break

            if (is_vectorized and done[0]) or (not is_vectorized and done):
                break

        print(f"Episode {episode + 1} complete: {step + 1} steps")

    print(f"Frames saved to {frames_dir}")


def record_video(env, model, args, is_vectorized=False):
    """Record video with proper codec handling"""
    # Setup output path
    if args.output is None:
        model_name = "random" if args.random or model is None else args.algo.lower()
        task_suffix = f"_{args.task}" if hasattr(args, 'task') and args.task else ""
        args.output = f"data/videos/humanoid_{model_name}{task_suffix}_recording.mp4"

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Try to setup video writer
    out = setup_video_writer(args.output, args.fps, args.width, args.height)

    if out is None or args.frames_only:
        record_frames_only(env, model, args, is_vectorized)
        return

    # Record video
    print("Starting video recording...")
    total_steps = 0

    for episode in range(args.episodes):
        print(f"Recording episode {episode + 1}/{args.episodes}")
        
        if is_vectorized:
            obs = env.reset()
        else:
            obs, info = env.reset()

        for step in range(args.steps):
            action = get_action(model, obs, env, is_vectorized)
            obs, reward, done, info = safe_step(env, action, is_vectorized)

            try:
                frame = safe_render(env, is_vectorized)
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_resized = cv2.resize(frame_bgr, (args.width, args.height))
                    out.write(frame_resized)
                    total_steps += 1

            except Exception as e:
                print(f"Rendering error at episode {episode}, step {step}: {e}")
                break

            if (is_vectorized and done[0]) or (not is_vectorized and done):
                break

            if total_steps % 100 == 0:
                print(f"Recorded {total_steps} frames...")

        print(f"Episode {episode + 1} complete: {step + 1} steps")

    out.release()
    print(f"Video saved as '{args.output}'")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Record videos with trained SB3 model or random policy.")
    parser.add_argument("--env", type=str, default="Humanoid-v5", help="Gymnasium environment id")
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
    
    # NEW: Custom environment options
    parser.add_argument("--task", type=str, choices=["standing", "walking", "navigation"],
                        help="Custom task type (uses HumanoidEnv wrapper)")
    parser.add_argument("--vecnorm", type=str, help="Path to VecNormalize stats (.pkl file)")
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 50)
    print("HUMANOID VIDEO RECORDING")
    print("=" * 50)

    if not setup_display():
        print("Failed to setup display. Exiting.")
        return 1

    # Create environment (handles both standard and custom)
    env, is_vectorized = create_environment(
        env_name=args.env,
        render_mode="rgb_array", 
        task_type=args.task,
        vecnorm_path=args.vecnorm
    )
    
    if env is None:
        print("Failed to create environment")
        return 1

    # Test environment if it's not vectorized
    if not is_vectorized and not test_environment(env):
        print("Environment test failed. Check your setup.")
        env.close()
        return 1

    # Load model
    model = None if args.random else load_model(args.model, env, args.algo, is_vectorized)

    try:
        record_video(env, model, args, is_vectorized)
    except Exception as e:
        print(f"Recording failed: {e}")
        import traceback
        traceback.print_exc()
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