import argparse
import os
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import yaml

# Ensure project root is on sys.path (so `src.*` imports work like in training)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Also add `src` for direct package-style imports without the `src.` prefix if needed
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.utils.visualization import setup_display, test_environment  # noqa: E402

# import SB3 algorithms we support
try:
    from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False

# Try to import custom environment with robust fallbacks
CUSTOM_ENV_AVAILABLE = False
make_standing_env = None
make_standing_curriculum_env = None
try:
    from src.environments.standing_env import make_standing_env  # type: ignore
    from src.environments.standing_curriculum import make_standing_curriculum_env  # type: ignore
    CUSTOM_ENV_AVAILABLE = True
except Exception:
    try:
        # Fallback if only `src` is on sys.path as a package root
        from environments.standing_env import make_standing_env  # type: ignore
        from environments.standing_curriculum import make_standing_curriculum_env  # type: ignore
        CUSTOM_ENV_AVAILABLE = True
    except Exception:
        CUSTOM_ENV_AVAILABLE = False

# Suppress Gymnasium warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class InferenceActionWarmup(gym.Wrapper):
    """
    Warm up action smoothing history during inference.
    
    CRITICAL FIX: Action smoothing creates temporal dependencies.
    During taining, prev_action has realistic values, but at reset it's zeros.
    This wrapper builds realistic action history before evaluation starts.
    """
    def __init__(self, env, warmup_steps=10, warmup_noise=0.01):
        super().__init__(env)
        self.warmup_steps = warmup_steps
        self.warmup_noise = warmup_noise
        print(f" InferenceActionWarmup enabled: {warmup_steps} warmup steps")
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Warmup phase: take small random actions to build action history
        for i in range(self.warmup_steps):
            # Gradually increase noise (start from nearly zero)
            scale = (i + 1) / self.warmup_steps * self.warmup_noise
            action = self.env.action_space.sample() * scale
            obs, _, terminated, truncated, _ = self.env.step(action)
            
            # If warmup causes termination (shouldn't happen), restart
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        
        return obs, info


def load_training_config(config_path=None):
    """Load configuration from training_config.yaml"""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, 'config', 'training_config.yaml')
    
    if not os.path.exists(config_path):
        print(f"WARNING: Config file not found at {config_path}")
        print("Using default hardcoded values")
        return None
    
    try:
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Extract standing config
        standing_config = full_config.get('standing', {})
        print(f" Loaded training config from {config_path}")
        return standing_config
    except Exception as e:
        print(f"ERROR loading config: {e}")
        return None


def create_environment(env_name, render_mode="rgb_array", task_type=None, vecnorm_path=None):
    """Create environment - handles both standard and custom setups"""
    env = None
    vec_env = None
    
    # ALWAYS use custom environment for standing task if available
    if task_type == "standing":
        if not CUSTOM_ENV_AVAILABLE:
            print("WARNING: Custom environment for standing task not found. Falling back to standard environment.")
            print(f"Creating standard environment: {env_name}")
            env = gym.make(env_name, render_mode=render_mode)
            return env, False
        
        print(f"Creating custom {task_type} environment...")
        
        # Load training configuration from YAML (matches training exactly!)
        yaml_config = load_training_config()
        
        if yaml_config:
            # Use config from YAML file
            training_config = {
                'obs_history': yaml_config.get('obs_history', 4),
                'obs_include_com': yaml_config.get('obs_include_com', True),
                'obs_feature_norm': yaml_config.get('obs_feature_norm', True),
                'action_smoothing': yaml_config.get('action_smoothing', True),
                'action_smoothing_tau': yaml_config.get('action_smoothing_tau', 0.5),
                # Curriculum settings (will stay at final stage for inference)
                'curriculum_start_stage': 4,   # Start at FINAL stage (1.40m)
                'curriculum_max_stage': 4,     # Stay at final stage
            }
            print(f"  Using config from YAML:")
            print(f"    obs_history: {training_config['obs_history']}")
            print(f"    action_smoothing_tau: {training_config['action_smoothing_tau']}")
        else:
            # Fallback to hardcoded values if YAML not found
            training_config = {
                'obs_history': 4,
                'obs_include_com': True,
                'obs_feature_norm': True,
                'action_smoothing': True,
                'action_smoothing_tau': 0.5,  # Updated default
                'curriculum_start_stage': 4,  # Final stage (1.40m)
                'curriculum_max_stage': 4,
            }
            print("  Using fallback config (YAML not found)")
        
        # Create base environment
        base_env = make_standing_curriculum_env(render_mode=render_mode, config=training_config)
        
        # Wrap with InferenceActionWarmup
        # This builds realistic action history before evaluation starts
        base_env = InferenceActionWarmup(base_env, warmup_steps=10)
        
        # Reset the environment BEFORE wrapping in VecEnv
        # This allows the observation space to freeze to the correct dimension (1484)
        # Otherwise VecEnv allocates buffer for the initial size (1424)
        print("Pre-warming environment to freeze observation dimension...")
        _ = base_env.reset()
        print(f"Environment observation space after reset: {base_env.observation_space.shape}")
        
        # Now wrap in VecEnv with the correct frozen dimension
        vec_env = DummyVecEnv([lambda: base_env])
        
        # CRITICAL: Load VecNormalize stats if available
        if vecnorm_path:
            # Try relative path first, then absolute from PROJECT_ROOT
            if not os.path.exists(vecnorm_path):
                alt_vecnorm = os.path.join(PROJECT_ROOT, vecnorm_path)
                if os.path.exists(alt_vecnorm):
                    vecnorm_path = alt_vecnorm
            
            if os.path.exists(vecnorm_path):
                print(f"Loading VecNormalize from {vecnorm_path}")
                try:
                    vec_env = VecNormalize.load(vecnorm_path, vec_env)
                    vec_env.training = False  # IMPORTANT: Set to eval mode
                    vec_env.norm_reward = False  # Don't normalize rewards during inference
                    print(f" VecNormalize loaded and configured for inference")
                except Exception as e:
                    print(f" VecNormalize loading failed: {e}")
                    raise
            else:
                print(f" WARNING: No VecNormalize file found at {vecnorm_path}")
                print(f" Current working directory: {os.getcwd()}")
                print("Model will likely fail without normalization!")
        else:
            print(f" WARNING: No VecNormalize path provided")
            print("Model will likely fail without normalization!")
            
        # Debug: print observation space shape for verification
        try:
            print(f"Observation space (custom standing): {vec_env.observation_space.shape}")
        except Exception:
            pass
        return vec_env, True  # Return (env, is_vectorized)
    
    # Standard environment fallback for other tasks
    print(f"Creating standard environment: {env_name}")
    env = gym.make(env_name, render_mode=render_mode)
    return env, False

def load_model(model_path, env, algo, is_vectorized=False):
    """Load a Stable-Baselines3 model from .zip."""
    if model_path is None:
        print("No valid model provided, using random policy")
        return None
    
    # Try relative path first, then absolute from PROJECT_ROOT
    if not os.path.exists(model_path):
        alt_path = os.path.join(PROJECT_ROOT, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
            print(f"Found model at: {model_path}")
        else:
            print(f"Model file not found at: {model_path}")
            print(f"Also tried: {alt_path}")
            print(f"Current working directory: {os.getcwd()}")
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

    print("Starting video recording...")
    total_steps = 0
    total_frames = 0
    max_frames = args.episodes * args.steps  # Total budget
    
    for episode in range(args.episodes):
        print(f"Recording episode {episode + 1}/{args.episodes}")
        
        # Reset environment correctly based on vectorization
        if is_vectorized:
            obs = env.reset()  # Returns NumPy array for VecEnv
        else:
            obs, _ = env.reset()  # Returns (obs, info) tuple for Gym env
        
        episode_steps = 0
        
        while episode_steps < args.steps and total_frames < max_frames:
            # Get action using the correct observation
            action = get_action(model, obs, env, is_vectorized)
            if is_vectorized:
                obs, reward, done, info = safe_step(env, action, is_vectorized)
            else:
                obs, reward, done, _ = safe_step(env, action, is_vectorized)  # Ignore info if not needed
            
            # Render frame
            try:
                frame = safe_render(env, is_vectorized)
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_resized = cv2.resize(frame_bgr, (args.width, args.height))
                    out.write(frame_resized)
                    total_frames += 1
                    episode_steps += 1
            except Exception as e:
                print(f"Frame error: {e}")
                break
            
            # Check termination
            is_done = (is_vectorized and done[0]) or (not is_vectorized and done)
            
            if is_done:
                # Print why it failed
                if is_vectorized and len(info) > 0:
                    height = info[0].get('height', 'unknown')
                    print(f"   Episode terminated at step {episode_steps}, height={height}")
                break
        
        if episode_steps >= args.steps:
            print(f"   Episode {episode + 1} SUCCESS: {episode_steps} steps")
        
        print(f"  Recorded {episode_steps} steps, total frames: {total_frames}")
    
    out.release()
    print(f"Video saved: {args.output} ({total_frames} frames, {total_frames/args.fps:.1f}s)")

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

    # Autofill VecNormalize path for standing task if not provided
    if (args.task == "standing") and not args.vecnorm:
        # Try both possible default names
        default_paths = [
            os.path.join(PROJECT_ROOT, "models", "saved_models", "vecnorm_standing.pkl"),
            os.path.join(PROJECT_ROOT, "models", "saved_models", "vecnorm.pkl"),
        ]
        for default_vecnorm in default_paths:
            if os.path.exists(default_vecnorm):
                args.vecnorm = default_vecnorm
                print(f"Using default VecNormalize stats: {args.vecnorm}")
                break
        else:
            print(f"WARNING: No VecNormalize stats found. Tried: {default_paths}")

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

    # Print obs space to confirm correct dimension (should be 350 for standing)
    try:
        print(f"Observation space: {env.observation_space}")
    except Exception:
        pass

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