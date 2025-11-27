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

# Try to import custom environments with robust fallbacks
CUSTOM_ENV_AVAILABLE = False
make_standing_env = None
make_standing_curriculum_env = None
make_walking_env = None
make_walking_curriculum_env = None

try:
    from src.environments.standing_env import make_standing_env  # type: ignore
    from src.environments.standing_curriculum import make_standing_curriculum_env  # type: ignore
    from src.environments.walking_env import make_walking_env  # type: ignore
    from src.environments.walking_curriculum import make_walking_curriculum_env  # type: ignore
    CUSTOM_ENV_AVAILABLE = True
except Exception:
    try:
        from environments.standing_env import make_standing_env  # type: ignore
        from environments.standing_curriculum import make_standing_curriculum_env  # type: ignore
        from environments.walking_env import make_walking_env  # type: ignore
        from environments.walking_curriculum import make_walking_curriculum_env  # type: ignore
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
    During training, prev_action has realistic values, but at reset it's zeros.
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


def load_training_config(config_path=None, task='standing'):
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
        
        # Extract task-specific config
        task_config = full_config.get(task, full_config.get('standing', {}))
        print(f" Loaded {task} config from {config_path}")
        return task_config
    except Exception as e:
        print(f"ERROR loading config: {e}")
        return None


def create_environment(env_name, render_mode="rgb_array", task_type=None, vecnorm_path=None, 
                       vx_target=None, vy_target=None):
    """Create environment - handles both standard and custom setups"""
    env = None
    vec_env = None
    
    # Handle walking task
    if task_type == "walking":
        if not CUSTOM_ENV_AVAILABLE or make_walking_env is None:
            print("WARNING: Walking environment not available. Check imports.")
            return None, False
        
        print(f"Creating custom walking environment...")
        
        # Load walking configuration from YAML
        yaml_config = load_training_config(task=task_type)
        
        if yaml_config:
            training_config = {
                'obs_history': yaml_config.get('obs_history', 4),
                'obs_include_com': yaml_config.get('obs_include_com', True),
                'obs_feature_norm': yaml_config.get('obs_feature_norm', True),
                'action_smoothing': yaml_config.get('action_smoothing', True),
                'action_smoothing_tau': 0.2,  # CRITICAL: Must match training
                'max_episode_steps': yaml_config.get('max_episode_steps', 5000),
                'random_height_init': False,
                'velocity_weight': yaml_config.get('velocity_weight', 5.0),
                'max_commanded_speed': 3.0,  # Allow full range during inference
            }
        else:
            training_config = {
                'obs_history': 4,
                'obs_include_com': True,
                'obs_feature_norm': True,
                'action_smoothing': True,
                'action_smoothing_tau': 0.2,
                'max_episode_steps': 5000,
                'random_height_init': False,
                'velocity_weight': 5.0,
                'max_commanded_speed': 3.0,
            }
        
        # Set fixed velocity command if provided
        if vx_target is not None and vy_target is not None:
            training_config['fixed_command'] = (vx_target, vy_target)
            print(f"  Fixed velocity command: vx={vx_target:.2f}, vy={vy_target:.2f} m/s")
        
        print(f"  Config: obs_history={training_config['obs_history']}, "
              f"tau={training_config['action_smoothing_tau']}")
        
        # Create base walking environment
        base_env = make_walking_env(render_mode=render_mode, config=training_config)
        
        # Wrap with InferenceActionWarmup
        base_env = InferenceActionWarmup(base_env, warmup_steps=10)
        
        # Pre-warm environment to freeze observation dimension
        print("Pre-warming environment to freeze observation dimension...")
        _ = base_env.reset()
        print(f"Environment observation space after reset: {base_env.observation_space.shape}")
        
        # Wrap in VecEnv
        vec_env = DummyVecEnv([lambda: base_env])
        
        # Load VecNormalize if available
        if vecnorm_path:
            if not os.path.exists(vecnorm_path):
                alt_vecnorm = os.path.join(PROJECT_ROOT, vecnorm_path)
                if os.path.exists(alt_vecnorm):
                    vecnorm_path = alt_vecnorm
            
            if os.path.exists(vecnorm_path):
                print(f"Loading VecNormalize from {vecnorm_path}")
                try:
                    vec_env = VecNormalize.load(vecnorm_path, vec_env)
                    vec_env.training = False
                    vec_env.norm_reward = False
                    print(f" VecNormalize loaded and configured for inference")
                except Exception as e:
                    print(f" VecNormalize loading failed: {e}")
                    raise
            else:
                print(f" WARNING: No VecNormalize file found at {vecnorm_path}")
        
        print(f"Observation space (walking): {vec_env.observation_space.shape}")
        return vec_env, True
    
    # Handle standing task (existing code)
    if task_type == "standing":
        if not CUSTOM_ENV_AVAILABLE:
            print("WARNING: Custom environment for standing task not found.")
            env = gym.make(env_name, render_mode=render_mode)
            return env, False
        
        print(f"Creating custom standing environment...")
        
        yaml_config = load_training_config(task='standing')
        
        if yaml_config:
            training_config = {
                'obs_history': yaml_config.get('obs_history', 4),
                'obs_include_com': yaml_config.get('obs_include_com', True),
                'obs_feature_norm': yaml_config.get('obs_feature_norm', True),
                'action_smoothing': yaml_config.get('action_smoothing', True),
                'action_smoothing_tau': 0.2,
                'max_episode_steps': yaml_config.get('max_episode_steps', 5000),
                'random_height_init': False,
            }
        else:
            training_config = {
                'obs_history': 4,
                'obs_include_com': True,
                'obs_feature_norm': True,
                'action_smoothing': True,
                'action_smoothing_tau': 0.2,
                'max_episode_steps': 5000,
                'random_height_init': False,
            }
        
        base_env = make_standing_env(render_mode=render_mode, config=training_config)
        base_env = InferenceActionWarmup(base_env, warmup_steps=10)
        
        print("Pre-warming environment to freeze observation dimension...")
        _ = base_env.reset()
        print(f"Environment observation space after reset: {base_env.observation_space.shape}")
        
        vec_env = DummyVecEnv([lambda: base_env])
        
        if vecnorm_path:
            if not os.path.exists(vecnorm_path):
                alt_vecnorm = os.path.join(PROJECT_ROOT, vecnorm_path)
                if os.path.exists(alt_vecnorm):
                    vecnorm_path = alt_vecnorm
            
            if os.path.exists(vecnorm_path):
                print(f"Loading VecNormalize from {vecnorm_path}")
                try:
                    vec_env = VecNormalize.load(vecnorm_path, vec_env)
                    vec_env.training = False
                    vec_env.norm_reward = False
                    print(f" VecNormalize loaded")
                except Exception as e:
                    print(f" VecNormalize loading failed: {e}")
                    raise
        
        print(f"Observation space (standing): {vec_env.observation_space.shape}")
        return vec_env, True
    
    # Standard environment fallback
    print(f"Creating standard environment: {env_name}")
    env = gym.make(env_name, render_mode=render_mode)
    return env, False


def load_model(model_path, env, algo, is_vectorized=False):
    """Load a Stable-Baselines3 model from .zip."""
    if model_path is None:
        print("No valid model provided, using random policy")
        return None
    
    if not os.path.exists(model_path):
        alt_path = os.path.join(PROJECT_ROOT, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"Model file not found at: {model_path}")
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
        model = algo_cls.load(model_path, env=env, device="auto", print_system_info=False)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
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
        if len(result) == 4:
            return result
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
    else:
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
            frame = env.render(mode='rgb_array')
        else:
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


def record_video(env, model, args, is_vectorized=False):
    """Record video with proper codec handling"""
    # Setup output path
    if args.output is None:
        model_name = "random" if args.random or model is None else args.algo.lower()
        task_suffix = f"_{args.task}" if args.task else ""
        vel_suffix = ""
        if args.vx_target is not None and args.vy_target is not None:
            vel_suffix = f"_vx{args.vx_target:.1f}_vy{args.vy_target:.1f}"
        args.output = f"data/videos/humanoid_{model_name}{task_suffix}{vel_suffix}_recording.mp4"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    out = setup_video_writer(args.output, args.fps, args.width, args.height)
    if out is None:
        print("ERROR: Could not initialize video writer")
        return

    print("Starting video recording...")
    total_steps = 0
    total_frames = 0
    
    # Tracking metrics for walking
    all_velocity_errors = []
    all_heights = []
    
    for episode in range(args.episodes):
        print(f"\nRecording episode {episode + 1}/{args.episodes}")
        
        if is_vectorized:
            obs = env.reset()
        else:
            obs, _ = env.reset()
        
        episode_steps = 0
        episode_velocity_errors = []
        episode_heights = []
        
        while episode_steps < args.steps:
            action = get_action(model, obs, env, is_vectorized)
            obs, reward, done, info = safe_step(env, action, is_vectorized)
            
            # Collect walking metrics
            if is_vectorized and len(info) > 0:
                if 'velocity_error' in info[0]:
                    episode_velocity_errors.append(info[0]['velocity_error'])
                if 'height' in info[0]:
                    episode_heights.append(info[0]['height'])
            
            # Render frame
            try:
                frame = safe_render(env, is_vectorized)
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_resized = cv2.resize(frame_bgr, (args.width, args.height))
                    
                    # Add overlay text for walking
                    if args.task == "walking" and is_vectorized and len(info) > 0:
                        vel_err = info[0].get('velocity_error', 0)
                        height = info[0].get('height', 0)
                        cmd_vx = info[0].get('commanded_vx', 0)
                        cmd_vy = info[0].get('commanded_vy', 0)
                        actual_vx = info[0].get('x_velocity', 0)
                        actual_vy = info[0].get('y_velocity', 0)
                        
                        # Add text overlay
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame_resized, f"Cmd: ({cmd_vx:.2f}, {cmd_vy:.2f}) m/s", 
                                   (10, 30), font, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame_resized, f"Act: ({actual_vx:.2f}, {actual_vy:.2f}) m/s", 
                                   (10, 55), font, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame_resized, f"Vel Err: {vel_err:.3f} m/s", 
                                   (10, 80), font, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame_resized, f"Height: {height:.3f} m", 
                                   (10, 105), font, 0.6, (255, 255, 255), 2)
                    
                    out.write(frame_resized)
                    total_frames += 1
                    episode_steps += 1
            except Exception as e:
                print(f"Frame error: {e}")
                break
            
            is_done = (is_vectorized and done[0]) or (not is_vectorized and done)
            if is_done:
                if is_vectorized and len(info) > 0:
                    height = info[0].get('height', 'unknown')
                    print(f"   Episode terminated at step {episode_steps}, height={height}")
                break
        
        # Print episode summary for walking
        if episode_velocity_errors:
            avg_vel_err = np.mean(episode_velocity_errors)
            avg_height = np.mean(episode_heights)
            all_velocity_errors.extend(episode_velocity_errors)
            all_heights.extend(episode_heights)
            
            print(f"  Episode {episode + 1}: {episode_steps} steps, "
                  f"avg vel error: {avg_vel_err:.4f} m/s, avg height: {avg_height:.3f} m")
        else:
            print(f"  Episode {episode + 1}: {episode_steps} steps")
    
    out.release()
    print(f"\nVideo saved: {args.output} ({total_frames} frames, {total_frames/args.fps:.1f}s)")
    
    # Print overall walking metrics
    if all_velocity_errors:
        print(f"\n=== Walking Performance Summary ===")
        print(f"  Total velocity errors recorded: {len(all_velocity_errors)}")
        print(f"  Mean velocity error: {np.mean(all_velocity_errors):.4f} m/s")
        print(f"  Std velocity error: {np.std(all_velocity_errors):.4f} m/s")
        print(f"  Mean height: {np.mean(all_heights):.3f} m")
        print(f"  Height std: {np.std(all_heights):.4f} m")


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
    
    # Custom environment options
    parser.add_argument("--task", type=str, choices=["standing", "walking"],
                        help="Custom task type (uses HumanoidEnv wrapper)")
    parser.add_argument("--vecnorm", type=str, help="Path to VecNormalize stats (.pkl file)")
    
    # Walking-specific: velocity commands
    parser.add_argument("--vx_target", type=float, default=None,
                        help="Target x-velocity in world frame (m/s)")
    parser.add_argument("--vy_target", type=float, default=None,
                        help="Target y-velocity in world frame (m/s)")
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 60)
    print("HUMANOID VIDEO RECORDING")
    print("=" * 60)

    if not setup_display():
        print("Failed to setup display. Exiting.")
        return 1

    # Autofill VecNormalize path for tasks if not provided
    if args.task and not args.vecnorm:
        if args.task == "standing":
            default_paths = [
                os.path.join(PROJECT_ROOT, "models", "vecnorm.pkl"),
                os.path.join(PROJECT_ROOT, "models", "saved_models", "vecnorm.pkl"),
            ]
        elif args.task == "walking":
            default_paths = [
                os.path.join(PROJECT_ROOT, "models", "vecnorm_walking.pkl"),
                os.path.join(PROJECT_ROOT, "models", "saved_models", "vecnorm_walking.pkl"),
            ]
        else:
            default_paths = []
        
        for default_vecnorm in default_paths:
            if os.path.exists(default_vecnorm):
                args.vecnorm = default_vecnorm
                print(f"Using default VecNormalize stats: {args.vecnorm}")
                break

    # Create environment
    env, is_vectorized = create_environment(
        env_name=args.env,
        render_mode="rgb_array", 
        task_type=args.task,
        vecnorm_path=args.vecnorm,
        vx_target=args.vx_target,
        vy_target=args.vy_target
    )
    
    if env is None:
        print("Failed to create environment")
        return 1

    print(f"Observation space: {env.observation_space}")

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

    print("=" * 60)
    print("RECORDING COMPLETE!")
    print(f"Output: {args.output}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
