"""
Quick test to determine the exact observation space dimension from training config
"""
import os
import sys

# Get the script's directory and navigate to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Add to path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print(f"Project root: {PROJECT_ROOT}")
print(f"Src dir: {SRC_DIR}")

try:
    from environments.standing_curriculum import make_standing_curriculum_env
    print("✓ Imported from environments.standing_curriculum")
except ImportError:
    try:
        from src.environments.standing_curriculum import make_standing_curriculum_env
        print("✓ Imported from src.environments.standing_curriculum")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("\nTrying direct import...")
        # Last resort: add the environments directory directly
        env_dir = os.path.join(PROJECT_ROOT, 'src', 'environments')
        sys.path.insert(0, env_dir)
        from standing_curriculum import make_standing_curriculum_env
        print("✓ Imported directly from standing_curriculum")

from stable_baselines3.common.vec_env import DummyVecEnv

# Match training config EXACTLY
training_config = {
    'obs_history': 4,
    'obs_include_com': True,
    'obs_feature_norm': True,
    'action_smoothing': True,
    'action_smoothing_tau': 0.2,
    'curriculum_start_stage': 3,
    'curriculum_max_stage': 3,
}

# Create environment the same way as training
env = make_standing_curriculum_env(render_mode=None, config=training_config)
vec_env = DummyVecEnv([lambda: env])

# Reset to freeze the observation dimension
obs = vec_env.reset()

print(f"Observation shape after reset: {obs.shape}")
print(f"Environment observation space: {vec_env.observation_space}")

# Try to get the frozen dimension from the unwrapped env
try:
    unwrapped = vec_env.envs[0].unwrapped
    if hasattr(unwrapped, '_proc_obs_dim'):
        print(f"Frozen processed obs dim: {unwrapped._proc_obs_dim}")
except Exception as e:
    print(f"Could not get frozen dim: {e}")