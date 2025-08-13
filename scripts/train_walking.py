"""
Humanoid walking PPO training with parallel VecEnvs and reward shaping
Colab-optimised for headless MuJoCo + T4 GPU
"""

# ======================================================
# EARLY ENVIRONMENT CONFIGURATION (before any Mujoco/Gym imports)
# ======================================================
import os
os.environ.setdefault("MUJOCO_GL", "egl")       # Headless rendering backend
os.environ.setdefault("OMP_NUM_THREADS", "1")   # Limit CPU threading
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import yaml
from datetime import datetime
import torch
torch.set_num_threads(1)

# ======================================================
# PROJECT IMPORTS
# ======================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agents.walking_agent import WalkingAgent  # <-- use the parallel VecEnv + VecNormalize version

# ======================================================
# SETUP HELPERS
# ======================================================
def setup_colab_environment():
    """Quick setup for Colab runtime."""
    os.environ.setdefault("MUJOCO_GL", "egl")

    try:
        import gymnasium
        import stable_baselines3
    except ImportError:
        print("Installing dependencies...")
        os.system(
            "pip install --upgrade "
            "gymnasium[mujoco]==0.29.1 "
            "mujoco>=3.1.4 "
            "stable-baselines3>=2.3.0 "
            "tensorboard pyyaml"
        )

    try:
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        else:
            print("Using CPU (training will be slower)")
            return 'cpu'
    except Exception:
        return 'cpu'


def load_config():
    """Load or create default training config."""
    config_path = 'config/training_config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
    else:
        print("Using default configuration")
        config = {
            'walking': {
                # PPO / Training parameters
                'total_timesteps': 3_000_000,
                'learning_rate': 3e-4,
                'batch_size': 256,
                'n_steps': 2048,      # per-env; total rollout = n_steps * n_envs
                'clip_range': 0.1,
                'ent_coef': 0.001,
                'save_freq': 250_000,
                'eval_freq': 50_000,
                'verbose': 1,
                'seed': 42,
                'n_envs': 2,           # Colab typically has 2 vCPUs
                'normalize': True,     # VecNormalize obs+reward
            }
        }
    return config

# ======================================================
# TRAINING MAIN
# ======================================================
def main():
    print("=== Humanoid Walking PPO Training ===")
    device = setup_colab_environment()

    # Load configuration
    config = load_config()
    walking_config = config.get('walking', {})
    walking_config['device'] = device

    # Experiment naming
    timestamp = datetime.now().strftime("%m%d_%H%M")
    experiment_name = f"walking_colab_{timestamp}"
    walking_config['log_dir'] = f"data/logs/{experiment_name}"
    os.makedirs(walking_config['log_dir'], exist_ok=True)

    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Timesteps: {walking_config['total_timesteps']:,}")
    print(f"Log dir: {walking_config['log_dir']}")

    try:
        # Create and train agent
        agent = WalkingAgent(walking_config)
        model = agent.train()

        # Evaluate without rendering
        results = agent.evaluate(n_episodes=3, render=False)

        # Save final eval results
        results_path = f"{walking_config['log_dir']}/final_results.txt"
        with open(results_path, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Mean Reward: {results['mean_reward']:.2f}\n")
            f.write(f"Std Reward: {results['std_reward']:.2f}\n")
            f.write(f"Configuration: {walking_config}\n")
        print(f"Results saved to: {results_path}")

        print("Best model: models/saved_models/best_walking_model.zip")
        print("Final model: models/saved_models/final_walking_model.zip")

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        if 'agent' in locals() and hasattr(agent, 'model') and agent.model is not None:
            interrupted_path = f"models/saved_models/interrupted_walking_{timestamp}.zip"
            agent.model.save(interrupted_path)
            print(f"Progress saved to: {interrupted_path}")

    finally:
        if 'agent' in locals():
            agent.close()
        print("Cleanup complete.")

# ======================================================
# QUICK ENV TEST
# ======================================================
def quick_test():
    print("Running quick test...")
    os.environ.setdefault("MUJOCO_GL", "egl")
    setup_colab_environment()

    from src.environments.humanoid_env import make_humanoid_env
    env = make_humanoid_env("walking")
    obs, info = env.reset(seed=0)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    env.close()

# ======================================================
# SIMPLE TRAIN FUNCTION (from a Colab cell)
# ======================================================
def train_walking(timesteps=3_000_000, device='auto'):
    cfg = {
        'total_timesteps': timesteps,
        'device': device,
        'learning_rate': 3e-4,
        'batch_size': 256,
        'n_steps': 2048,
        'clip_range': 0.1,
        'ent_coef': 0.001,
        'verbose': 1,
        'n_envs': 2,
        'normalize': True,
        'log_dir': f"data/logs/walking_{datetime.now().strftime('%m%d_%H%M')}"
    }
    setup_colab_environment()
    agent = WalkingAgent(cfg)
    model = agent.train()
    results = agent.evaluate(n_episodes=3, render=False)
    agent.close()
    return model, results

# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    main()
