"""
Simple training script for humanoid walking
Designed to run in Google Colab
"""

import os
import sys
import yaml
from datetime import datetime

# Add project root to sys.path for imports (Colab-friendly and robust)
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our modules
from src.agents.walking_agent import WalkingAgent


def setup_colab_environment():
    """Quick setup for Colab"""
    # Install dependencies if needed
    try:
        import gymnasium
        import stable_baselines3
    except ImportError:
        print("Installing dependencies...")
        os.system("pip install gymnasium[mujoco] stable-baselines3[extra] tensorboard pyyaml")
    
    # Create directories
    dirs = ['data/logs', 'data/checkpoints', 'models/saved_models', 'data/videos']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        else:
            print("Using CPU (training will be slower)")
            return 'cpu'
    except:
        return 'cpu'


def load_config():
    """Load training configuration"""
    config_path = 'config/training_config.yaml'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
    else:
        # Default config if file doesn't exist
        print("Using default configuration")
        config = {
            'walking': {
                'total_timesteps': 500000,
                'learning_rate': 0.0003,
                'batch_size': 64,
                'n_steps': 2048,
                'save_freq': 25000,
                'eval_freq': 10000,
                'verbose': 1,
                'seed': 42
            }
        }
    
    return config


def main():
    """Main training function"""
    print("Humanoid Walking Training")
    print("=" * 40)
    
    # Setup environment
    device = setup_colab_environment()

    import os
    os.environ["MUJOCO_GL"] = "egl"
    
    # Load configuration
    config = load_config()
    
    # Get walking config and add device
    walking_config = config.get('walking', {})
    walking_config['device'] = device
    
    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%m%d_%H%M")
    experiment_name = f"walking_colab_{timestamp}"
    walking_config['log_dir'] = f"data/logs/{experiment_name}"
    
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Timesteps: {walking_config.get('total_timesteps', 500000):,}")
    print(f"Log dir: {walking_config['log_dir']}")
    print()
    
    # Create and train agent
    try:
        print("Creating agent...")
        agent = WalkingAgent(walking_config)
        
        print("Starting training...")
        model = agent.train()
        
        print("\nTraining completed!")
        
        # Quick evaluation
        print("Running evaluation...")
        results = agent.evaluate(n_episodes=3, render=False)
        
        # Save results
        results_path = f"data/logs/{experiment_name}/final_results.txt"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Mean Reward: {results['mean_reward']:.2f}\n")
            f.write(f"Std Reward: {results['std_reward']:.2f}\n")
            f.write(f"Configuration: {walking_config}\n")
        
        print(f"\nResults saved to: {results_path}")
        print(f"Best model: models/saved_models/best_walking_model.zip")
        print(f"Final model: models/saved_models/final_walking_model.zip")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        if 'agent' in locals():
            # Save current progress
            interrupted_path = f"models/saved_models/interrupted_walking_{timestamp}.zip"
            if hasattr(agent, 'model') and agent.model is not None:
                agent.model.save(interrupted_path)
                print(f"Progress saved to: {interrupted_path}")
    
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    
    finally:
        # Cleanup
        if 'agent' in locals():
            agent.close()
        print("Cleanup completed")


def quick_test():
    """Quick test to verify everything works"""
    print("Running quick test...")
    
    setup_colab_environment()
    
    # Test environment creation
    from src.environments.humanoid_env import make_humanoid_env
    
    env = make_humanoid_env("walking")
    obs, info = env.reset()
    
    print(f"Environment test passed!")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")
    
    env.close()


# For running in Colab cells
def train_walking(timesteps=500000, device='auto'):
    """Simple function to call from Colab cell"""
    config = {
        'total_timesteps': timesteps,
        'device': device,
        'learning_rate': 0.0003,
        'batch_size': 64,
        'save_freq': 25000,
        'eval_freq': 10000,
        'verbose': 1,
        'log_dir': f"data/logs/walking_{datetime.now().strftime('%m%d_%H%M')}"
    }
    
    setup_colab_environment()
    
    agent = WalkingAgent(config)
    model = agent.train()
    results = agent.evaluate(n_episodes=3)
    agent.close()
    
    return model, results


if __name__ == "__main__":
    # If you want to run the full script
    main()