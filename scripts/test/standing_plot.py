import sys
import os
import argparse

# Add project root to path
project_root = '/content/humanoidNavigation'
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from src.environments.standing_env import make_standing_env

def main():
    parser = argparse.ArgumentParser(description="Diagnose standing model performance")
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/saved_models/best_standing_model.zip",
        help="Path to model file (default: models/saved_models/best_standing_model.zip)"
    )
    parser.add_argument(
        "--vecnorm",
        type=str,
        default="models/saved_models/vecnorm_standing.pkl",
        help="Path to VecNormalize stats (default: models/saved_models/vecnorm_standing.pkl)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of steps to run (default: 500)"
    )
    parser.add_argument(
        "--target-height",
        type=float,
        default=1.3,
        help="Target height in meters (default: 1.3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="height_trajectory.png",
        help="Output plot filename (default: height_trajectory.png)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model files and exit"
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("=== Available Models ===")
        model_dir = "models/saved_models"
        checkpoint_dir = "data/checkpoints"
        
        if os.path.exists(model_dir):
            print(f"\nIn {model_dir}:")
            for f in sorted(os.listdir(model_dir)):
                size = os.path.getsize(os.path.join(model_dir, f)) / 1024
                print(f"  - {f} ({size:.1f} KB)")
        
        if os.path.exists(checkpoint_dir):
            print(f"\nIn {checkpoint_dir}:")
            for f in sorted(os.listdir(checkpoint_dir)):
                if f.endswith('.zip'):
                    size = os.path.getsize(os.path.join(checkpoint_dir, f)) / 1024
                    print(f"  - {f} ({size:.1f} KB)")
        return
    
    print(f"=== Running Diagnostic ===")
    print(f"Model: {args.model}")
    print(f"VecNormalize: {args.vecnorm}")
    print(f"Steps: {args.steps}")
    print(f"Target height: {args.target_height}m\n")
    
    # Load model
    config = {'target_height': args.target_height, 'max_episode_steps': 2000}
    
    # Create base environment
    base_env = DummyVecEnv([lambda: make_standing_env(render_mode=None, config=config)])
    
    # Load VecNormalize stats if they exist
    try:
        env = VecNormalize.load(args.vecnorm, base_env)
        env.training = False
        env.norm_reward = False
        print("✓ Loaded VecNormalize")
    except Exception as e:
        env = base_env
        print(f"⚠ No VecNormalize found: {e}")
    
    # Load model
    try:
        model = PPO.load(args.model, env=env)
        print(f"✓ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nRun with --list-models to see available models")
        return
    
    # Run evaluation
    obs = env.reset()
    heights = []
    actions_taken = []
    
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Extract height from vectorized env
        heights.append(env.envs[0].unwrapped.data.qpos[2])
        actions_taken.append(np.abs(action).mean())
        
        if done[0]:  # VecEnv returns array
            print(f"Episode ended at step {step}")
            break
    
    # Plot height over time
    plt.figure(figsize=(12, 6))
    plt.plot(heights, linewidth=2, label='Actual height')
    plt.axhline(y=args.target_height, color='r', linestyle='--', 
                label=f'Target ({args.target_height}m)', linewidth=2)
    plt.axhline(y=args.target_height - 0.05, color='g', linestyle=':', 
                label=f'Acceptable range', linewidth=1, alpha=0.5)
    plt.axhline(y=args.target_height + 0.05, color='g', linestyle=':', 
                linewidth=1, alpha=0.5)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Height (m)', fontsize=12)
    plt.title('Height trajectory of trained model', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {args.output}")
    
    # Results
    mean_height = np.mean(heights)
    std_height = np.std(heights)
    height_error = abs(mean_height - args.target_height)
    
    print(f"\n=== DIAGNOSTIC RESULTS ===")
    print(f"Mean action magnitude: {np.mean(actions_taken):.3f}")
    print(f"Std action magnitude: {np.std(actions_taken):.3f}")
    print(f"Mean height: {mean_height:.3f}m")
    print(f"Std height (stability): {std_height:.3f}m")
    print(f"Height error from {args.target_height}m: {height_error:.3f}m")
    print(f"Episode length: {len(heights)} steps")
    
    # Check if standing well
    print(f"\n=== ASSESSMENT ===")
    if height_error < 0.05:
        print("✓ Height: GOOD (within 5cm of target)")
    elif height_error < 0.10:
        print("⚠ Height: OK (within 10cm of target)")
    else:
        print(f"❌ Height: POOR (error {height_error:.3f}m)")
    
    if std_height < 0.08:
        print("✓ Stability: GOOD (std < 0.08)")
    elif std_height < 0.12:
        print("⚠ Stability: OK (std < 0.12)")
    else:
        print(f"❌ Stability: POOR (std {std_height:.3f})")
    
    if len(heights) > 400:
        print("✓ Episode length: GOOD (survived > 400 steps)")
    elif len(heights) > 200:
        print("⚠ Episode length: OK (survived > 200 steps)")
    else:
        print(f"❌ Episode length: POOR (only {len(heights)} steps)")
    
    env.close()

if __name__ == "__main__":
    main()