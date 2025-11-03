#!/usr/bin/env python3
"""
Fixed diagnostic script for standing model with proper environment matching.
This version creates the environment EXACTLY as it was during training.
"""
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

# Import the CORRECT environment - the one used during training
from src.environments.standing_curriculum_env import make_standing_curriculum_env

def main():
    parser = argparse.ArgumentParser(description="Diagnose standing model performance")
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/saved_models/final_standing_model.zip",
        help="Path to model file"
    )
    parser.add_argument(
        "--vecnorm",
        type=str,
        default="models/saved_models/vecnorm.pkl",
        help="Path to VecNormalize stats"
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
    
    # Create environment config matching training setup
    training_config = {
        'target_height': args.target_height,
        'max_episode_steps': 2000,
        # Features used during training
        'obs_history': 4,              # History stacking
        'obs_include_com': True,       # COM features
        'obs_feature_norm': True,      # Feature normalization
        'action_smoothing': True,      # Action smoothing
        'action_smoothing_tau': 0.2,   # Smoothing parameter
        # Curriculum settings (stay at final stage for inference)
        'curriculum_start_stage': 3,   # Start at final stage
        'curriculum_max_stage': 3,     # Stay at final stage
    }
    
    print("Creating curriculum environment with training configuration...")
    print(f"  - History stacking: {training_config['obs_history']} frames")
    print(f"  - COM features: {training_config['obs_include_com']}")
    print(f"  - Action smoothing: {training_config['action_smoothing']}")
    print(f"  - Curriculum stage: {training_config['curriculum_start_stage']}\n")
    
    # Create base environment - MUST reset before wrapping in VecEnv!
    base_env = make_standing_curriculum_env(render_mode=None, config=training_config)
    
    # CRITICAL: Reset the environment first to freeze observation space
    _ = base_env.reset()
    print(f"✓ Environment observation space: {base_env.observation_space.shape}")
    
    # Now wrap in VecEnv
    vec_env = DummyVecEnv([lambda: base_env])
    
    # Load VecNormalize stats if they exist
    try:
        env = VecNormalize.load(args.vecnorm, vec_env)
        env.training = False
        env.norm_reward = False
        print("✓ Loaded VecNormalize stats")
    except Exception as e:
        env = vec_env
        print(f"⚠ No VecNormalize found: {e}")
        print("  Continuing without normalization...")
    
    # Load model
    try:
        model = PPO.load(args.model, env=env)
        print(f"✓ Model loaded successfully")
        print(f"  Model observation space: {model.observation_space.shape}\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nRun with --list-models to see available models")
        if env is not None:
            env.close()
        return
    
    # Run evaluation
    print("Running evaluation...")
    obs = env.reset()
    heights = []
    actions_taken = []
    rewards_collected = []
    
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Extract metrics from vectorized env
        try:
            # Get the unwrapped environment to access raw state
            unwrapped = env.envs[0]
            while hasattr(unwrapped, 'env'):
                unwrapped = unwrapped.env
            
            # Get height from the MuJoCo data
            height = unwrapped.data.qpos[2]
            heights.append(height)
            actions_taken.append(np.abs(action).mean())
            rewards_collected.append(reward[0] if isinstance(reward, np.ndarray) else reward)
        except Exception as e:
            print(f"Warning at step {step}: Could not extract metrics - {e}")
            break
        
        if done[0]:  # VecEnv returns array
            print(f"Episode ended at step {step}")
            break
    
    env.close()
    
    if len(heights) == 0:
        print("❌ No data collected. Check environment configuration.")
        return
    
    # Create comprehensive plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Height trajectory
    ax1 = axes[0]
    ax1.plot(heights, linewidth=2, label='Actual height', color='blue')
    ax1.axhline(y=args.target_height, color='r', linestyle='--', 
                label=f'Target ({args.target_height}m)', linewidth=2)
    ax1.axhline(y=args.target_height - 0.05, color='g', linestyle=':', 
                label=f'Acceptable range (±5cm)', linewidth=1, alpha=0.5)
    ax1.axhline(y=args.target_height + 0.05, color='g', linestyle=':', 
                linewidth=1, alpha=0.5)
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Height (m)', fontsize=12)
    ax1.set_title('Height Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Action magnitude
    ax2 = axes[1]
    ax2.plot(actions_taken, linewidth=1.5, label='Mean |action|', color='orange')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Action Magnitude', fontsize=12)
    ax2.set_title('Action Magnitude Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reward
    ax3 = axes[2]
    ax3.plot(rewards_collected, linewidth=1.5, label='Reward', color='green')
    ax3.set_xlabel('Step', fontsize=12)
    ax3.set_ylabel('Reward', fontsize=12)
    ax3.set_title('Reward Per Step', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {args.output}")
    
    # Calculate metrics
    mean_height = np.mean(heights)
    std_height = np.std(heights)
    height_error = abs(mean_height - args.target_height)
    mean_action = np.mean(actions_taken)
    total_reward = np.sum(rewards_collected)
    mean_reward = np.mean(rewards_collected)
    
    # Count time in acceptable range
    in_range = sum(1 for h in heights if abs(h - args.target_height) < 0.05)
    percent_in_range = (in_range / len(heights)) * 100
    
    print(f"\n{'='*50}")
    print(f"{'DIAGNOSTIC RESULTS':^50}")
    print(f"{'='*50}")
    
    print(f"\n📊 Height Metrics:")
    print(f"   Mean height:           {mean_height:.3f}m")
    print(f"   Target height:         {args.target_height:.3f}m")
    print(f"   Height error:          {height_error:.3f}m ({height_error*100:.1f}cm)")
    print(f"   Std deviation:         {std_height:.3f}m")
    print(f"   Time in range (±5cm):  {percent_in_range:.1f}%")
    
    print(f"\n🎮 Action Metrics:")
    print(f"   Mean action magnitude: {mean_action:.3f}")
    print(f"   Std action magnitude:  {np.std(actions_taken):.3f}")
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Episode length:        {len(heights)} steps")
    print(f"   Total reward:          {total_reward:.1f}")
    print(f"   Mean reward:           {mean_reward:.2f}")
    
    # Assessment
    print(f"\n{'='*50}")
    print(f"{'ASSESSMENT':^50}")
    print(f"{'='*50}\n")
    
    # Height assessment
    if height_error < 0.05:
        print("✅ Height: EXCELLENT (within 5cm of target)")
    elif height_error < 0.10:
        print("⚠️  Height: GOOD (within 10cm of target)")
    elif height_error < 0.20:
        print("⚠️  Height: FAIR (within 20cm of target)")
    else:
        print(f"❌ Height: POOR (error {height_error*100:.1f}cm)")
    
    # Stability assessment
    if std_height < 0.05:
        print("✅ Stability: EXCELLENT (std < 5cm)")
    elif std_height < 0.08:
        print("⚠️  Stability: GOOD (std < 8cm)")
    elif std_height < 0.12:
        print("⚠️  Stability: FAIR (std < 12cm)")
    else:
        print(f"❌ Stability: POOR (std {std_height*100:.1f}cm)")
    
    # Consistency assessment
    if percent_in_range > 80:
        print(f"✅ Consistency: EXCELLENT ({percent_in_range:.1f}% in range)")
    elif percent_in_range > 60:
        print(f"⚠️  Consistency: GOOD ({percent_in_range:.1f}% in range)")
    elif percent_in_range > 40:
        print(f"⚠️  Consistency: FAIR ({percent_in_range:.1f}% in range)")
    else:
        print(f"❌ Consistency: POOR ({percent_in_range:.1f}% in range)")
    
    # Episode length assessment
    if len(heights) >= 500:
        print("✅ Duration: EXCELLENT (completed full episode)")
    elif len(heights) > 400:
        print("⚠️  Duration: GOOD (survived > 400 steps)")
    elif len(heights) > 200:
        print("⚠️  Duration: FAIR (survived > 200 steps)")
    else:
        print(f"❌ Duration: POOR (only {len(heights)} steps)")
    
    print(f"\n{'='*50}\n")

if __name__ == "__main__":
    main()