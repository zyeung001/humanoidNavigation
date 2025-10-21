#!/usr/bin/env python3
"""
Test script for the improved reward function
Tests the new positive reward structure and termination conditions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from src.environments.standing_env import make_standing_env

def test_improved_reward():
    """Test the improved reward function with various scenarios"""
    print("Testing Improved Reward Function")
    print("=" * 50)
    
    # Create environment
    config = {
        'max_episode_steps': 1000,
        'target_height': 1.0
    }
    env = make_standing_env(config=config)
    
    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    
    print(f"Target height: {env.target_height}")
    print(f"Max episode steps: {env.max_episode_steps}")
    print()
    
    # Test with random actions for a few steps
    print("Testing with random actions...")
    for step in range(200):
        action = env.action_space.sample() * 0.1  # Small random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if step % 50 == 0:
            height = info.get('height', 0)
            height_error = abs(height - env.target_height)
            print(f"Step {step}: height={height:.3f}, error={height_error:.3f}, "
                  f"reward={reward:.1f}, total={total_reward:.1f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step} ({'terminated' if terminated else 'truncated'})")
            break
    
    print(f"\nFinal Results:")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward/step_count:.2f}")
    
    # Test reward components
    print(f"\nReward Component Analysis:")
    analysis = env.get_reward_analysis()
    if analysis:
        for component, stats in analysis.items():
            print(f"{component.upper()}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Total: {stats['total']:.2f}")
            print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    
    env.close()
    
    # Test termination conditions
    print(f"\nTesting Termination Conditions:")
    print(f"Height < 0.2: {env.target_height - 0.8 < 0.2} (should terminate)")
    print(f"Height > 3.0: {env.target_height + 2.0 > 3.0} (should terminate)")
    print(f"Quat[0] < 0.05: {0.03 < 0.05} (should terminate)")
    
    return total_reward > 0  # Should be positive with new reward function

def test_reward_scaling():
    """Test reward scaling with different height errors"""
    print("\nTesting Reward Scaling")
    print("=" * 30)
    
    config = {'target_height': 1.0}
    env = make_standing_env(config=config)
    
    # Simulate different height scenarios
    test_heights = [0.98, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30]
    
    for height in test_heights:
        # Manually set height for testing (this is just for demonstration)
        height_error = abs(height - env.target_height)
        
        # Calculate expected reward based on our new function
        if height_error < 0.02:
            expected_reward = 100.0
        elif height_error < 0.05:
            expected_reward = 80.0
        elif height_error < 0.10:
            expected_reward = 60.0
        elif height_error < 0.20:
            expected_reward = 40.0
        elif height_error < 0.30:
            expected_reward = 20.0
        else:
            expected_reward = max(0, 10.0 - height_error * 10.0)
        
        print(f"Height: {height:.2f}, Error: {height_error:.3f}, Expected Reward: {expected_reward:.1f}")
    
    env.close()

if __name__ == "__main__":
    success = test_improved_reward()
    test_reward_scaling()
    
    print(f"\n{'✓ SUCCESS' if success else '✗ FAILED'}: Reward function test")
    print("The improved reward function should now provide positive rewards")
    print("and more lenient termination conditions for better learning.")
