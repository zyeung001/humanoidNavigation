#!/usr/bin/env python3
"""
analyze_rewards.py

Diagnostic script to analyze reward component balance in the walking environment.
Helps verify that velocity tracking is now the dominant objective.

Usage:
    python scripts/debug/analyze_rewards.py [--steps 1000] [--speed 0.3]
"""

import os
import sys
import argparse
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environments.walking_curriculum import make_walking_curriculum_env


def analyze_reward_components(env, n_steps: int = 1000, verbose: bool = True):
    """
    Run environment with random actions and analyze reward components.
    
    Returns breakdown of reward sources to verify balance.
    """
    obs, info = env.reset()
    
    # Track rewards by component
    tracking_rewards = []
    height_rewards = []
    upright_rewards = []
    stability_rewards = []
    
    # Track other metrics
    velocity_errors = []
    heights = []
    actual_speeds = []
    commanded_speeds = []
    episode_lengths = []
    current_ep_length = 0
    total_rewards = []
    
    print(f"\n{'='*70}")
    print("REWARD COMPONENT ANALYSIS")
    print(f"{'='*70}")
    print(f"Running {n_steps} steps with random actions...")
    print(f"Commanded speed range: 0 - {env.max_commanded_speed:.2f} m/s")
    print(f"{'='*70}\n")
    
    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        current_ep_length += 1
        
        # Collect metrics
        total_rewards.append(reward)
        velocity_errors.append(info.get('velocity_error', 0))
        heights.append(info.get('height', 1.4))
        actual_speeds.append(info.get('actual_speed', 0))
        commanded_speeds.append(info.get('commanded_speed', 0))
        
        # Get reward components from environment history
        if hasattr(env, 'reward_history'):
            if env.reward_history.get('velocity_tracking'):
                tracking_rewards.append(env.reward_history['velocity_tracking'][-1])
            if env.reward_history.get('height'):
                height_rewards.append(env.reward_history['height'][-1])
            if env.reward_history.get('upright'):
                upright_rewards.append(env.reward_history['upright'][-1])
            if env.reward_history.get('velocity'):
                stability_rewards.append(env.reward_history['velocity'][-1])
        
        done = terminated or truncated
        if done:
            episode_lengths.append(current_ep_length)
            current_ep_length = 0
            obs, info = env.reset()
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print("\n📊 REWARD COMPONENT BREAKDOWN:")
    print("-"*50)
    
    if tracking_rewards:
        print(f"  Velocity Tracking: {np.mean(tracking_rewards):+8.2f} ± {np.std(tracking_rewards):.2f} per step")
    if height_rewards:
        print(f"  Height Reward:     {np.mean(height_rewards):+8.2f} ± {np.std(height_rewards):.2f} per step")
    if upright_rewards:
        print(f"  Upright Reward:    {np.mean(upright_rewards):+8.2f} ± {np.std(upright_rewards):.2f} per step")
    if stability_rewards:
        print(f"  Stability Reward:  {np.mean(stability_rewards):+8.2f} ± {np.std(stability_rewards):.2f} per step")
    print("  -"*25)
    print(f"  TOTAL:             {np.mean(total_rewards):+8.2f} ± {np.std(total_rewards):.2f} per step")
    
    # Verify tracking dominance
    print("\n🎯 TRACKING DOMINANCE CHECK:")
    print("-"*50)
    
    if tracking_rewards and height_rewards and upright_rewards:
        tracking_pct = abs(np.mean(tracking_rewards)) / (
            abs(np.mean(tracking_rewards)) + 
            abs(np.mean(height_rewards)) + 
            abs(np.mean(upright_rewards)) + 
            abs(np.mean(stability_rewards)) + 0.001
        ) * 100
        
        survival_sum = abs(np.mean(height_rewards)) + abs(np.mean(upright_rewards)) + abs(np.mean(stability_rewards))
        
        print(f"  Tracking reward percentage:  {tracking_pct:.1f}%")
        print(f"  Tracking / Survival ratio:   {abs(np.mean(tracking_rewards)) / (survival_sum + 0.001):.2f}x")
        
        if tracking_pct >= 50:
            print(f"  ✅ GOOD: Tracking is dominant ({tracking_pct:.0f}% of positive rewards)")
        elif tracking_pct >= 35:
            print(f"  ⚠️ MARGINAL: Tracking is significant but not dominant ({tracking_pct:.0f}%)")
        else:
            print(f"  ❌ ISSUE: Survival rewards still dominate ({100-tracking_pct:.0f}%)")
    
    print("\n📈 PERFORMANCE METRICS:")
    print("-"*50)
    print(f"  Avg velocity error:    {np.mean(velocity_errors):.3f} m/s")
    print(f"  Avg height:            {np.mean(heights):.3f} m")
    print(f"  Avg actual speed:      {np.mean(actual_speeds):.3f} m/s")
    print(f"  Avg commanded speed:   {np.mean(commanded_speeds):.3f} m/s")
    print(f"  Avg episode length:    {np.mean(episode_lengths) if episode_lengths else current_ep_length:.0f} steps")
    print(f"  Episodes completed:    {len(episode_lengths)}")
    
    # Curriculum info
    if hasattr(env, 'stage'):
        print("\n📚 CURRICULUM STATUS:")
        print("-"*50)
        print(f"  Current stage:         {env.stage}")
        print(f"  Max commanded speed:   {env.max_commanded_speed:.2f} m/s")
        print(f"  Velocity tolerance:    {env.velocity_tolerances[env.stage]:.2f} m/s")
        print(f"  Min episode length:    {env.min_episode_lengths[env.stage]} steps")
        if hasattr(env, 'success_buffer') and env.success_buffer:
            print(f"  Recent success rate:   {np.mean(env.success_buffer):.1%}")
    
    print("\n" + "="*70)
    
    return {
        'tracking_mean': np.mean(tracking_rewards) if tracking_rewards else 0,
        'height_mean': np.mean(height_rewards) if height_rewards else 0,
        'upright_mean': np.mean(upright_rewards) if upright_rewards else 0,
        'total_mean': np.mean(total_rewards),
        'velocity_error_mean': np.mean(velocity_errors),
        'actual_speed_mean': np.mean(actual_speeds),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze walking reward components")
    parser.add_argument('--steps', type=int, default=2000, help='Number of steps to run')
    parser.add_argument('--speed', type=float, default=0.3, help='Max commanded speed')
    parser.add_argument('--stage', type=int, default=0, help='Curriculum stage')
    args = parser.parse_args()
    
    # Create config matching training setup
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.2,
        'random_height_init': False,
        'max_commanded_speed': args.speed,
        'curriculum_start_stage': args.stage,
        'curriculum_max_stage': 6,
        'push_enabled': False,  # Disable for analysis
        'domain_rand': False,
        
        # Reward weights (should match new config)
        'reward_tracking_weight': 25.0,
        'reward_tracking_bandwidth': 2.0,
        'reward_direction_weight': 8.0,
        'reward_height_weight': 3.0,
        'reward_upright_weight': 2.0,
        'reward_alive_weight': 0.5,
        'reward_consistency_weight': 8.0,
    }
    
    print("Creating walking curriculum environment...")
    env = make_walking_curriculum_env(render_mode=None, config=config)
    
    try:
        analyze_reward_components(env, n_steps=args.steps)
    finally:
        env.close()
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()

