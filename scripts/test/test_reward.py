"""
test_reward.py
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.environments.standing_env import make_standing_env

def test_actual_rewards():
    """Test rewards with actual environment dynamics"""
    
    print("=" * 80)
    print("TESTING ACTUAL REWARD FUNCTION")
    print("=" * 80)
    
    # Create environment
    config = {'target_height': 1.25, 'max_episode_steps': 500}
    env = make_standing_env(render_mode=None, config=config)
    
    # Test 1: Random actions for 100 steps
    print("\nTest 1: Random actions")
    print("-" * 40)
    obs, _ = env.reset()
    total_reward = 0
    heights = []
    rewards = []
    
    for step in range(100):
        action = env.action_space.sample() * 0.1  # Small random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        heights.append(info['height'])
        rewards.append(reward)
        
        if step % 20 == 0:
            print(f"Step {step}: height={info['height']:.3f}, reward={reward:.1f}")
        
        if terminated:
            print(f"Terminated at step {step}")
            break
    
    print(f"Average height: {np.mean(heights):.3f}")
    print(f"Average reward per step: {np.mean(rewards):.1f}")
    print(f"Total reward: {total_reward:.1f}")
    
    # Test 2: Zero actions (should maintain initial position)
    print("\nTest 2: Zero actions (testing stability)")
    print("-" * 40)
    obs, _ = env.reset()
    total_reward = 0
    heights = []
    
    for step in range(100):
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        heights.append(info['height'])
        
        if step % 20 == 0:
            print(f"Step {step}: height={info['height']:.3f}, reward={reward:.1f}")
        
        if terminated:
            print(f"Terminated at step {step}")
            break
    
    print(f"Height range: {np.min(heights):.3f} to {np.max(heights):.3f}")
    print(f"Height stability (std): {np.std(heights):.4f}")
    
    # Test 3: What's the initial height?
    print("\nTest 3: Initial state analysis")
    print("-" * 40)
    for i in range(3):
        obs, info = env.reset()
        print(f"Reset {i+1}: initial height = {info['height']:.3f}")
    
    env.close()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey findings:")
    print("1. Natural fall behavior with zero actions")
    print("2. Reward magnitude per step")
    print("3. How quickly termination occurs")
    print("4. Whether 1.25m is a reasonable target")

if __name__ == "__main__":
    test_actual_rewards()