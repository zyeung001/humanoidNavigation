# test_reward_debug.py - Debug version to see what's actually happening
import sys
import os

project_root = '/content/humanoidNavigation'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.environments.standing_env import make_standing_env

def test_reward_debug():
    """Debug test to see actual values"""
    env = make_standing_env(render_mode=None, config={'max_episode_steps': 100})
    env.reset()
    
    print("\n" + "="*60)
    print("DEBUG: Testing reward calculation")
    print("="*60)
    
    # Test 1: Perfect standing
    print("\nTest 1: Perfect standing")
    env.env.unwrapped.data.qpos[2] = 1.3
    env.env.unwrapped.data.qpos[3] = 1.0
    env.env.unwrapped.data.qvel[:] = 0
    
    # Check actual values BEFORE step
    print(f"Before step:")
    print(f"  Height: {env.env.unwrapped.data.qpos[2]}")
    print(f"  Quat[0]: {env.env.unwrapped.data.qpos[3]}")
    print(f"  Velocity: {env.env.unwrapped.data.qvel[0:3]}")
    
    action = np.zeros(env.action_space.shape)
    _, reward, _, _, _ = env.step(action)
    
    # Check actual values AFTER step
    print(f"After step:")
    print(f"  Height: {env.env.unwrapped.data.qpos[2]}")
    print(f"  Quat[0]: {env.env.unwrapped.data.qpos[3]}")
    print(f"  Velocity: {env.env.unwrapped.data.qvel[0:3]}")
    print(f"  Reward: {reward}")
    
    # Test 2: With movement
    print("\n" + "-"*40)
    print("Test 2: Perfect height but moving")
    env.reset()
    env.env.unwrapped.data.qpos[2] = 1.3
    env.env.unwrapped.data.qpos[3] = 1.0
    env.env.unwrapped.data.qvel[0:3] = [0.1, 0.1, 0]
    
    print(f"Set velocity to: {env.env.unwrapped.data.qvel[0:3]}")
    _, reward, _, _, _ = env.step(action)
    print(f"After step velocity: {env.env.unwrapped.data.qvel[0:3]}")
    print(f"Reward: {reward}")
    
    env.close()

if __name__ == "__main__":
    test_reward_debug()