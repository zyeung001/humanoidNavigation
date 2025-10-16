# test_reward.py - Run this BEFORE training to check reward function
import sys
import os

# Add project root to path
project_root = '/content/humanoidNavigation'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from src.environments.standing_env import make_standing_env

def test_reward_scenarios():
    """Test reward function in different scenarios"""
    env = make_standing_env(render_mode=None, config={'max_episode_steps': 100})
    env.reset()
    
    scenarios = [
        ("Perfect standing", 1.3, 1.0, [0,0,0], [0,0,0]),
        ("Slightly low", 1.28, 1.0, [0,0,0], [0,0,0]),
        ("Too low", 1.2, 1.0, [0,0,0], [0,0,0]),
        ("Perfect height but moving", 1.3, 1.0, [0.1,0.1,0], [0,0,0]),
        ("Perfect height but tilted", 1.3, 0.95, [0,0,0], [0,0,0]),
        ("Good height but drifting", 1.29, 1.0, [0.05,0.05,0], [0,0,0]),
        ("Falling", 0.8, 0.7, [0,0,-0.5], [0,0,0]),
    ]
    
    print("\n" + "="*60)
    print("REWARD FUNCTION TEST")
    print("="*60)
    print(f"{'Scenario':<30} {'Reward':>10}")
    print("-"*60)
    
    for name, height, quat_w, vel, ang_vel in scenarios:
        env.env.unwrapped.data.qpos[2] = height
        env.env.unwrapped.data.qpos[3] = quat_w
        env.env.unwrapped.data.qvel[0:3] = vel
        env.env.unwrapped.data.qvel[3:6] = ang_vel
        env.env.unwrapped.data.qpos[0:2] = [0, 0]  # Reset position
        
        action = np.zeros(env.action_space.shape)
        _, reward, terminated, _, _ = env.step(action)
        
        status = " [TERM]" if terminated else ""
        print(f"{name:<30}: {reward:10.2f}{status}")
    
    print("="*60)
    print("\nInterpretation:")
    print("- Perfect standing should give highest reward (~150)")
    print("- Any movement should reduce reward significantly")
    print("- Wrong height should give negative or very low reward")
    print("- Falling should terminate\n")
    
    env.close()

if __name__ == "__main__":
    test_reward_scenarios()