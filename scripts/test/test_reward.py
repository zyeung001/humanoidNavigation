import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environments.standing_env import make_standing_env
import numpy as np

def test_new_reward():
    """Test the new reward function"""
    config = {'target_height': 1.25, 'max_episode_steps': 200}
    env = make_standing_env(render_mode=None, config=config)
    
    print("Testing NEW reward function")
    print("=" * 60)
    
    # Test 1: Perfect standing should give ~60-70 reward
    obs, _ = env.reset()
    
    # Simulate perfect standing (no movement)
    print("\nTest 1: Zero actions (perfect stillness)")
    total = 0
    for i in range(10):
        obs, reward, term, trunc, info = env.step(np.zeros(env.action_space.shape))
        total += reward
        if i == 0:
            print(f"  First step reward: {reward:.1f}")
    print(f"  Average reward: {total/10:.1f} (should be ~40-60)")
    
    # Test 2: Small actions
    print("\nTest 2: Small actions")
    obs, _ = env.reset()
    total = 0
    for i in range(10):
        action = np.random.randn(*env.action_space.shape) * 0.1
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        if i == 0:
            print(f"  First step reward: {reward:.1f}")
    print(f"  Average reward: {total/10:.1f}")
    
    # Test 3: Large actions (should be penalized)
    print("\nTest 3: Large random actions")
    obs, _ = env.reset()
    total = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        if i == 0:
            print(f"  First step reward: {reward:.1f}")
        if term:
            print(f"  Terminated at step {i}")
            break
    print(f"  Average reward: {total/max(1,i):.1f} (should be negative)")
    
    env.close()
    print("\n✅ If perfect standing gives 40-60 and large actions give negative, the reward is fixed!")

if __name__ == "__main__":
    test_new_reward()