
import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environments.standing_env import make_standing_env

def test_actual_rewards():
    """Test the ACTUAL reward function in the environment"""
    print("=" * 70)
    print("TESTING ACTUAL ENVIRONMENT REWARDS")
    print("=" * 70)
    
    config = {'target_height': 1.3, 'max_episode_steps': 5000}
    env = make_standing_env(render_mode=None, config=config)
    
    # Test 1: Zero actions (should be ~40)
    print("\n1️⃣  TEST: Zero actions (perfect stillness)")
    print("-" * 70)
    obs, _ = env.reset()
    rewards = []
    for i in range(10):
        obs, reward, term, trunc, info = env.step(np.zeros(env.action_space.shape))
        rewards.append(reward)
        if i == 0:
            print(f"   First reward: {reward:.2f}")
            print(f"   Height: {info.get('height', 'N/A'):.3f}")
    avg = np.mean(rewards)
    print(f"   Average: {avg:.2f}")
    print(f"   ✓ PASS" if 35 <= avg <= 45 else f"   ✗ FAIL (expected 35-45)")
    
    # Test 2: Small random actions (should be ~30-35)
    print("\n2️⃣  TEST: Small random actions (gentle corrections)")
    print("-" * 70)
    obs, _ = env.reset()
    rewards = []
    for i in range(10):
        action = np.random.randn(*env.action_space.shape) * 0.1  # Small actions
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        if i == 0:
            print(f"   First reward: {reward:.2f}")
            print(f"   Action magnitude: {np.abs(action).mean():.3f}")
    avg = np.mean(rewards)
    print(f"   Average: {avg:.2f}")
    print(f"   ✓ PASS" if 25 <= avg <= 40 else f"   ✗ FAIL (expected 25-40)")
    
    # Test 3: Large random actions (should be NEGATIVE!)
    print("\n3️⃣  TEST: Large random actions (thrashing)")
    print("-" * 70)
    obs, _ = env.reset()
    rewards = []
    for i in range(10):
        action = env.action_space.sample()  # Full random in [-1, 1]
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        if i == 0:
            print(f"   First reward: {reward:.2f}")
            print(f"   Action magnitude: {np.abs(action).mean():.3f}")
            print(f"   Action squared sum: {np.sum(np.square(action)):.2f}")
        if term:
            print(f"   ⚠️  Terminated at step {i}")
            break
    avg = np.mean(rewards)
    print(f"   Average: {avg:.2f}")
    print(f"   ✓ PASS - NEGATIVE!" if avg < 0 else f"   ✗ FAIL (should be negative, got {avg:.2f})")
    
    # Test 4: Maximum thrashing (should be very negative)
    print("\n4️⃣  TEST: Maximum actions (worst case)")
    print("-" * 70)
    obs, _ = env.reset()
    rewards = []
    for i in range(5):
        action = np.ones(env.action_space.shape)  # All actions = 1.0
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        if i == 0:
            print(f"   First reward: {reward:.2f}")
            print(f"   Action squared sum: {np.sum(np.square(action)):.2f}")
        if term:
            print(f"   ⚠️  Terminated at step {i}")
            break
    avg = np.mean(rewards)
    print(f"   Average: {avg:.2f}")
    print(f"   ✓ PASS - VERY NEGATIVE!" if avg < -10 else f"   ✗ FAIL (should be < -10, got {avg:.2f})")
    
    env.close()
    
    print("\n" + "=" * 70)
    print("REWARD FUNCTION DIAGNOSIS")
    print("=" * 70)
    print("If Test 3 and Test 4 are NOT negative, your reward function is broken!")
    print("The control cost penalty is either:")
    print("  1. Not implemented correctly")
    print("  2. Too weak (coefficient < 0.5)")
    print("  3. Not being applied to the action")
    print("\nExpected behavior:")
    print("  Test 1 (no action):      +35 to +45")
    print("  Test 2 (small action):   +25 to +40")
    print("  Test 3 (random action):  -5 to -20  ⚠️ MUST BE NEGATIVE")
    print("  Test 4 (max action):     -20 to -80 ⚠️ MUST BE VERY NEGATIVE")
    print("=" * 70)

if __name__ == "__main__":
    test_actual_rewards()