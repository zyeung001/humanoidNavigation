"""
Diagnostic: Print the ACTUAL reward calculation line by line
This will show us what's happening inside _compute_task_reward
"""
import sys
import os
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environments.standing_env import make_standing_env

def diagnose_reward_calculation():
    """Manually trace through reward calculation"""
    print("=" * 70)
    print("REWARD CALCULATION DIAGNOSIS")
    print("=" * 70)
    
    config = {'target_height': 1.3, 'max_episode_steps': 5000}
    env = make_standing_env(render_mode=None, config=config)
    
    # Reset and take one step with maximum action
    obs, _ = env.reset()
    action = np.ones(env.action_space.shape)  # All actions = 1.0
    
    print("\n🔍 Taking ONE step with maximum actions...")
    print(f"Action shape: {action.shape}")
    print(f"Action values: all 1.0")
    print(f"Action squared sum: {np.sum(np.square(action)):.2f}")
    print(f"Expected control cost: -0.5 × {np.sum(np.square(action)):.2f} = {-0.5 * np.sum(np.square(action)):.2f}")
    
    # Take the step
    obs, reward, term, trunc, info = env.step(action)
    
    # Extract state
    height = env.env.unwrapped.data.qpos[2]
    quat = env.env.unwrapped.data.qpos[3:7]
    height_error = abs(height - 1.3)
    
    print("\n📊 State after step:")
    print(f"  Height: {height:.3f} (target: 1.3, error: {height_error:.3f})")
    print(f"  Quaternion w: {quat[0]:.3f}")
    
    print("\n🧮 Manual reward calculation:")
    height_reward = 30.0 * np.exp(-10.0 * height_error**2)
    upright_reward = 10.0 * max(0.0, (quat[0] - 0.5) / 0.5)
    control_cost = -0.5 * np.sum(np.square(action))
    manual_total = height_reward + upright_reward + control_cost
    
    print(f"  Height reward:  {height_reward:6.2f}")
    print(f"  Upright reward: {upright_reward:6.2f}")
    print(f"  Control cost:   {control_cost:6.2f}")
    print(f"  Manual total:   {manual_total:6.2f}")
    
    print(f"\n🎯 Actual reward from env: {reward:.2f}")
    print(f"\n⚠️  DISCREPANCY: {abs(reward - manual_total):.2f}")
    
    if abs(reward - manual_total) > 0.1:
        print("\n❌ PROBLEM DETECTED!")
        print("The environment is NOT using the simplified reward function!")
        print("\nPossible causes:")
        print("1. Changes to standing_env.py were not saved")
        print("2. Python is using cached/old version of the module")
        print("3. The _compute_task_reward method wasn't replaced properly")
        print("\n📝 SOLUTION:")
        print("1. Open src/environments/standing_env.py")
        print("2. Find _compute_task_reward method")
        print("3. Add this line RIGHT AT THE START of the method:")
        print("   print(f'DEBUG: Using simplified reward, action sum = {np.sum(np.square(action)):.2f}')")
        print("4. Save and run this test again")
        print("5. If you don't see that print, the method isn't being called")
    else:
        print("\n✅ Reward calculation is correct!")
        print("But if Test 4 is still not negative, check the termination logic.")
    
    env.close()

if __name__ == "__main__":
    diagnose_reward_calculation()