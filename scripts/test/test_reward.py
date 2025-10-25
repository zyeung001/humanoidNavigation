import sys
import os

# Add the project root to Python path for Google Colab compatibility
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environments.standing_env import make_standing_env
import numpy as np

def test_new_reward():
    """Test the new reward function"""
    config = {'target_height': 1.3, 'max_episode_steps': 5000}
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
    print(f"  Average reward: {total/10:.1f} (should be ~35-50)")
    
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
    
    # Test 4: Test reward components breakdown
    print("\nTest 4: Reward component analysis")
    obs, _ = env.reset()
    obs, reward, term, trunc, info = env.step(np.zeros(env.action_space.shape))
    
    # Get detailed info about the state
    height = info.get('height', 0)
    quat_w = info.get('quaternion_w', 0)
    x_vel = info.get('x_velocity', 0)
    y_vel = info.get('y_velocity', 0)
    z_vel = info.get('z_velocity', 0)
    
    print(f"  Height: {height:.3f} (target: 1.3)")
    print(f"  Quaternion w: {quat_w:.3f}")
    print(f"  Velocities: x={x_vel:.3f}, y={y_vel:.3f}, z={z_vel:.3f}")
    print(f"  Reward: {reward:.3f}")
    
    # Calculate expected reward components
    height_error = abs(height - 1.3)
    expected_height_reward = 20.0 * np.exp(-10.0 * height_error**2)
    expected_upright = 10.0 * quat_w if quat_w > 0 else 0
    expected_alive = 5.0
    xy_vel = np.sqrt(x_vel**2 + y_vel**2)
    expected_vel_penalty = -2.0 * xy_vel - 3.0 * abs(z_vel)
    
    print(f"  Expected components:")
    print(f"    Alive: {expected_alive:.1f}")
    print(f"    Height: {expected_height_reward:.1f}")
    print(f"    Upright: {expected_upright:.1f}")
    print(f"    Vel penalty: {expected_vel_penalty:.1f}")
    
    env.close()
    print("\n✅ If perfect standing gives 35-50 and large actions give negative, the reward is fixed!")

if __name__ == "__main__":
    test_new_reward()