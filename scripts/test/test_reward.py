# test_reward.py - Run this BEFORE training to check reward function
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
    ]
    
    for name, height, quat_w, vel, ang_vel in scenarios:
        env.env.unwrapped.data.qpos[2] = height
        env.env.unwrapped.data.qpos[3] = quat_w
        env.env.unwrapped.data.qvel[0:3] = vel
        env.env.unwrapped.data.qvel[3:6] = ang_vel
        
        action = np.zeros(env.action_space.shape)
        _, reward, _, _, _ = env.step(action)
        print(f"{name:30s}: {reward:8.2f}")
    
    env.close()

if __name__ == "__main__":
    test_reward_scenarios()