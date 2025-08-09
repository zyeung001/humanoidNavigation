import gymnasium as gym
import numpy as np

env = gym.make("Humanoid-v5", render_mode="human")

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() 
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()