import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from IPython.display import Video
import os

# Create environment with video recording wrapper
env = gym.make("Humanoid-v5", render_mode="rgb_array")
env = RecordVideo(env, video_folder="./videos/", episode_trigger=lambda x: True)

obs, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()

# Find the video file
video_files = [f for f in os.listdir("./videos/") if f.endswith('.mp4')]
if video_files:
    Video(f"./videos/{video_files[0]}", width=640, height=480)