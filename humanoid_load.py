# First, install required packages and set up virtual display
!apt-get update > /dev/null 2>&1
!apt-get install -y xvfb python-opengl > /dev/null 2>&1

# Set up virtual display
import os
os.environ['DISPLAY'] = ':99'

# Start virtual display in background
!Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Wait a moment for display to start
import time
time.sleep(2)

# Now you can use your gymnasium code
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from IPython.display import Video
import numpy as np

# Create environment - now it should work
env = gym.make("Humanoid-v5", render_mode="rgb_array")
env = RecordVideo(env, video_folder="./videos/", episode_trigger=lambda x: True)

obs, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()

# Display the video
import os
video_files = [f for f in os.listdir("./videos/") if f.endswith('.mp4')]
if video_files:
    display(Video(f"./videos/{video_files[0]}", width=640, height=480))