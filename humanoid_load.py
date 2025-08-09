import subprocess
import os
import time
import gymnasium as gym
import numpy as np
import cv2

def setup_display():
    """Setup virtual display for headless environment"""
    try:
        # Install packages (only works if you have sudo access)
        subprocess.run(['apt-get', 'update'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['apt-get', 'install', '-y', 'xvfb', 'python-opengl'], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        os.environ['DISPLAY'] = ':99'
        os.environ['MUJOCO_GL'] = 'osmesa'
        
        subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        print("Display setup complete")
        
    except Exception as e:
        print(f"Error setting up display: {e}")

def main():
    # Setup display
    setup_display()
    
    # Create environment
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('humanoid_simulation.mp4', fourcc, 30.0, (640, 480))
    
    obs, info = env.reset()
    
    print("Starting simulation...")
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get frame and save to video
        try:
            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_resized = cv2.resize(frame_bgr, (640, 480))
            out.write(frame_resized)
        except Exception as e:
            print(f"Rendering error at step {i}: {e}")
            break
        
        if terminated or truncated:
            obs, info = env.reset()
        
        if i % 100 == 0:
            print(f"Step {i}/500")
    
    env.close()
    out.release()
    
    print("Video saved as 'humanoid_simulation.mp4'")

if __name__ == "__main__":
    main()