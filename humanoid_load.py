import subprocess
import os
import time
import sys

def setup_display():
    """Enhanced setup for MuJoCo rendering"""
    try:
        print("Installing packages...")
        subprocess.run(['apt-get', 'update'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['apt-get', 'install', '-y', 'xvfb', 'python3-opengl', 
                       'mesa-utils', 'libosmesa6-dev', 'freeglut3-dev'], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Set environment variables BEFORE importing gymnasium
        os.environ['DISPLAY'] = ':99'
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        
        print("Starting virtual display...")
        subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait longer for display to initialize
        time.sleep(5)
        
        print("Display setup complete")
        return True
        
    except Exception as e:
        print(f"Error setting up display: {e}")
        return False

def test_environment():
    """Test if the environment can be created and rendered"""
    try:
        import gymnasium as gym
        
        print("Testing environment creation...")
        env = gym.make("Humanoid-v5", render_mode="rgb_array")
        
        print("Testing environment reset...")
        obs, info = env.reset()
        
        print("Testing rendering...")
        frame = env.render()
        
        env.close()
        print(f"Success! Frame shape: {frame.shape}")
        return True
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        return False

def run_simulation():
    """Run the actual simulation"""
    import gymnasium as gym
    import numpy as np
    import cv2
    
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('humanoid_simulation.mp4', fourcc, 30.0, (640, 480))
    
    obs, info = env.reset()
    
    print("Starting simulation...")
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
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

def main():
    print("="*50)
    print("HUMANOID SIMULATION SETUP")
    print("="*50)
    
    # Step 1: Setup display
    if not setup_display():
        print("Failed to setup display. Exiting.")
        return
    
    # Step 2: Test environment
    if not test_environment():
        print("Environment test failed. Check your setup.")
        return
    
    # Step 3: Run simulation
    run_simulation()
    
    print("="*50)
    print("SIMULATION COMPLETE!")
    print("Video file: humanoid_simulation.mp4")
    print("="*50)

if __name__ == "__main__":
    main()