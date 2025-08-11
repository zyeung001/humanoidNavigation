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
    import os  # Add this import
    
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    
    # Try different codecs in order of preference
    codecs_to_try = [
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
    ]
    
    out = None
    video_path = '/content/humanoid_simulation.mp4'  # Use absolute path
    
    for codec_name, fourcc in codecs_to_try:
        print(f"Trying codec: {codec_name}")
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        if out.isOpened():
            print(f"Successfully initialized VideoWriter with {codec_name}")
            break
        else:
            print(f"{codec_name} failed")
            out.release()
            out = None
    
    if out is None:
        print("ERROR: Could not initialize any video codec!")
        print("Falling back to saving individual frames...")
        
        # Fallback: save frames as images
        frames_dir = '/content/frames/'
        os.makedirs(frames_dir, exist_ok=True)
        
        obs, info = env.reset()
        
        for i in range(100):  # Fewer frames for image sequence
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            try:
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame_bgr, (640, 480))
                cv2.imwrite(f'{frames_dir}/frame_{i:04d}.png', frame_resized)
            except Exception as e:
                print(f"Frame save error at step {i}: {e}")
                break
            
            if terminated or truncated:
                obs, info = env.reset()
            
            if i % 25 == 0:
                print(f"Frame {i}/100")
        
        env.close()
        print(f"Frames saved to {frames_dir}")
        return
    
    # If video writer worked, proceed with original code
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
    print(f"Video saved as '{video_path}'")

def main():  # ADD THIS FUNCTION
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
    print("Video file: /content/humanoid_simulation.mp4")
    print("="*50)

if __name__ == "__main__":
    main()