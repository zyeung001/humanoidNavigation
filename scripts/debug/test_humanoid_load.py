#!/usr/bin/env python3
# test_humanoid_load.py
"""
Fixed humanoid loading script with better error handling and reliability
"""

import subprocess
import os
import time
import sys
import yaml
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config/environment_config.yaml"):
    """Load configuration from YAML file with better error handling"""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}")
            logger.info("Creating default config file...")
            create_default_config(config_path)
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Config loaded from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        logger.info("Using default settings...")
        return get_default_config()
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        return get_default_config()

def create_default_config(config_path):
    """Create a default config file"""
    config = get_default_config()
    
    # Create config directory if it doesn't exist
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)
    logger.info(f"Default config created at {config_path}")

def get_default_config():
    """Default configuration if YAML file is missing"""
    return {
        'environment': {
            'name': 'Humanoid-v5', 
            'render_mode': 'rgb_array',
            'max_episode_steps': 1000
        },
        'recording': {
            'fps': 30,
            'width': 640,
            'height': 480,
            'default_episodes': 5
        },
        'simulation': {
            'max_steps': 500,
            'reset_on_done': True,
            'save_frames': True
        }
    }

def check_dependencies():
    """Check if required packages are available"""
    required_packages = ['gymnasium', 'numpy', 'cv2', 'mujoco']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'mujoco':
                import mujoco
            else:
                __import__(package)
            logger.info(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install gymnasium[mujoco] opencv-python numpy")
        return False
    
    return True

def setup_display():
    """Enhanced setup for MuJoCo rendering with better error handling"""
    try:
        logger.info("Setting up display environment...")
        
        # Check if we're in a headless environment
        if 'DISPLAY' not in os.environ or os.environ.get('DISPLAY') == '':
            logger.info("No display detected, setting up virtual display...")
            
            # Install required packages
            logger.info("Installing display packages...")
            result = subprocess.run([
                'apt-get', 'update', '&&', 
                'apt-get', 'install', '-y', 'xvfb', 'python3-opengl', 
                'mesa-utils', 'libosmesa6-dev', 'freeglut3-dev'
            ], shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning("Package installation may have failed, continuing...")
            
            # Set environment variables BEFORE importing gymnasium
            os.environ['DISPLAY'] = ':99'
            os.environ['MUJOCO_GL'] = 'osmesa'
            os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
            
            # Start Xvfb
            logger.info("Starting virtual display...")
            xvfb_process = subprocess.Popen([
                'Xvfb', ':99', '-screen', '0', '1024x768x24', '-ac', '+extension', 'GLX'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for display to initialize
            time.sleep(3)
            
            # Check if Xvfb started successfully
            if xvfb_process.poll() is not None:
                logger.error("Xvfb failed to start")
                return False
        
        logger.info("Display setup complete")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up display: {e}")
        logger.info("Trying to continue without virtual display...")
        # Set fallback environment variables
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        return True

def test_environment(config):
    """Test if the environment can be created and rendered"""
    try:
        import gymnasium as gym
        
        env_name = config['environment']['name']
        render_mode = config['environment']['render_mode']
        
        logger.info(f"Testing environment creation: {env_name}")
        env = gym.make(env_name, render_mode=render_mode)
        
        logger.info("Testing environment reset...")
        obs, info = env.reset()
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        
        logger.info("Testing rendering...")
        frame = env.render()
        
        if frame is not None:
            logger.info(f"Success! Frame shape: {frame.shape}")
        else:
            logger.warning("Rendering returned None")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"Environment test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_simulation(config):
    """Run the actual simulation using config settings"""
    try:
        import gymnasium as gym
        import numpy as np
        import cv2
        
        # Get settings from config
        env_name = config['environment']['name']
        render_mode = config['environment']['render_mode']
        fps = config['recording']['fps']
        width = config['recording']['width']
        height = config['recording']['height']
        max_steps = config['simulation'].get('max_steps', 500)
        
        logger.info(f"Creating environment: {env_name}")
        env = gym.make(env_name, render_mode=render_mode)
        
        # Create output directories
        video_dir = Path('data/videos')
        frames_dir = Path('data/videos/frames')
        video_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = video_dir / 'humanoid_simulation.mp4'
        
        # Initialize video writer
        out = None
        codecs_to_try = [
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ]
        
        for codec_name, fourcc in codecs_to_try:
            logger.info(f"Trying codec: {codec_name}")
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            if out.isOpened():
                logger.info(f"Successfully initialized VideoWriter with {codec_name}")
                break
            else:
                logger.warning(f"{codec_name} failed")
                if out:
                    out.release()
                out = None
        
        # Run simulation
        obs, info = env.reset()
        
        logger.info(f"Starting simulation:")
        logger.info(f"  - Steps: {max_steps}")
        logger.info(f"  - Resolution: {width}x{height}")
        logger.info(f"  - FPS: {fps}")
        
        saved_frames = 0
        
        for step in range(max_steps):
            # Take random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            try:
                frame = env.render()
                if frame is not None:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_resized = cv2.resize(frame_bgr, (width, height))
                    
                    # Save to video if writer is available
                    if out and out.isOpened():
                        out.write(frame_resized)
                    
                    # Always save some frames as backup
                    if step % 25 == 0:
                        frame_path = frames_dir / f'frame_{step:04d}.png'
                        cv2.imwrite(str(frame_path), frame_resized)
                        saved_frames += 1
                
            except Exception as e:
                logger.error(f"Rendering error at step {step}: {e}")
                break
            
            if terminated or truncated:
                if config['simulation'].get('reset_on_done', True):
                    obs, info = env.reset()
                    logger.info(f"Episode ended at step {step}, resetting...")
            
            if step % 100 == 0:
                logger.info(f"Step {step}/{max_steps}")
        
        # Cleanup
        env.close()
        if out:
            out.release()
        
        logger.info(f"Simulation complete!")
        if out and out.isOpened():
            logger.info(f"Video saved as '{video_path}'")
        logger.info(f"Saved {saved_frames} individual frames to '{frames_dir}'")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main execution function"""
    print("=" * 60)
    print("HUMANOID SIMULATION SETUP")
    print("=" * 60)
    

    logger.info("Checking dependencies...")
    if not check_dependencies():
        logger.error("Missing required dependencies. Please install them and try again.")
        return 1
    

    logger.info("Loading configuration...")
    config = load_config()
    logger.info(f"Environment: {config['environment']['name']}")
    logger.info(f"Recording: {config['recording']['width']}x{config['recording']['height']} @ {config['recording']['fps']}fps")
    

    logger.info("Setting up display...")
    if not setup_display():
        logger.error("Failed to setup display. Trying to continue anyway...")
    

    logger.info("Testing environment...")
    if not test_environment(config):
        logger.error("Environment test failed. Please check your MuJoCo installation.")
        return 1
    

    logger.info("Running simulation...")
    run_simulation(config)
    
    print("=" * 60)
    print("SIMULATION COMPLETE!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())