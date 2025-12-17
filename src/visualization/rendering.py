# rendering.py
"""
Display and rendering utilities for MuJoCo environments.

Handles headless display setup and environment testing.
"""

import os
import numpy as np


def setup_display(backend: str = "egl") -> bool:
    """
    Setup display for MuJoCo rendering.
    
    Works in both headless (server/Colab) and desktop environments.
    
    Args:
        backend: Rendering backend ('egl', 'osmesa', 'glfw')
        
    Returns:
        True if setup succeeded
    """
    try:
        os.environ.setdefault("MUJOCO_GL", backend)
        print(f"[rendering] MUJOCO_GL={os.environ['MUJOCO_GL']}")
        return True
    except Exception as e:
        print(f"[rendering] {backend} failed: {e}. Falling back to osmesa.")
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        return True


def test_environment(env, verbose: bool = True) -> bool:
    """
    Test if environment can render properly.
    
    Args:
        env: Gymnasium environment
        verbose: Print diagnostic messages
        
    Returns:
        True if environment renders correctly
    """
    try:
        # Reset and render one frame
        obs, info = env.reset()
        frame = env.render()

        if frame is None:
            if verbose:
                print("[rendering] render() returned None")
            return False
            
        if not isinstance(frame, np.ndarray):
            if verbose:
                print(f"[rendering] render() did not return ndarray (got {type(frame)})")
            return False
            
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            if verbose:
                print(f"[rendering] unexpected frame shape: {frame.shape}")
            return False

        # Step once and render again
        if hasattr(env, "action_space"):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            _ = env.render()

        if verbose:
            print(f"[rendering] Env test OK. Frame shape={frame.shape}, dtype={frame.dtype}")
        return True
        
    except Exception as e:
        if verbose:
            print(f"[rendering] Environment test failed: {e}")
        return False


def create_video_writer(
    output_path: str,
    fps: int = 30,
    width: int = 640,
    height: int = 480
):
    """
    Create a video writer with fallback codecs.
    
    Args:
        output_path: Path to output video file
        fps: Frames per second
        width: Video width
        height: Video height
        
    Returns:
        cv2.VideoWriter or None if failed
    """
    try:
        import cv2
    except ImportError:
        print("[rendering] OpenCV not installed")
        return None
    
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
    ]
    
    for codec_name, fourcc in codecs_to_try:
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"[rendering] VideoWriter initialized with {codec_name}")
            return writer
        writer.release()
    
    print("[rendering] Failed to initialize VideoWriter with any codec")
    return None

