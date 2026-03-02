"""
Visualization utilities for humanoid training
"""
import os
import numpy as np

from src.utils import configure_mujoco_gl


def setup_display():
    """Setup display for Colab/server/local environments (platform-aware)."""
    configure_mujoco_gl()
    gl = os.environ.get("MUJOCO_GL", "(unset — using default)")
    print(f"[visualization] MUJOCO_GL={gl}")
    return True

def test_environment(env) -> bool:
    """Test if environment can render properly"""
    try:
        # Reset and render one frame
        obs, info = env.reset()
        frame = env.render()

        if frame is None:
            print("[visualization] render() returned None")
            return False
        if not isinstance(frame, np.ndarray):
            print(f"[visualization] render() did not return ndarray (got {type(frame)})")
            return False
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            print(f"[visualization] unexpected frame shape: {frame.shape}")
            return False

        # Step once and render again
        if hasattr(env, "action_space"):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            _ = env.render()

        print(f"[visualization] Env test OK. Frame shape={frame.shape}, dtype={frame.dtype}")
        return True
    except Exception as e:
        print(f"[visualization] Environment test failed: {e}")
        return False