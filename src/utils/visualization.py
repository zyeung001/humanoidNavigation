# src/utils/visualization.py

def setup_display() -> bool:
    """
    Prepare headless rendering for MuJoCo/Gymnasium in environments like Colab.
    - Ensures MUJOCO_GL=egl so rgb_array rendering works without an X server.
    """
    try:
        import os
        if os.environ.get("MUJOCO_GL") is None:
            os.environ["MUJOCO_GL"] = "egl"
            print("[visualization] MUJOCO_GL not set; defaulting to 'egl' for headless rendering.")
        else:
            print(f"[visualization] MUJOCO_GL={os.environ['MUJOCO_GL']}")
        return True
    except Exception as e:
        print(f"[visualization] setup_display failed: {e}")
        return False


def test_environment(env) -> bool:
    try:
        import numpy as np

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
