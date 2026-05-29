"""
Load and view custom_humanoid.xml in the MuJoCo interactive viewer.
Run with: py -3.13 scripts/view_custom_humanoid.py
Controls: mouse drag to orbit, scroll to zoom, space to pause, ESC to quit.
"""
import sys
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("ERROR: mujoco not installed.")
    print("  py -3.13 -m pip install mujoco")
    sys.exit(1)

MJCF = Path(__file__).parent.parent / "models" / "custom_humanoid.xml"

def main():
    print(f"Loading {MJCF}")
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data  = mujoco.MjData(model)

    print(f"  Bodies:    {model.nbody - 1}")   # minus world body
    print(f"  Joints:    {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print(f"  qpos dims: {model.nq}")
    print(f"  qvel dims: {model.nv}")
    print()
    print("Joint names:", [model.joint(i).name for i in range(model.njnt)])
    print()
    print("Launching viewer — ESC or close window to quit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
