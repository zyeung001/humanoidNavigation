"""
View humanoid_real.xml — press Space to pause/unpause physics.
Run with: py -3.13 scripts/view_humanoid_real.py [--static]

--static : show the model without stepping physics (just geometry check)
"""
import sys
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("ERROR: py -3.13 -m pip install mujoco")
    sys.exit(1)

STATIC = '--static' in sys.argv

MJCF = Path(__file__).parent.parent / "models" / "humanoid_real.xml"
if not MJCF.exists():
    print(f"ERROR: {MJCF} not found.  Run scripts/convert_urdf_to_mjcf.py first.")
    sys.exit(1)

model = mujoco.MjModel.from_xml_path(str(MJCF))
data  = mujoco.MjData(model)

# Use body pos attribute for initial state (NOT keyframe at z=0)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print(f"Bodies:    {model.nbody - 1}")
print(f"DOF:       {model.njnt}  (1 freejoint + 17 actuated)")
print(f"Actuators: {model.nu}")
print()
if STATIC:
    print("STATIC MODE: geometry only, no physics.")
    print("Check that the robot is upright and parts are assembled correctly.")
    print("Then run without --static to test physics.")
else:
    print("PHYSICS MODE: running simulation.")
    print("If robot explodes, run with --static first to check geometry orientation.")
print()
print("Blue box = 60g electronics  |  red dots = FSR  |  yellow = IMU")
print("Press Space to pause, ESC to quit.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if not STATIC:
            mujoco.mj_step(model, data)
        viewer.sync()
