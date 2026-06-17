"""
View a humanoid MJCF — press Space to pause/unpause physics.

Run with: py -3.13 scripts/view_humanoid_real.py [model] [--static] [--zero]
  model    : path to an .xml (default models/humanoid_real.xml; pass
             models/humanoid_real_v2.xml to view the rebuilt model)
  --static : show geometry without stepping physics (geometry check)
  --zero   : start at qpos0 (zeros) instead of the 'home' keyframe pose
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
ZERO   = '--zero' in sys.argv
paths  = [a for a in sys.argv[1:] if not a.startswith('--')]

MJCF = Path(paths[0]) if paths else Path(__file__).parent.parent / "models" / "humanoid_real_v2.xml"
if not MJCF.exists():
    print(f"ERROR: {MJCF} not found.")
    sys.exit(1)

model = mujoco.MjModel.from_xml_path(str(MJCF))
data  = mujoco.MjData(model)

# Start at the 'home' keyframe (the solved standing pose) unless --zero given.
home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
if home >= 0 and not ZERO:
    mujoco.mj_resetDataKeyframe(model, data, home)
    print(f"Loaded 'home' keyframe from {MJCF.name}")
else:
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

# Command the home pose so position servos hold the stance (will lean after ~1s
# with no ankle — that is expected; RL supplies balance at train time).
hold_ctrl = data.qpos[7:7 + model.nu].copy() if (home >= 0 and not ZERO) else None

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if not STATIC:
            if hold_ctrl is not None:
                data.ctrl[:] = hold_ctrl
            mujoco.mj_step(model, data)
        viewer.sync()
