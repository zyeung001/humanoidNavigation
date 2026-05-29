"""View humanoid_real_raw.xml static (no physics) to inspect Fusion geometry."""
from pathlib import Path
import mujoco
import mujoco.viewer

MJCF = Path(__file__).parent.parent / "models" / "humanoid_real_raw.xml"
model = mujoco.MjModel.from_xml_path(str(MJCF))
data  = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print(f"Bodies: {model.nbody}  DOF: {model.njnt}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
