"""Render humanoid_real.xml from several angles to PNGs for a quick visual check.
Run: py -3.13 scripts/render_humanoid_real.py
"""
from pathlib import Path
import numpy as np
import mujoco

try:
    from PIL import Image
    def save(arr, path): Image.fromarray(arr).save(path)
except ImportError:
    import imageio.v2 as imageio
    def save(arr, path): imageio.imwrite(path, arr)

MJCF = Path(__file__).parent.parent / "models" / "humanoid_real.xml"
OUT  = Path(__file__).parent.parent / "data" / "renders"
OUT.mkdir(parents=True, exist_ok=True)

model = mujoco.MjModel.from_xml_path(str(MJCF))
data  = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)  # 'home' keyframe = upright
mujoco.mj_forward(model, data)

# Center the camera on the robot's bounding box.
xs = data.xpos[1:]
center = (xs.max(0) + xs.min(0)) / 2
extent = float(np.linalg.norm(xs.max(0) - xs.min(0))) + 0.3

cam = mujoco.MjvCamera()
cam.lookat[:] = center
cam.distance = extent * 1.6

renderer = mujoco.Renderer(model, height=480, width=360)
for name, az, el in [("front", 90, -10), ("side", 0, -10), ("front45", 45, -15)]:
    cam.azimuth = az
    cam.elevation = el
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    p = OUT / f"humanoid_{name}.png"
    save(img, str(p))
    print(f"saved {p}")
print("DONE")
