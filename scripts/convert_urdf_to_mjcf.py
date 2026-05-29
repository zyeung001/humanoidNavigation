"""
Convert HumanoidReal (Fusion 360 -> fusion2urdf export) to a MuJoCo MJCF.

Source: C:/Users/zachc/Downloads/HumanoidAssembly/HumanoidReal_description
  - HumanoidReal.xacro references meshes by their real part names (base_link.stl,
    mount_2.stl, Servo_SC15_1.stl, ...). All STLs are present, so no mesh
    remapping or box substitution is needed.
  - 17 revolute joints (the actuated DOF) + ~51 fixed joints that weld the
    decorative/structural meshes onto their parent bodies.
  - Several links export zero inertia; these are clamped so MuJoCo accepts them.

Pipeline: xacro -> plain URDF -> MuJoCo compile -> raw MJCF -> post-process.

Usage: py -3.13 scripts/convert_urdf_to_mjcf.py
"""
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SRC      = Path(r"C:/Users/zachc/Downloads/HumanoidAssembly/HumanoidReal_description")
PROJ     = Path(__file__).parent.parent
MESH_DST = PROJ / "models" / "humanoid_meshes"
URDF_TMP = SRC / "urdf" / "HumanoidReal_fixed.urdf"
MJCF_RAW = PROJ / "models" / "humanoid_real_raw.xml"
MJCF_OUT = PROJ / "models" / "humanoid_real.xml"

MIN_I = 1e-6  # minimum diagonal inertia MuJoCo will accept for a moving body

# ── 1. Copy meshes ─────────────────────────────────────────────────────────
MESH_DST.mkdir(parents=True, exist_ok=True)
src_meshes = list((SRC / "meshes").glob("*.stl"))
for stl in src_meshes:
    shutil.copy2(stl, MESH_DST / stl.name)
print(f"Copied {len(src_meshes)} STL files to {MESH_DST}")

# ── 2. xacro -> plain URDF ─────────────────────────────────────────────────
text = (SRC / "urdf" / "HumanoidReal.xacro").read_text(encoding="utf-8")

# Drop xacro includes (materials/trans/gazebo) and the xacro namespace.
text = re.sub(r'<xacro:include[^>]*/>\s*\n?', '', text)
text = text.replace(' xmlns:xacro="http://www.ros.org/wiki/xacro"', '')

# Materials are defined in the stripped materials.xacro; remove dangling refs.
text = re.sub(r'<material name="[^"]*"\s*/>', '', text)
text = re.sub(r'<material name="[^"]*">.*?</material>', '', text, flags=re.DOTALL)

# Point mesh paths at the copied mesh directory (absolute, forward slashes).
mesh_dir = str(MESH_DST).replace("\\", "/")
text = text.replace('package://HumanoidReal_description/meshes/', f'{mesh_dir}/')

# Clamp zero/negative diagonal inertia and zero the off-diagonals.
def fix_inertia_tag(m):
    s = m.group(0)
    for attr in ('ixx', 'iyy', 'izz'):
        s = re.sub(rf'{attr}="([^"]*)"',
                   lambda am, a=attr: f'{a}="{max(abs(float(am.group(1))), MIN_I)}"', s)
    for attr in ('ixy', 'ixz', 'iyz'):
        s = re.sub(rf'{attr}="[^"]*"', f'{attr}="0.0"', s)
    return s
text = re.sub(r'<inertia [^/]*/>', fix_inertia_tag, text)

# Add a world link + floating joint so base_link becomes a MuJoCo freejoint.
floating_block = (
    '<link name="world_link"/>\n'
    '<joint name="base_joint" type="floating">\n'
    '  <parent link="world_link"/>\n'
    '  <child link="base_link"/>\n'
    '</joint>\n'
)
text = text.replace('<link name="base_link">', floating_block + '<link name="base_link">')

URDF_TMP.write_text(text, encoding="utf-8")
print("Wrote cleaned URDF")

# ── 3. Compile with MuJoCo, save raw MJCF ──────────────────────────────────
try:
    import mujoco
except ImportError:
    print("ERROR: py -3.13 -m pip install mujoco numpy")
    sys.exit(1)

print("Compiling with MuJoCo...")
model = mujoco.MjModel.from_xml_path(str(URDF_TMP))
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

joint_names = [model.joint(i).name for i in range(model.njnt)]
actuated = [jn for jn in joint_names if jn != 'base_joint']

# Use lowest GEOM z (not body-frame xpos): fusion2urdf scatters body frames
# across Fusion's world coords while geoms are offset back to their true
# positions. Using xpos would lift the robot by the wrong amount and leave
# feet floating ~0.3 m above the floor at the home keyframe.
non_world_geom_mask = model.geom_bodyid != 0
non_world_geom_z = data.geom_xpos[non_world_geom_mask, 2]
min_z = float(non_world_geom_z.min()) if non_world_geom_z.size else 0.0
print(f"  Bodies={model.nbody-1}  DOF(njnt)={model.njnt}  Actuated={len(actuated)}  lowest geom z={min_z:.3f}")

mujoco.mj_saveLastXML(str(MJCF_RAW), model)

# ── 4. Post-process MJCF ───────────────────────────────────────────────────
print("Post-processing...")
tree = ET.parse(str(MJCF_RAW))
root = tree.getroot()

def get_or_create(parent, tag):
    # NOTE: `parent.find(tag) or ...` is wrong — an element with no children is
    # falsy, so it would create a duplicate. Test against None explicitly.
    el = parent.find(tag)
    return el if el is not None else ET.SubElement(parent, tag)

# 4a. Compiler — keep visuals so the viewer shows real meshes.
c = get_or_create(root, 'compiler')
c.set('angle', 'radian')
c.set('balanceinertia', 'true')
c.set('discardvisual', 'false')

# 4b. Physics options.
o = get_or_create(root, 'option')
o.set('timestep', '0.005')
o.set('gravity', '0 0 -9.81')
o.set('iterations', '50')
o.set('solver', 'Newton')
o.set('tolerance', '1e-10')

# 4c. Defaults.
default = get_or_create(root, 'default')
jd = get_or_create(default, 'joint')
jd.set('damping', '1.0')
jd.set('armature', '0.01')
gd = get_or_create(default, 'geom')
gd.set('contype', '1')
gd.set('conaffinity', '1')
gd.set('condim', '3')
gd.set('friction', '0.8 0.1 0.01')

# 4d. Floor + light.
worldbody = root.find('worldbody')
for g in worldbody.findall('geom'):
    if g.get('name') == 'floor':
        worldbody.remove(g)
floor = ET.Element('geom', name='floor', type='plane',
                   size='10 10 0.1', rgba='.9 .9 .9 1', condim='3')
worldbody.insert(0, floor)
light = ET.Element('light', directional='true', diffuse='.8 .8 .8',
                   pos='0 0 5', dir='0 0 -1')
worldbody.insert(0, light)

# 4e. Lift base_link so the lowest body sits just above the floor.
def find_body(name):
    for b in worldbody.iter('body'):
        if b.get('name') == name:
            return b
    return None

lift = -min_z + 0.02
base_elem = find_body('base_link')
if base_elem is not None:
    cur = base_elem.get('pos', '0 0 0').split()
    new_z = (float(cur[2]) if len(cur) == 3 else 0.0) + lift
    base_elem.set('pos', f'{cur[0] if len(cur)==3 else 0} {cur[1] if len(cur)==3 else 0} {new_z:.4f}')
    print(f"  Lifted base_link by {lift:.3f}m")

# 4f. Actuators — one position-agnostic motor per revolute joint.
act = root.find('actuator')
if act is not None:
    root.remove(act)
act = ET.SubElement(root, 'actuator')
for jname in actuated:
    ET.SubElement(act, 'motor', name=jname, joint=jname,
                  ctrllimited='true', ctrlrange='-1 1', gear='50')
print(f"  Added {len(act)} actuators")

# 4g. Keyframe — robot upright, all joints at 0.
keyframe = get_or_create(root, 'keyframe')
kf = ET.SubElement(keyframe, 'key', name='home')
qpos = [0, 0, round(lift, 4), 1, 0, 0, 0] + [0] * len(actuated)
kf.set('qpos', ' '.join(str(v) for v in qpos))

# ── 5. Write final MJCF ────────────────────────────────────────────────────
ET.indent(tree, space='  ')
tree.write(str(MJCF_OUT), encoding='unicode', xml_declaration=True)
print(f"Saved {MJCF_OUT}")

# Remove the raw intermediate so only the final model remains in models/.
MJCF_RAW.unlink(missing_ok=True)

# ── 6. Verify ──────────────────────────────────────────────────────────────
m2 = mujoco.MjModel.from_xml_path(str(MJCF_OUT))
n_act = m2.nu
n_jnt = m2.njnt - 1  # minus freejoint
print(f"Verify: Bodies={m2.nbody-1}  DOF={m2.njnt}  Actuators={n_act}  Sensors={m2.nsensor}")
if n_jnt == 17 and n_act == 17:
    print("  SUCCESS: 17 DOF, 17 actuators")
else:
    print(f"  WARNING: expected 17 actuated DOF + 17 actuators, got {n_jnt} / {n_act}")
