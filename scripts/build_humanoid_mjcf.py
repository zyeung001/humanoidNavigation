"""
Build a Menagerie-style MJCF for HumanoidReal from the Fusion 360 export, using
PRIMITIVE collision geometry (the sim2real standard for legged robots).

Pipeline:
  1. xacro -> plain URDF (strip ROS/gazebo, keep REAL Fusion masses)
  2. MuJoCo compiles the URDF -> assembled body tree + combined inertials
  3. for every MOVING body, fit collision PRIMITIVES from its visual-mesh verts:
       - foot meshes        -> box  (flat ground-contact polygon)
       - elongated segments -> capsule (along the long OBB axis)
       - blocky segments    -> box
     visual stays full-mesh; collision is 1-2 primitives/body
  4. visual/collision default classes, self-collision ON with parent-child
     excludes, <position> actuators sized to SC15 servos, home keyframe
  5. compile + a 500-step zero-control drop test (NaN / penetration / drift)

Visual meshes are the OBJ copies under models/_obj_work/<mesh>/<mesh>.obj.

Usage: py -3.13 scripts/build_humanoid_mjcf.py
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SRC      = Path(r"C:/Users/zachc/Downloads/HumanoidAssembly/HumanoidReal_description")
PROJ     = Path(__file__).parent.parent
MESH_STL = PROJ / "models" / "humanoid_meshes"        # source STLs (mm)
MESH_OBJ = PROJ / "models" / "humanoid_meshes_obj"    # generated visual OBJs (permanent)
URDF_TMP = SRC / "urdf" / "HumanoidReal_fixed.urdf"
MJCF_RAW = PROJ / "models" / "humanoid_real_v2_raw.xml"
MJCF_OUT = PROJ / "models" / "humanoid_real_v2.xml"

MESH_SCALE = "0.001 0.001 0.001"   # mm OBJ -> meters (matches URDF scale)
MIN_I      = 1e-6
CAPSULE_ASPECT = 1.6               # long/median extent above this -> capsule, else box

# --- Servo actuator params, per the BOM (kp/kv = estimate, tune via sysID) ---
#   Legs/waist (11): Feetech SCS15-class, 15 kg*cm @6V -> 1.47 N*m stall (~1.8 @8.4V)
#   Arms (6, Revolute 1-6): Feetech SCS0009, peak stall 2.3 kg*cm @6V -> 0.226 N*m
# kp lowered 20->12 (legs) so the servo is less "taut"; kv adds velocity feedback
# (PD, not pure-P) so the actuator damps setpoint jitter instead of ringing.
ARM_JOINTS = {f'Revolute {i}' for i in range(1, 7)}
SCS15   = {'kp': 12.0, 'kv': 2.0, 'force': 1.47}
SCS0009 = {'kp': 5.0,  'kv': 0.5, 'force': 0.226}
def servo_of(joint_name):
    return SCS0009 if joint_name in ARM_JOINTS else SCS15


def convert_stl_to_obj() -> int:
    """Generate visual OBJs (mm) from the source STLs into the permanent mesh dir."""
    import trimesh
    MESH_OBJ.mkdir(parents=True, exist_ok=True)
    stls = sorted(MESH_STL.glob("*.stl"))
    for p in stls:
        trimesh.load(p, process=False).export(MESH_OBJ / (p.stem + ".obj"))
    return len(stls)


def strip_xacro_to_urdf() -> None:
    text = (SRC / "urdf" / "HumanoidReal.xacro").read_text(encoding="utf-8")
    text = re.sub(r'<xacro:include[^>]*/>\s*\n?', '', text)
    text = text.replace(' xmlns:xacro="http://www.ros.org/wiki/xacro"', '')
    text = re.sub(r'<material name="[^"]*"\s*/>', '', text)
    text = re.sub(r'<material name="[^"]*">.*?</material>', '', text, flags=re.DOTALL)
    stl_dir = str(PROJ / "models" / "humanoid_meshes").replace("\\", "/")
    text = text.replace('package://HumanoidReal_description/meshes/', f'{stl_dir}/')

    def fix_inertia(m):
        s = m.group(0)
        for a in ('ixx', 'iyy', 'izz'):
            s = re.sub(rf'{a}="([^"]*)"',
                       lambda am, a=a: f'{a}="{max(abs(float(am.group(1))), MIN_I)}"', s)
        for a in ('ixy', 'ixz', 'iyz'):
            s = re.sub(rf'{a}="[^"]*"', f'{a}="0.0"', s)
        return s
    text = re.sub(r'<inertia [^/]*/>', fix_inertia, text)

    floating = ('<link name="world_link"/>\n'
                '<joint name="base_joint" type="floating">\n'
                '  <parent link="world_link"/>\n  <child link="base_link"/>\n</joint>\n')
    text = text.replace('<link name="base_link">', floating + '<link name="base_link">')
    URDF_TMP.write_text(text, encoding="utf-8")


# ── primitive fitting ───────────────────────────────────────────────────────
def _fit_obb(pts):
    import trimesh
    T, ext = trimesh.bounds.oriented_bounds(trimesh.points.PointCloud(pts))
    M = np.linalg.inv(T)               # OBB frame -> body frame
    return M[:3, 3], M[:3, :3], np.asarray(ext)


def _quat(R):
    import trimesh
    M = np.eye(4)
    M[:3, :3] = R
    return trimesh.transformations.quaternion_from_matrix(M)   # [w,x,y,z]


def _box(pts):
    c, R, ext = _fit_obb(pts)
    return {'type': 'box', 'pos': c, 'quat': _quat(R), 'size': ext / 2}


def _capsule(pts):
    c, R, ext = _fit_obb(pts)
    o = np.argsort(ext)                # ascending; o[2] = longest
    radius = 0.5 * float((ext[o[0]] + ext[o[1]]) / 2)
    half   = max(float(ext[o[2]]) / 2 - radius, 1e-3)
    Rz = np.column_stack([R[:, o[0]], R[:, o[1]], R[:, o[2]]])   # long axis -> z
    if np.linalg.det(Rz) < 0:
        Rz[:, 0] = -Rz[:, 0]
    return {'type': 'capsule', 'pos': c, 'quat': _quat(Rz), 'size': np.array([radius, half])}


def _primitive(pts, force_box=False):
    _, _, ext = _fit_obb(pts)
    aspect = ext.max() / (np.median(ext) + 1e-9)
    return _box(pts) if (force_box or aspect < CAPSULE_ASPECT) else _capsule(pts)


def reground_inertia(model):
    """name -> (com_local, fullinertia[ixx iyy izz ixy ixz iyz], mass).

    Fusion exports inertia about the ASSEMBLY origin -> tensors 100-1000x too big
    (radius of gyration 0.3-0.4m for a 0.05m part). Recompute each body's COM and
    inertia from the convex hull of its own mesh geometry, scaled to the body's
    real (Fusion) mass. Keeps total mass; grounds the tensor in actual geometry.
    """
    import mujoco
    import trimesh
    out = {}
    for bid in range(1, model.nbody):
        verts = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[g]
            adr, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[g])
            verts.append(model.mesh_vert[adr:adr + n] @ R.reshape(3, 3).T + model.geom_pos[g])
        if not verts:
            continue
        mass = float(model.body_mass[bid])
        try:
            hull = trimesh.PointCloud(np.vstack(verts)).convex_hull
            vol = float(hull.volume)
            if vol <= 1e-12 or mass <= 0:
                continue
            inertia = np.asarray(hull.moment_inertia) * (mass / vol)  # unit-density -> real mass
            com = np.asarray(hull.center_mass)
        except Exception:
            continue
        out[model.body(bid).name] = (com,
                                     [inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                      inertia[0, 1], inertia[0, 2], inertia[1, 2]], mass)
    return out


def fit_joint_pivots(model, data):
    """joint -> pivot in child-body-local coords = parent<->child mesh seam.

    Fusion exports every joint at pos='0 0 0' while the body origin sits ~0.6-0.8m
    from the actual limb, so limbs tear off their hinge the instant a joint rotates.
    The true pivot is where parent and child geometry meet (closest-point seam).
    """
    from scipy.spatial import cKDTree

    def bverts(bid):
        pts = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid or model.geom_dataid[g] < 0:
                continue
            mid = model.geom_dataid[g]
            adr, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            pts.append(model.mesh_vert[adr:adr + n] @ data.geom_xmat[g].reshape(3, 3).T
                       + data.geom_xpos[g])
        return np.vstack(pts) if pts else None

    piv = {}
    for j in range(model.njnt):
        name = model.joint(j).name
        if name == 'base_joint':
            continue
        cb = model.jnt_bodyid[j]
        pb = model.body_parentid[cb]
        cv, pv = bverts(cb), bverts(pb)
        if cv is None or pv is None:
            continue
        ci = cv[::max(1, len(cv) // 1500)]
        pi = pv[::max(1, len(pv) // 1500)]
        dist, idx = cKDTree(pi).query(ci)
        k = int(dist.argmin())
        seam = (ci[k] + pi[idx[k]]) / 2
        piv[name] = data.xmat[cb].reshape(3, 3).T @ (seam - data.xpos[cb])
    return piv


def fit_body_primitives(model):
    """name -> [primitive dicts] for every moving body, from its visual-mesh verts."""
    import mujoco
    out = {}
    for bid in range(1, model.nbody):
        foot, rest = [], []
        for gi in range(model.ngeom):
            if model.geom_bodyid[gi] != bid or model.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[gi]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[adr:adr + num]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[gi])
            vb = v @ R.reshape(3, 3).T + model.geom_pos[gi]
            (foot if 'foot' in model.mesh(mid).name.lower() else rest).append(vb)
        prims = []
        if foot:
            prims.append(_primitive(np.vstack(foot), force_box=True))
        if rest and len(np.vstack(rest)) >= 4:
            prims.append(_primitive(np.vstack(rest)))
        out[model.body(bid).name] = prims
    return out


# ── main ────────────────────────────────────────────────────────────────────
def main() -> None:
    try:
        import mujoco
    except ImportError:
        sys.exit("ERROR: py -3.13 -m pip install mujoco trimesh")

    print(f"0. STL -> OBJ: {convert_stl_to_obj()} visual meshes -> {MESH_OBJ.name}/")
    print("1. xacro -> URDF")
    strip_xacro_to_urdf()

    print("2. compile URDF")
    model = mujoco.MjModel.from_xml_path(str(URDF_TMP))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    zs = data.geom_xpos[model.geom_bodyid != 0, 2]
    min_z = float(zs.min()) if zs.size else 0.0
    actuated = [model.joint(i).name for i in range(model.njnt)
                if model.joint(i).name != 'base_joint']
    jnt_range = {model.joint(i).name: model.jnt_range[i] for i in range(model.njnt)}
    print(f"   bodies={model.nbody-1} actuated={len(actuated)} "
          f"mass={model.body_subtreemass[1]:.3f}kg low_z={min_z:.3f}")

    print("3. reground inertia + fit joint pivots + collision primitives")
    inertia = reground_inertia(model)
    pivots = fit_joint_pivots(model, data)
    prims = fit_body_primitives(model)
    ncap = sum(1 for v in prims.values() for p in v if p['type'] == 'capsule')
    nbox = sum(1 for v in prims.values() for p in v if p['type'] == 'box')
    print(f"   {ncap} capsules + {nbox} boxes across {len(prims)} bodies")
    mujoco.mj_saveLastXML(str(MJCF_RAW), model)

    print("4. rewrite MJCF")
    tree = ET.parse(str(MJCF_RAW))
    root = tree.getroot()
    worldbody = root.find('worldbody')

    comp = root.find('compiler')
    comp = comp if comp is not None else ET.SubElement(root, 'compiler')
    comp.set('angle', 'radian')
    comp.set('meshdir', MESH_OBJ.name)
    comp.set('autolimits', 'true')
    opt = root.find('option')
    opt = opt if opt is not None else ET.SubElement(root, 'option')
    opt.set('timestep', '0.005')
    opt.set('integrator', 'implicitfast')

    for d in root.findall('default'):
        root.remove(d)
    default = ET.Element('default')
    root.insert(list(root).index(comp) + 1, default)
    # armature 0.01 -> 0.05: 0.01 was a generic Humanoid-v5 placeholder. Geared
    # servos have real reflected rotor inertia; more armature smooths the plant
    # and reduces twitch (still an estimate, tune via sysID).
    ET.SubElement(default, 'joint', damping='1.0', armature='0.05')
    vis = ET.SubElement(default, 'default', {'class': 'visual'})
    ET.SubElement(vis, 'geom', group='2', type='mesh', contype='0', conaffinity='0', density='0')
    col = ET.SubElement(default, 'default', {'class': 'collision'})
    # solref timeconst (0.01) = 2*timestep, the stiffest contact that stays stable at
    # dt=0.005; solimp ramps to 0.99 so resting penetration stays sub-mm.
    ET.SubElement(col, 'geom', group='3', condim='3', friction='0.8 0.1 0.01',
                  solref='0.01 1', solimp='0.95 0.99 0.001', rgba='0.8 0.3 0.3 0.4')

    # fix joint pivots so limbs rotate about their real hinge, not the body origin
    for body in worldbody.iter('body'):
        for jt in body.findall('joint'):
            if jt.get('name') in pivots:
                jt.set('pos', ' '.join(f'{x:.5f}' for x in pivots[jt.get('name')]))

    # replace Fusion's inflated inertials with geometry-grounded ones
    for body in worldbody.iter('body'):
        name = body.get('name')
        if name not in inertia:
            continue
        com, fi, mass = inertia[name]
        inel = body.find('inertial')
        if inel is None:
            inel = ET.SubElement(body, 'inertial')
        inel.attrib.clear()
        inel.set('pos', ' '.join(f'{x:.6f}' for x in com))
        inel.set('mass', f'{mass:.6f}')
        inel.set('fullinertia', ' '.join(f'{x:.3e}' for x in fi))

    # swap mesh geoms -> visual-only; collect used meshes; append fitted primitives
    used = set()
    for body in worldbody.iter('body'):
        name = body.get('name')
        for g in list(body.findall('geom')):
            mesh = g.get('mesh')
            if mesh is None:
                continue
            pos, quat = g.get('pos', '0 0 0'), g.get('quat', '1 0 0 0')
            body.remove(g)
            used.add(mesh)
            ET.SubElement(body, 'geom', {'class': 'visual', 'mesh': mesh, 'pos': pos, 'quat': quat})
        for p in prims.get(name, []):
            attrs = {'class': 'collision', 'type': p['type'],
                     'pos': ' '.join(f'{x:.5f}' for x in p['pos']),
                     'quat': ' '.join(f'{x:.5f}' for x in p['quat']),
                     'size': ' '.join(f'{x:.5f}' for x in p['size'])}
            body.append(ET.Element('geom', attrs))

    for a in root.findall('asset'):
        root.remove(a)
    asset = ET.Element('asset')
    root.insert(list(root).index(default) + 1, asset)
    for m in sorted(used):
        ET.SubElement(asset, 'mesh', name=m, file=f'{m}.obj', scale=MESH_SCALE)

    for g in worldbody.findall('geom'):
        if g.get('name') == 'floor':
            worldbody.remove(g)
    worldbody.insert(0, ET.Element('geom', name='floor', type='plane',
                                   size='10 10 0.1', rgba='.9 .9 .9 1', condim='3'))
    worldbody.insert(0, ET.Element('light', directional='true', diffuse='.8 .8 .8',
                                   pos='0 0 5', dir='0 0 -1'))

    lift = -min_z + 0.02
    for b in worldbody.iter('body'):
        if b.get('name') == 'base_link':
            cur = b.get('pos', '0 0 0').split()
            b.set('pos', f'{cur[0]} {cur[1]} {float(cur[2]) + lift:.4f}')
            break

    contact = ET.SubElement(root, 'contact')
    def excl(body):
        for child in body.findall('body'):
            contact.append(ET.Element('exclude', body1=body.get('name'), body2=child.get('name')))
            excl(child)
    for top in worldbody.findall('body'):
        excl(top)

    for a in root.findall('actuator'):
        root.remove(a)
    act = ET.SubElement(root, 'actuator')
    for j in actuated:
        lo, hi = jnt_range[j]
        s = servo_of(j)
        ET.SubElement(act, 'position', name=j, joint=j, kp=str(s['kp']),
                      kv=str(s['kv']),
                      ctrlrange=f'{lo:.4f} {hi:.4f}',
                      forcerange=f'-{s["force"]} {s["force"]}')

    for k in root.findall('keyframe'):
        root.remove(k)
    kf = ET.SubElement(ET.SubElement(root, 'keyframe'), 'key', name='home')
    sp_path = PROJ / "models" / "standing_pose.json"
    if sp_path.exists():
        sp = json.loads(sp_path.read_text())
        qpos = sp['base_pos'] + [1, 0, 0, 0] + [round(sp['joints'].get(n, 0.0), 5) for n in actuated]
        print(f"   home keyframe = solved standing pose ({sp_path.name})")
    else:
        qpos = [0, 0, round(lift, 4), 1, 0, 0, 0] + [0] * len(actuated)
        print("   home keyframe = zeros (no standing_pose.json found)")
    kf.set('qpos', ' '.join(map(str, qpos)))

    ET.indent(tree, space='  ')
    tree.write(str(MJCF_OUT), encoding='unicode', xml_declaration=True)

    # auto-exclude designed self-overlaps: any non-adjacent body pair touching at the
    # home pose (bulky brackets whose primitives overlap but shouldn't collide).
    mt = mujoco.MjModel.from_xml_path(str(MJCF_OUT))
    dt = mujoco.MjData(mt)
    mujoco.mj_resetDataKeyframe(mt, dt, 0)
    mujoco.mj_forward(mt, dt)
    existing = {(e.get('body1'), e.get('body2')) for e in contact}
    added = 0
    for i in range(dt.ncon):
        c = dt.contact[i]
        b1 = mt.geom_bodyid[c.geom1]
        b2 = mt.geom_bodyid[c.geom2]
        if b1 <= 0 or b2 <= 0:
            continue
        n1, n2 = mt.body(b1).name, mt.body(b2).name
        if (n1, n2) not in existing and (n2, n1) not in existing:
            contact.append(ET.Element('exclude', body1=n1, body2=n2))
            existing.add((n1, n2))
            added += 1
    if added:
        ET.indent(tree, space='  ')
        tree.write(str(MJCF_OUT), encoding='unicode', xml_declaration=True)
    MJCF_RAW.unlink(missing_ok=True)
    print(f"   wrote {MJCF_OUT}  (+{added} auto-excluded self-overlap pairs)")

    print("5. compile + drop test")
    m2 = mujoco.MjModel.from_xml_path(str(MJCF_OUT))
    d2 = mujoco.MjData(m2)
    floor_gid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    mujoco.mj_resetDataKeyframe(m2, d2, 0)
    z0 = float(d2.qpos[2])
    nan = False
    for _ in range(1500):
        d2.ctrl[:] = 0
        mujoco.mj_step(m2, d2)
        if not np.all(np.isfinite(d2.qpos)):
            nan = True
            break
    # judge floor (locomotion-relevant) penetration separately from inert self-overlap
    floor_pen = self_pen = 0.0
    for ci in range(d2.ncon):
        c = d2.contact[ci]
        is_floor = floor_gid in (c.geom1, c.geom2)
        pen = -float(c.dist)
        if is_floor:
            floor_pen = max(floor_pen, pen)
        else:
            self_pen = max(self_pen, pen)
    settled = float(np.linalg.norm(d2.qvel))
    # pivot check: each limb must rotate about a nearby hinge, not fly off
    d3 = mujoco.MjData(m2)
    mujoco.mj_resetDataKeyframe(m2, d3, 0)
    mujoco.mj_forward(m2, d3)
    worst_swing = 0.0
    for j in range(m2.njnt):
        if m2.joint(j).name == 'base_joint':
            continue
        bid = m2.jnt_bodyid[j]
        gs = [g for g in range(m2.ngeom) if m2.geom_bodyid[g] == bid]
        p0 = d3.geom_xpos[gs].mean(0).copy()
        mujoco.mj_resetDataKeyframe(m2, d3, 0)
        d3.qpos[m2.jnt_qposadr[j]] += 0.5
        mujoco.mj_forward(m2, d3)
        worst_swing = max(worst_swing, float(np.linalg.norm(d3.geom_xpos[gs].mean(0) - p0)))
        mujoco.mj_resetDataKeyframe(m2, d3, 0)
        mujoco.mj_forward(m2, d3)
    print(f"   actuators={m2.nu} geoms={m2.ngeom} excludes={m2.nexclude}")
    print(f"   DROP TEST: nan={nan}  settled|qvel|={settled:.4f}  base_z {z0:.3f}->{float(d2.qpos[2]):.3f}")
    print(f"   floor_penetration={floor_pen*1000:.2f}mm  self_overlap={self_pen*1000:.2f}mm")
    print(f"   PIVOT CHECK: worst limb swing for 0.5rad = {worst_swing*1000:.1f}mm (sane <150mm)")
    ok = (not nan) and m2.nu == 17 and floor_pen < 0.002 and settled < 0.05 and worst_swing < 0.15
    print("   => " + ("PASS" if ok else "REVIEW NEEDED"))


if __name__ == "__main__":
    main()
