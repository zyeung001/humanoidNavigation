"""Structural-integrity test for a converted MJCF humanoid.

Verifies the model loads and *stays assembled* - i.e. every body is part of one
kinematic tree, the inertials are physically plausible, and under a stabilizing
PD controller the joints hold their relative configuration without exploding or
producing NaNs. Passive collapse (a torque-free humanoid sagging to the floor)
is NOT treated as a failure; segments separating / blowing up IS.

Run after every change to the MJCF:
    py -3.13 scripts/test_model_integrity.py [path/to/model.xml]

Exit code 0 = PASS, 1 = FAIL.
"""
import sys
from pathlib import Path

import numpy as np
import mujoco
from scipy.spatial import cKDTree

DEFAULT_XML = Path(__file__).parent.parent / "models" / "humanoid_real_v2.xml"

# thresholds
MAX_QVEL = 50.0          # rad/s or m/s - above this = explosion
HOLD_ERR_DEG = 20.0      # max joint drift from held target after settling
RG_MAX = 0.20            # radius of gyration (m) above which an inertial looks inflated
KP, KD = 0.4, 0.05       # PD gains for the hold test


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def main(xml_path):
    print("=" * 64)
    print(f"Model integrity test: {xml_path}")
    print("=" * 64)
    results = []

    # 1. loads / compiles
    try:
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        d = mujoco.MjData(m)
        results.append(check("compiles", True, f"{m.nbody-1} bodies, {m.nu} actuators, nq={m.nq}"))
    except Exception as e:
        check("compiles", False, f"{type(e).__name__}: {e}")
        return 1

    # 2. single connected kinematic tree (every body traces back to world=0)
    disconnected = []
    for b in range(1, m.nbody):
        p, seen = b, set()
        while p != 0 and p not in seen:
            seen.add(p)
            p = m.body_parentid[p]
        if p != 0:
            disconnected.append(m.body(b).name)
    results.append(check("single kinematic tree", not disconnected,
                         "all bodies reach world" if not disconnected
                         else f"orphans: {disconnected}"))

    # 2b. component completeness: every declared mesh is actually placed as a geom
    placed = {m.geom_dataid[g] for g in range(m.ngeom)
              if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
    orphan_meshes = [m.mesh(i).name for i in range(m.nmesh) if i not in placed]
    results.append(check("all meshes placed (no missing parts)", not orphan_meshes,
                         f"{m.nmesh} meshes, all placed as geoms" if not orphan_meshes
                         else f"declared but never placed: {orphan_meshes}"))

    # 2c. no clumping: every non-root body is its own articulated link (has a joint)
    welded = [m.body(b).name for b in range(1, m.nbody)
              if not any(m.jnt_bodyid[j] == b for j in range(m.njnt))]
    results.append(check("no welded/clumped links (each body has a joint)", not welded,
                         f"{m.nbody-1} bodies all articulated" if not welded
                         else f"bodies with no joint: {welded}"))

    # 2d. DOF inventory + every joint actuated
    HINGE = mujoco.mjtJoint.mjJNT_HINGE
    n_hinge = int((m.jnt_type == HINGE).sum())
    n_free = int((m.jnt_type == mujoco.mjtJoint.mjJNT_FREE).sum())
    results.append(check("every articulated joint is actuated", m.nu == n_hinge,
                         f"{n_hinge} hinges, {n_free} free, {m.nu} actuators"))

    # 3. inertial sanity (mass > 0, radius of gyration not wildly inflated)
    bad_mass, inflated = [], []
    for b in range(1, m.nbody):
        mass = m.body_mass[b]
        if mass <= 0:
            bad_mass.append(m.body(b).name)
            continue
        rg = np.sqrt(m.body_inertia[b].max() / mass)
        if rg > RG_MAX:
            inflated.append(f"{m.body(b).name}={rg:.2f}m")
    results.append(check("masses positive", not bad_mass,
                         "" if not bad_mass else f"zero/neg mass: {bad_mass}"))
    results.append(check(f"inertials plausible (rg<{RG_MAX}m)", not inflated,
                         "all bodies plausible" if not inflated
                         else f"{len(inflated)} inflated: {inflated[:6]}"))

    # 4. passive rollout - no NaN / no explosion
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    nan = expl = False
    for _ in range(600):
        mujoco.mj_step(m, d)
        if np.isnan(d.qpos).any():
            nan = True
            break
        if np.abs(d.qvel).max() > MAX_QVEL:
            expl = True
            break
    results.append(check("passive: no NaN", not nan))
    results.append(check("passive: no explosion", not expl,
                         f"max|qvel|={np.abs(d.qvel).max():.2f}"))

    # 5. no RESTING self-collision: at the home pose no two links may interpenetrate
    #    (designed overlaps must be <contact><exclude>'d; dynamic contacts while the
    #    robot moves are expected and desirable - that is collision *working*)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    home_sc = sum(1 for i in range(d.ncon)
                  if m.geom_bodyid[d.contact[i].geom1] > 0 and m.geom_bodyid[d.contact[i].geom2] > 0)
    results.append(check("no resting self-collision (home pose)", home_sc == 0,
                         f"{home_sc} body-body contacts at rest"))

    # 6. PD-hold - under a controller, do the joints stay assembled?
    #    Position actuators (ctrl = target angle) hold the pose directly; torque
    #    motors (ctrl = torque) get an explicit PD law. Detect which this model uses.
    mujoco.mj_resetDataKeyframe(m, d, 0)
    qadr = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
    vadr = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
    mujoco.mj_forward(m, d)
    qtar = d.qpos[qadr].copy()
    is_position = bool((m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE).any())
    hold_nan = False
    for _ in range(1000):
        if is_position:
            d.ctrl[:] = qtar                       # position servo: command home angles
        else:
            err = qtar - d.qpos[qadr]
            d.ctrl[:] = np.clip(KP * err - KD * d.qvel[vadr], -1, 1)
        mujoco.mj_step(m, d)
        if np.isnan(d.qpos).any():
            hold_nan = True
            break
    final_err = np.degrees(np.abs(qtar - d.qpos[qadr]).max()) if not hold_nan else float("nan")
    results.append(check(f"PD-hold keeps joints assembled (<{HOLD_ERR_DEG} deg)",
                         (not hold_nan) and final_err < HOLD_ERR_DEG,
                         f"max joint drift={final_err:.1f} deg, settled|qvel|={np.abs(d.qvel).max():.3f}"))

    # 7. geometric connectivity - every link's mesh actually touches its parent's mesh
    #    (catches a segment that is articulated in the tree but floating in space)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    def body_world_verts(b):
        out = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].reshape(-1, 3)
            out.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        return np.vstack(out) if out else np.zeros((0, 3))

    def min_gap_mm(cv, pv):
        # nearest distance between two vertex sets (subsampled), via KDTree
        ci = cv[::max(1, len(cv) // 1500)]
        pi = pv[::max(1, len(pv) // 1500)]
        return float(cKDTree(pi).query(ci)[0].min()) * 1000

    gaps = []
    for b in range(1, m.nbody):
        p = m.body_parentid[b]
        if p == 0:
            continue
        cv, pv = body_world_verts(b), body_world_verts(p)
        if len(cv) == 0 or len(pv) == 0:
            continue
        if min_gap_mm(cv, pv) > 15:
            gaps.append(f"{m.body(b).name}<->{m.body(p).name}={min_gap_mm(cv, pv):.0f}mm")
    results.append(check("geometric connectivity (links touch parents, <15mm)", not gaps,
                         "all child links touch their parent" if not gaps
                         else f"floating links: {gaps}"))

    # 8. joint pivots are at the real hinge: rotating a joint must NOT tear the child
    #    away from its parent. Catches misplaced 'joint pos' (link origin not baked) -
    #    parts touch at the home pose but fly apart the instant the joint moves.
    torn = []
    for j in range(m.njnt):
        if m.jnt_type[j] != HINGE:
            continue
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[m.jnt_qposadr[j]] += 0.4
        mujoco.mj_forward(m, d)
        b = m.jnt_bodyid[j]
        p = m.body_parentid[b]
        cv, pv = body_world_verts(b), body_world_verts(p)
        if len(cv) == 0 or len(pv) == 0:
            continue
        g = min_gap_mm(cv, pv)
        if g > 30:
            torn.append(f"{m.joint(j).name}={g:.0f}mm")
    results.append(check("joint pivots correct (child stays attached when rotated, <30mm)", not torn,
                         "all joints rotate about the real hinge" if not torn
                         else f"pivot off -> tears away: {torn}"))

    print("-" * 64)
    passed = all(results)
    print(f"RESULT: {'PASS - model is structurally sound' if passed else 'FAIL - see above'}"
          f"  ({sum(results)}/{len(results)} checks)")
    return 0 if passed else 1


if __name__ == "__main__":
    xml = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_XML
    sys.exit(main(xml))
