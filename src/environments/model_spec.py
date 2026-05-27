"""
Model introspection for custom MuJoCo humanoids.

The standing/walking envs hardcode constants tuned for the default
Humanoid-v5 model: an extra "+15" obs-dim correction, a 1.40m target height,
foot bodies [6, 9], etc. Those numbers are wrong for any other morphology.

When a custom xml_file is loaded, this module derives the equivalent
constants by inspecting the compiled model at its home keyframe. When no
xml_file is supplied, the envs keep the legacy hardcoded path so the
standard-humanoid training is unaffected.
"""
from dataclasses import dataclass
from typing import List, Optional
import os

import numpy as np
import gymnasium as gym
import mujoco


@dataclass
class ModelSpec:
    xml_file: str
    body_names: List[str]
    joint_names: List[str]
    foot_body_ids: List[int]        # 2 bodies that own the lowest geoms at standing
    standing_com_z: float            # subtree_com[0][2] at home keyframe — "standing height"
    standing_qpos_z: float           # root freejoint z at home keyframe
    actual_obs_dim: int              # measured by env.reset() — replaces the +15 magic constant
    lowest_geom_z: float             # for diagnostics + keyframe lift fixes
    has_home_keyframe: bool
    n_bodies: int
    n_actuators: int


def is_custom_xml(xml_file: Optional[str]) -> bool:
    return bool(xml_file)


def introspect_model(xml_file: str) -> ModelSpec:
    abs_path = os.path.abspath(xml_file)
    m = mujoco.MjModel.from_xml_path(abs_path)
    d = mujoco.MjData(m)

    has_home = m.nkey > 0
    if has_home:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)

    body_names = [m.body(i).name for i in range(m.nbody)]
    joint_names = [m.joint(i).name for i in range(m.njnt)]

    geom_z = d.geom_xpos[:, 2].copy()
    geom_body = m.geom_bodyid
    # Exclude geoms that belong to the world body (floor, lights, etc.).
    non_world = geom_body != 0
    foot_bodies: list[int] = []
    for gi in np.argsort(geom_z):
        if not non_world[gi]:
            continue
        bid = int(geom_body[gi])
        if bid not in foot_bodies:
            foot_bodies.append(bid)
        if len(foot_bodies) == 2:
            break
    foot_bodies.sort(key=lambda bid: d.xpos[bid][1])  # stable left/right order

    non_world_geom_z = geom_z[non_world]
    lowest_geom_z = float(non_world_geom_z.min()) if non_world_geom_z.size else 0.0

    standing_com_z = float(d.subtree_com[0][2])
    standing_qpos_z = float(d.qpos[2]) if d.qpos.shape[0] >= 3 else 0.0

    env = gym.make(
        "Humanoid-v5",
        xml_file=abs_path,
        exclude_current_positions_from_observation=False,
    )
    obs, _ = env.reset()
    actual_obs_dim = int(obs.shape[0])
    env.close()

    return ModelSpec(
        xml_file=abs_path,
        body_names=body_names,
        joint_names=joint_names,
        foot_body_ids=foot_bodies,
        standing_com_z=standing_com_z,
        standing_qpos_z=standing_qpos_z,
        actual_obs_dim=actual_obs_dim,
        lowest_geom_z=lowest_geom_z,
        has_home_keyframe=has_home,
        n_bodies=m.nbody - 1,
        n_actuators=int(m.nu),
    )


def summarize(spec: ModelSpec) -> str:
    return (
        f"ModelSpec({os.path.basename(spec.xml_file)}): "
        f"bodies={spec.n_bodies} actuators={spec.n_actuators} "
        f"obs_dim={spec.actual_obs_dim} "
        f"standing_com_z={spec.standing_com_z:.3f}m "
        f"qpos_z={spec.standing_qpos_z:.3f}m "
        f"feet={spec.foot_body_ids} "
        f"lowest_geom_z={spec.lowest_geom_z:.3f}m"
    )
