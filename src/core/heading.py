"""Shared heading-source yaw-rate measurement for the waist-twist (WTR) ablation.

Used IDENTICALLY by the Humanoid-v5 walking env and the real-robot turning env so the
torso-vs-pelvis `heading_source` toggle is *literally the same code path* on both bodies --
that identity is the scientific claim of the hardware validation.

CRITICAL -- the freejoint root differs by morphology, and the two morphologies are mirrored:

    body         freejoint root      torso source        pelvis source
    -----------  ------------------  ------------------  ------------------
    Humanoid-v5  torso (upper)       qvel[5] (root)      objVelocity('pelvis')
    real robot   base_link = pelvis  objVelocity('0003_8')  qvel[5] (root)

`qvel[5]` is the freejoint's *local-frame* angular-velocity z = the yaw rate of WHATEVER
body the freejoint is attached to. So the fast-path is keyed on which source IS the root
(`root_is_torso`), NOT on the label "torso." The non-root source uses
`mj_objectVelocity(flag=0)` = world-frame yaw z. These are NOT interchangeable (under tilt,
local qvel[5] != world objVel z), so each source keeps its exact original op -> the sim
numbers stay bit-reproducible across the refactor.
"""
from __future__ import annotations

import numpy as np
import mujoco


class HeadingYaw:
    """Resolve the actual yaw rate of the configured heading source body.

    Args:
        heading_source: 'torso' or 'pelvis' (the experimental variable).
        torso_body, pelvis_body: MuJoCo body names for each role in THIS morphology.
        root_is_torso: True if the freejoint root body is the torso (Humanoid-v5),
            False if the root is the pelvis (real robot, base_link).
    """

    def __init__(self, heading_source: str, torso_body: str, pelvis_body: str,
                 root_is_torso: bool):
        self.source = str(heading_source).lower()
        if self.source not in ("torso", "pelvis"):
            print(f"WARNING: heading_source={self.source!r} not in (torso, pelvis); using 'torso'")
            self.source = "torso"
        self.torso_body = torso_body
        self.pelvis_body = pelvis_body
        self.root_is_torso = bool(root_is_torso)
        self._body_id = None  # lazily resolved id of the NON-root source body

    def source_is_root(self) -> bool:
        """True when the selected source body is the freejoint root (use qvel[5])."""
        return (self.source == "torso") == self.root_is_torso

    def actual_yaw_rate(self, model, data) -> float:
        # Root source: freejoint local-frame angular-velocity z (exact original fast-path).
        if self.source_is_root():
            return float(data.qvel[5])

        # Non-root source: world-frame yaw rate of the named body via mj_objectVelocity.
        if self._body_id is None:
            name = self.torso_body if self.source == "torso" else self.pelvis_body
            for i in range(model.nbody):
                if model.body(i).name == name:
                    self._body_id = i
                    break
            if self._body_id is None:
                print(f"WARNING: heading body {name!r} not found; falling back to root qvel[5]")
                return float(data.qvel[5])

        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                 int(self._body_id), vel6, 0)  # flag 0 = world frame
        return float(vel6[2])
