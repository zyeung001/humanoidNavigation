"""The joint commands the REAL robot can be sent, in sim radians (action order).

On the robot, scripts/deploy/sim_real_map.py turns every command into servo units and clips it
to that joint's `servo_limit` in config/joint_servo_map.yaml (after clipping to its sim_range).
The MJCF joint ranges are wider than those limits in places -- the knees are +-90 deg in sim but
can only be commanded to about -3 deg (straight) on the robot, the waist roll +35 in sim against
+22 -- so a policy trained without this clip can lean on commands the robot will never carry
out. Measured 11/17 on walk_step_to_v4 @50M: knees hyperextended to -13 deg ~46% of the time,
waist roll past +22 deg 14% of the time. StandingEnv applies these limits when
`clip_to_servo_limits` is set.
"""
from __future__ import annotations

import numpy as np
import yaml


def real_command_limits(map_path: str, n_joints: int):
    """(lo, hi) rad per action index: servo_limit converted with centre/sign/units_per_rad,
    intersected with sim_range -- exactly what rad_to_units lets through."""
    m = yaml.safe_load(open(map_path, encoding="utf-8"))
    upr = float(m["units_per_rad"])
    lo = np.full(n_joints, -np.inf, dtype=np.float32)
    hi = np.full(n_joints, np.inf, dtype=np.float32)
    for j in m["joints"]:
        c = float(j.get("center", m["servo_center"]))
        a = (j["servo_limit"][0] - c) * j["sign"] / upr
        b = (j["servo_limit"][1] - c) * j["sign"] / upr
        i = int(j["idx"])
        lo[i] = max(min(a, b), j["sim_range"][0])
        hi[i] = min(max(a, b), j["sim_range"][1])
    if not np.all(np.isfinite(lo) & np.isfinite(hi)):
        raise ValueError(f"{map_path} does not cover all {n_joints} joints")
    return lo, hi
