"""A fixed feedback law the policy acts ON TOP OF: command = policy + K @ [proj_grav, gyro].

WHY (10/17). scratch_base-1016-2124 trained 60M steps on the fixed plant and tied commanding
the straight pose with no feedback at all (68.0 vs 68.6; held-out -1.0 +- 1.2). Its reward rose
only because its action-rate and saturation penalties shrank. Yet a two-gain law on the hips,
hip_pitch += 0.5 * pg_x + 0.1 * gyro_y, beats that pose by +6.0 +- 1.1 on held-out seeds:
balance this robot CAN do, which PPO did not find. With the law under the policy, an untrained net
already balances that well, and training only has to improve on it.

The term uses the newest observation frame exactly as the policy sees it (noise, bias and
sensor faults included), so the env with a prior and policy pi behaves bit-for-bit like the env
without one and policy pi + K @ s. On the robot, deploy_standing.py applies the same K from the
.npz (export_policy.py writes it), so a policy can never run without the law it was trained on.

Config, by joint name (config/joint_servo_map.yaml) and signal:

    feedback_prior:
      R_hip_pitch: {pg_x: 0.5, gyro_y: 0.1}
      L_hip_pitch: {pg_x: 0.5, gyro_y: 0.1}

or the (n_joints, 6) matrix itself, which is how an exported .npz hands it back.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# The first six entries of every proprioceptive frame, in order.
SIGNALS = ("pg_x", "pg_y", "pg_z", "gyro_x", "gyro_y", "gyro_z")


def joint_names(joint_map: str = "config/joint_servo_map.yaml") -> list:
    """Joint names in ACTION order -- the order the deploy loop drives the servos in."""
    import yaml  # noqa: PLC0415
    return [j["dof"] for j in yaml.safe_load(open(joint_map))["joints"]]


def gain_matrix(spec, names: Sequence[str]) -> Optional[np.ndarray]:
    """(n_joints, 6) float32 gains from a config entry, or None when there is no prior."""
    if spec is None or (isinstance(spec, dict) and not spec):
        return None
    n = len(names)
    if not isinstance(spec, dict):
        k = np.asarray(spec, dtype=np.float32)
        if k.shape != (n, len(SIGNALS)):
            raise ValueError(f"feedback_prior matrix is {k.shape}, expected {(n, len(SIGNALS))}")
        return None if not np.any(k) else k
    k = np.zeros((n, len(SIGNALS)), dtype=np.float32)
    for joint, gains in spec.items():
        if joint not in names:
            raise ValueError(f"feedback_prior: unknown joint {joint!r} (joints: {list(names)})")
        for signal, g in gains.items():
            if signal not in SIGNALS:
                raise ValueError(f"feedback_prior: unknown signal {signal!r} (signals: {SIGNALS})")
            k[names.index(joint), SIGNALS.index(signal)] = float(g)
    return k


def deadband_vector(spec) -> Optional[np.ndarray]:
    """(6,) per-signal deadbands from `feedback_prior_deadband: {gyro_y: 0.05}`, or None.

    WHY (10/25). The knee law (knees += 0.5 pg_x + 0.8 gyro_y) stood the robot untethered through
    ~20 pushes, but standing still it twitched: the robot's gyro reads ~0.034 rad/s of noise, which
    times 0.8 is ~1.6 deg of knee command flicking at 1-3 Hz. Pushes drive the gyro to 0.3-0.8, so
    a small dead zone removes the twitch and keeps the push response."""
    if not spec:
        return None
    if not isinstance(spec, dict):
        db = np.asarray(spec, dtype=np.float32)
        return None if not np.any(db) else db
    db = np.zeros(len(SIGNALS), dtype=np.float32)
    for signal, v in spec.items():
        if signal not in SIGNALS:
            raise ValueError(f"feedback_prior_deadband: unknown signal {signal!r} (signals: {SIGNALS})")
        db[SIGNALS.index(signal)] = float(v)
    return db


def apply_deadband(signals: np.ndarray, db: Optional[np.ndarray]) -> np.ndarray:
    """Soft dead zone: |s| <= db reads 0, beyond it db is taken off, so the law never jumps.
    scripts/deploy/numpy_policy.py repeats this line for the robot (it cannot import src/)."""
    if db is None:
        return signals
    return (np.sign(signals) * np.maximum(np.abs(signals) - db, 0.0)).astype(np.float32)


def describe(k: Optional[np.ndarray], names: Sequence[str]) -> str:
    if k is None:
        return "none"
    terms = []
    for i, j in zip(*np.nonzero(k)):
        terms.append(f"{names[i]} {k[i, j]:+g}*{SIGNALS[j]}")
    return ", ".join(terms)
