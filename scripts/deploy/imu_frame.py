#!/usr/bin/env python3
# imu_frame.py
"""
Transform IMU readings from the CHEST body (where the IMU is physically mounted) into the
PELVIS / base_link frame that the policy's observation is defined in.

The chest (MJCF body 0003_8) sits above the 3 waist joints, which are bent even at the home
pose (~7.3 deg), so a chest IMU reads a tilted gravity the policy would fight. Using the live
waist encoder angles we rotate proj_grav / ang_vel back into the pelvis frame.

Validated against humanoid_real_v2.xml: the composed rotation equals the sim's chest->pelvis
to 1e-16 and recovers proj_grav exactly (7.3 deg uncorrected tilt at home).

NOTE: ang_vel is rotation-only here (R @ gyro). The exact pelvis rate also subtracts the waist
joint-rate contribution, but that is ~0 during standing and would inject finite-diff noise, so
it is omitted in v1.
"""
import numpy as np

# Waist joint axes in the pelvis-aligned frame (from humanoid_real_v2.xml):
#   idx 8  waist_roll  = Revolute 22, axis (-1, 0, 0)
#   idx 9  waist_pitch = Revolute 20, axis (0, -0.999979, 0.006517)
#   idx 10 waist_yaw   = Revolute 19, axis (0,  0.006517, 0.999979)
A_ROLL = np.array([-1.0, 0.0, 0.0])
A_PITCH = np.array([0.0, -0.999979, 0.006517])
A_YAW = np.array([0.0, 0.006517, 0.999979])


def _rot(axis, theta):
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def chest_to_pelvis(waist_roll, waist_pitch, waist_yaw) -> np.ndarray:
    """R (3x3) such that v_pelvis = R @ v_chest, from the 3 waist angles (sim radians, 0=straight)."""
    return (_rot(A_ROLL, waist_roll) @ _rot(A_PITCH, waist_pitch) @ _rot(A_YAW, waist_yaw)).astype(np.float32)
