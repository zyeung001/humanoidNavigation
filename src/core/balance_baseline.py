"""Tilt-feedback balance baseline for the no-ankle humanoid (PD-baseline + RL-residual).

The pure RL policy can't reliably catch a body lean on hardware: with ~50 ms servo dead time
its correction arrives a phase-lag too late, so the balance loop limit-cycles / slowly diverges.
This baseline adds an IMMEDIATE, correctly-phased restoring command straight from the IMU
(projected gravity = lean, gyro = lean rate): lean forward -> corrective hip-pitch, lean
sideways -> corrective hip-roll. The RL policy then only outputs a small RESIDUAL on top, so its
authority is bounded and it cannot drive the instability. The SAME object runs in sim (training)
and on hardware (deploy), so there is no sim2real gap in the baseline.

Recovery on a no-ankle biped is the HIP strategy, and the sign of the tilt->joint map is not
obvious to derive, so it is PARAMETERIZED and tuned in sim (scripts/find_balance_signs.py): the
correct sign is whichever makes the baseline ALONE reduce a push-induced fall.

Output is an additive delta in sim joint-angle space (radians), applied as
    target = nominal_pose(=straight, joints 0) + baseline.delta(tilt) + rl_residual
"""
import numpy as np


class BalanceBaseline:
    def __init__(self, hip_pitch_idx, hip_roll_idx,
                 kp_pitch=0.0, kd_pitch=0.0, kp_roll=0.0, kd_roll=0.0,
                 sign_pitch=1.0, sign_roll=1.0, roll_mirror=1.0,
                 delta_clip=0.6):
        # hip_pitch_idx / hip_roll_idx: [right_idx, left_idx] into the n-joint action vector
        self.hp = list(hip_pitch_idx)
        self.hr = list(hip_roll_idx)
        self.kp_pitch = float(kp_pitch)
        self.kd_pitch = float(kd_pitch)
        self.kp_roll = float(kp_roll)
        self.kd_roll = float(kd_roll)
        self.sign_pitch = float(sign_pitch)
        self.sign_roll = float(sign_roll)
        self.roll_mirror = float(roll_mirror)   # +1: both rolls same sign; -1: mirrored
        self.delta_clip = float(delta_clip)

    def delta(self, proj_grav, ang_vel, n_joints):
        """proj_grav: body-frame gravity (upright ~ [0,0,-1]); ang_vel: body gyro [wx,wy,wz]."""
        out = np.zeros(n_joints, dtype=np.float32)
        tilt_fwd = float(proj_grav[0])    # +x forward lean component (0 upright)
        tilt_lat = float(proj_grav[1])    # +y left lean component
        pitch_rate = float(ang_vel[1])    # gyro about y (pitch)
        roll_rate = float(ang_vel[0])     # gyro about x (roll)
        bp = self.sign_pitch * (self.kp_pitch * tilt_fwd + self.kd_pitch * pitch_rate)
        br = self.sign_roll * (self.kp_roll * tilt_lat + self.kd_roll * roll_rate)
        bp = float(np.clip(bp, -self.delta_clip, self.delta_clip))
        br = float(np.clip(br, -self.delta_clip, self.delta_clip))
        out[self.hp[0]] += bp
        out[self.hp[1]] += bp
        out[self.hr[0]] += br
        out[self.hr[1]] += self.roll_mirror * br
        return out
