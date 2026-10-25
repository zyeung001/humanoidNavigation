#!/usr/bin/env python3
# posture_control.py
"""
Slow posture correction for standing: keep the weight centred over the feet. NumPy only; the Pi
runs this file and the sim suite imports the same file (src/robustness/policies.PosturePolicy).

WHY. The robot has no ankles and slow servos. A fast tilt-feedback law on the hips (the 10/17
feedback prior) helped in sim and made the real robot fall more easily both ways (10/18): it
moved the weight the wrong way at the wrong time. What a slow controller CAN do is shift the
weight back over the feet when the body leans slowly -- a tilted floor, a load on its back, the
slow forward creep seen on the robot -- without reacting to fast rocking at all.

HOW. The pelvis pitch (from the IMU's projected gravity) is low-pass filtered with a time
constant of about a second, compared with the stance's resting pitch, and the error bends both
knees: knee + moves the weight toward the heel (~4.8 mm per degree in sim, measured settled), so
leaning forward bends the knees and leaning back straightens them. The knee change is bounded;
the knees can only straighten ~3 deg from the 'knee2' stance before their servo limit.

The correction tilts the pelvis itself: with the feet flat, knees +1 deg pitch the pelvis back
~1.34 deg (sim, 10/24), so fed back raw it stops short of centring the weight. self_tilt removes
that part (lean = pitch - self_tilt * correction), but it ONLY holds with the feet flat and the
knees tracking: on the robot (10/25 log) the knees barely followed small commands and during a fall
the feet rock on an edge, so the term became positive feedback -- the knees sat bent +5 deg on
average while the robot was falling backward, and it swung between the limits for a minute. The
robot config sets it to 0.

PUSHES ARE NOT ITS JOB. The 160 mm feet ride out a push on their own; a push swings the pitch
10-30 deg and back in 1-2 s, and any slow controller that responds to it responds a second late,
in the rebound, and throws the robot the other way. So: a deadband (no correction for the small
leans the feet hold anyway), transient_deg (each sample may pull the filter only that far, so a
fast swing barely registers while a slow lean passes), and max_rate_deg_s (the knees move
smoothly, never jump).

ZERO. The controller treats its zero as "upright", so an IMU that sits a few degrees off on its
mount (it has moved twice on this robot) would make it lean the robot on purpose: in sim a 4 deg
IMU error cost 30% of the pushed-standing time. With zero_deg None (the default for the robot)
it measures the zero itself: no correction until it has seen a still stretch (window_s long,
lean spread under max_spread_deg) starting settle_s after reset, and the zero is that stretch's
mean lean. A constant IMU offset then cancels out. Leave the robot untouched for those seconds.

  ctrl = PostureController.from_config(cfg_dict, dt=0.025)
  ctrl.reset()
  command = stance + ctrl.offsets(proj_grav)       # every control step
"""

from __future__ import annotations

import numpy as np

KNEES = (3, 7)          # R_knee, L_knee in actuator order


def pitch_deg(proj_grav) -> float:
    """Pelvis pitch from projected gravity, + = leaning forward (the convention pose_sweep shows)."""
    return float(np.degrees(np.arctan2(proj_grav[0], -proj_grav[2])))


class PostureController:
    def __init__(self, gain: float, tau_s: float, zero_deg, min_deg: float, max_deg: float,
                 dt: float, self_tilt: float = 0.0, settle_s: float = 1.0, window_s: float = 1.5,
                 max_spread_deg: float = 1.5, deadband_deg: float = 0.0, max_rate_deg_s=None,
                 transient_deg=None, n_joints: int = 17, joints=KNEES):
        self.gain = float(gain)            # knee degrees per degree of filtered lean error
        self.tau_s = float(tau_s)          # low-pass time constant
        self.fixed_zero = None if zero_deg is None else float(zero_deg)   # None: measure it
        self.min_deg, self.max_deg = float(min_deg), float(max_deg)
        self.self_tilt = float(self_tilt)  # pelvis pitch deg caused per deg of correction
        self.deadband_deg = float(deadband_deg)       # lean the robot rides out on its feet alone
        self.max_rate_deg_s = None if max_rate_deg_s is None else float(max_rate_deg_s)
        self.transient_deg = None if transient_deg is None else float(transient_deg)
        self.settle_s, self.window_s = float(settle_s), float(window_s)
        self.max_spread_deg = float(max_spread_deg)
        self.dt = float(dt)
        self.n_joints = int(n_joints)
        self.joints = tuple(joints)
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict, dt: float, n_joints: int = 17) -> "PostureController":
        return cls(gain=cfg["gain"], tau_s=cfg["tau_s"], zero_deg=cfg.get("zero_deg"),
                   min_deg=cfg["min_deg"], max_deg=cfg["max_deg"], dt=dt,
                   self_tilt=cfg.get("self_tilt", 0.0), settle_s=cfg.get("settle_s", 1.0),
                   window_s=cfg.get("window_s", 1.5), max_spread_deg=cfg.get("max_spread_deg", 1.5),
                   deadband_deg=cfg.get("deadband_deg", 0.0), max_rate_deg_s=cfg.get("max_rate_deg_s"),
                   transient_deg=cfg.get("transient_deg"),
                   n_joints=n_joints, joints=tuple(cfg.get("joints", KNEES)))

    def reset(self) -> None:
        self._lean_f = None
        self._corr = 0.0
        self._t = 0
        self.zero_deg = self.fixed_zero
        self._window = []

    @property
    def engaged(self) -> bool:
        return self.zero_deg is not None

    @property
    def filtered_lean(self):
        return self._lean_f

    @property
    def correction(self) -> float:
        """The knee correction now in effect, degrees."""
        return self._corr

    def _measure_zero(self, lean: float) -> None:
        if self._t * self.dt < self.settle_s:
            return
        self._window.append(lean)
        if np.ptp(self._window) > self.max_spread_deg:       # it moved: start the window again
            self._window = [lean]
        if len(self._window) * self.dt >= self.window_s:
            self.zero_deg = float(np.mean(self._window))

    def offsets(self, proj_grav) -> np.ndarray:
        """Joint-command offsets (sim radians) to add to the stance this step."""
        lean = pitch_deg(proj_grav) - self.self_tilt * self._corr
        if self._lean_f is None:
            self._lean_f = lean
        else:
            if self.engaged and self.transient_deg is not None:
                # A push or a fall swings the pitch 10-30 deg for a second or two and the feet bring
                # it back; reacting to that a second late is what threw the robot the other way.
                # Each sample may pull the filter only transient_deg's worth, so slow leans pass and
                # fast swings barely register.
                lean = float(np.clip(lean, self._lean_f - self.transient_deg, self._lean_f + self.transient_deg))
            self._lean_f += (self.dt / (self.tau_s + self.dt)) * (lean - self._lean_f)
        if self.zero_deg is None:
            # The FILTERED lean: raw IMU samples scatter ~1 deg on their own, so a stillness test on
            # them never passes (in sim it never engaged at all).
            self._measure_zero(self._lean_f)
        self._t += 1
        if self.zero_deg is not None:
            err = self._lean_f - self.zero_deg
            err = float(np.sign(err)) * max(abs(err) - self.deadband_deg, 0.0)
            target = float(np.clip(self.gain * err, self.min_deg, self.max_deg))
            if self.max_rate_deg_s is not None:      # knees move smoothly, never a jump
                step = self.max_rate_deg_s * self.dt
                target = float(np.clip(target, self._corr - step, self._corr + step))
            self._corr = target
        out = np.zeros(self.n_joints, dtype=np.float32)
        out[list(self.joints)] = np.radians(self._corr)
        return out
