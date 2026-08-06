# arm_reach_env.py
"""Arm pose-reach environment for the real humanoid: the closed-loop sim2real stack demo.

Purpose (7/28): before betting everything on balance, prove the deploy stack end-to-end on a
task that cannot fall. The policy controls ONLY the 6 arm joints (servos 12-17); the pelvis
is kinematically held (the robot is seated / supported on hardware) and the legs are parked
at straight. Task: drive the arms to a commanded pose and hold it against gravity and
perturbation kicks -- on hardware, push the arm down and it must return. That behavior is
impossible open-loop, so it is an unfakeable demonstration that the RL reads the sensors.

Deployability contract (mirrored by scripts/deploy/deploy_arm_reach.py):
  - 40 Hz control, tau-EMA action smoothing, absolute radian targets (position servos).
  - Obs frame (24) = [target(6) | jpos(6) | jvel(6) | last_action(6)], 4-frame history = 96.
    jpos is the ABSOLUTE joint angle (0 = straight), NOT relative to the standing keyframe.
  - Targets sampled inside the SERVO EEPROM limits from config/joint_servo_map.yaml (the
    sim joint ranges are wider than the physical limits; training beyond them is wasted).
  - Actuator lag (dead time + first-order tau) randomized per episode, as measured on the
    real SCS servos; obs noise on jpos/jvel.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

N_ARM = 6
FRAME = 4 * N_ARM           # target | jpos | jvel | last_action
HISTORY = 4
OBS_DIM = FRAME * HISTORY   # 96


def arm_joint_specs(map_path):
    """The 6 arm entries (idx 11-16) of joint_servo_map.yaml, with the reachable sim-radian
    range implied by the servo EEPROM limits (tighter than the MJCF joint range)."""
    with open(map_path) as f:
        m = yaml.safe_load(f)
    upr = float(m["units_per_rad"])
    default_center = int(m.get("servo_center", 512))
    specs = []
    for j in m["joints"]:
        if j["idx"] < 11:
            continue
        center = int(j.get("center", default_center))
        sign = int(j["sign"])
        lo_u, hi_u = j["servo_limit"]
        a = sign * (lo_u - center) / upr
        b = sign * (hi_u - center) / upr
        specs.append({
            "mjcf": j["mjcf"], "dof": j["dof"], "idx": j["idx"],
            "servo_id": j["servo_id"],
            "reach_lo": min(a, b), "reach_hi": max(a, b),
        })
    assert len(specs) == N_ARM, f"expected {N_ARM} arm joints, got {len(specs)}"
    return specs


class ArmReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        cfg = dict(config or {})
        self.cfg = cfg
        # v2_armfix = v2 with the LEFT shoulder roll's mis-converted range corrected
        # ([-2.007, 0] -> [-0.4363, +1.5708]); without it the left arm has zero outward
        # travel in sim although the real servo moves symmetrically (bench-verified).
        xml = str(ROOT / cfg.get("xml_file", "models/humanoid_real_v2_armfix.xml"))
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.dt = float(cfg.get("control_dt", 0.025))                 # 40 Hz
        self.n_sub = max(1, round(self.dt / self.model.opt.timestep))

        specs = arm_joint_specs(ROOT / cfg.get("servo_map", "config/joint_servo_map.yaml"))
        self.specs = specs
        self.qadr = np.array([self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, s["mjcf"])]
            for s in specs])
        self.vadr = np.array([self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, s["mjcf"])]
            for s in specs])
        self.aadr = np.array([mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, s["mjcf"]) for s in specs])
        assert (self.qadr >= 0).all() and (self.aadr >= 0).all()
        margin = float(cfg.get("target_margin", 0.9))     # stay off the servo hard stops
        self.reach_lo = np.array([s["reach_lo"] for s in specs], dtype=np.float32) * margin
        self.reach_hi = np.array([s["reach_hi"] for s in specs], dtype=np.float32) * margin

        # everything that is NOT an arm actuator is parked at 0 (sim straight)
        self.non_arm_act = np.array([i for i in range(self.model.nu) if i not in self.aadr])

        self.tau = float(cfg.get("action_smoothing_tau", 0.3))
        self.max_steps = int(cfg.get("max_episode_steps", 800))
        self.hold_range = cfg.get("target_hold_steps", [120, 200])    # 3-5 s per target
        self.push_range = cfg.get("push_interval_steps", [80, 160])
        self.push_std = float(cfg.get("push_jvel_std", 1.5))          # rad/s kick on arm joints
        self.noise_jpos = float(cfg.get("noise_jpos", 0.005))
        self.noise_jvel = float(cfg.get("noise_jvel", 0.25))
        self.delay_ms = cfg.get("actuator_delay_ms", [30.0, 70.0])
        self.tau_ms = cfg.get("actuator_tau_ms", [90.0, 220.0])
        # Two-scale tracking kernel. A single soft kernel (the 7/28 first cut used
        # 3*exp(-6*mse)) keeps ~80% of the reward at 0.2 rad error, so PPO plateaus there;
        # the sharp precision term only pays out near zero error and supplies the gradient
        # that pulls the last 0.2 rad in.
        self.track_bw = float(cfg.get("reward_track_bandwidth", 8.0))
        self.w_track = float(cfg.get("reward_track_weight", 2.0))
        self.prec_bw = float(cfg.get("reward_precision_bandwidth", 80.0))
        self.w_prec = float(cfg.get("reward_precision_weight", 3.0))
        self.w_raw_rate = float(cfg.get("raw_action_rate_penalty", 0.5))
        self.w_jvel = float(cfg.get("jvel_penalty", 0.02))
        self.hold_bonus_tol = float(cfg.get("hold_bonus_tol", 0.08))  # rad, per joint
        self.hold_bonus = float(cfg.get("hold_bonus", 1.0))

        self.action_space = gym.spaces.Box(self.reach_lo, self.reach_hi, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self._root_qpos = None

    # ---- helpers ----
    def _frame(self):
        jpos = self.data.qpos[self.qadr].astype(np.float32)
        jvel = self.data.qvel[self.vadr].astype(np.float32)
        if self.noise_jpos > 0:
            jpos = jpos + self.rng.normal(0, self.noise_jpos, N_ARM).astype(np.float32)
        if self.noise_jvel > 0:
            jvel = jvel + self.rng.normal(0, self.noise_jvel, N_ARM).astype(np.float32)
        return np.concatenate([self.target, jpos, jvel, self.last_action]).astype(np.float32)

    def _obs(self):
        self.hist.append(self._frame())
        frames = list(self.hist)
        if len(frames) < HISTORY:
            frames = [np.zeros(FRAME, dtype=np.float32)] * (HISTORY - len(frames)) + frames
        return np.concatenate(frames)

    def _gravity_feasible(self, cand):
        """True if cand is REACHABLE AND HOLDABLE from the current arm state: command it
        directly for ~1.5 s of sim and require convergence (final error < 0.08 rad/joint).
        The weak arm servos (forcerange ~0.23 N*m) cannot hold every geometric pose against
        gravity, self-collision with the torso/mounts blocks some paths, and multi-joint
        combos change the gravity levers -- a static check misses all of that. Poses that
        fail would just saturate-and-sag (or stall on a mount), which trains nothing."""
        q, v, c0 = self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy()
        self.data.ctrl[self.non_arm_act] = 0.0
        self.data.ctrl[self.aadr] = cand
        for _ in range(160):                        # ~4 s: the kp=5 servos are slow movers
            for _ in range(self.n_sub):
                mujoco.mj_step(self.model, self.data)
            self.data.qpos[0:7] = self._root_qpos
            self.data.qvel[0:6] = 0.0
        ok = float(np.abs(self.data.qpos[self.qadr] - cand).max()) < 0.08
        self.data.qpos[:], self.data.qvel[:], self.data.ctrl[:] = q, v, c0
        mujoco.mj_forward(self.model, self.data)
        return ok

    def _sample_target(self):
        for _ in range(50):
            cand = self.rng.uniform(self.reach_lo, self.reach_hi).astype(np.float32)
            if self._gravity_feasible(cand):
                break
        else:
            cand = np.zeros(N_ARM, dtype=np.float32)   # straight down is always holdable
        self.target = cand
        self.hold_left = int(self.rng.integers(self.hold_range[0], self.hold_range[1] + 1))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed if seed is not None
                                         else int(self.np_random.integers(2 ** 31)))
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = 0.0                    # everything straight
        self.data.qpos[2] += 0.5                    # lift so the feet dangle (held/seated)
        self.data.qvel[:] = 0.0
        self._root_qpos = self.data.qpos[0:7].copy()
        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(N_ARM, dtype=np.float32)
        self._prev_raw = np.zeros(N_ARM, dtype=np.float32)
        self.hist = deque(maxlen=HISTORY)
        self._sample_target()
        self.push_left = int(self.rng.integers(self.push_range[0], self.push_range[1] + 1))
        delay_s = self.rng.uniform(*self.delay_ms) / 1000.0
        self._lag_buf = deque([self.last_action.copy()] * max(1, round(delay_s / self.dt)),
                              maxlen=max(1, round(delay_s / self.dt)))
        self._lag_alpha = 1.0 - np.exp(-self.dt / (self.rng.uniform(*self.tau_ms) / 1000.0))
        self._servo_state = np.zeros(N_ARM, dtype=np.float32)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        raw = np.clip(np.asarray(action, dtype=np.float32), self.reach_lo, self.reach_hi)
        d_raw = raw - self._prev_raw
        self._prev_raw = raw.copy()
        applied = ((1.0 - self.tau) * self.last_action + self.tau * raw).astype(np.float32)
        applied = np.clip(applied, self.reach_lo, self.reach_hi)
        self.last_action = applied

        # measured servo dynamics: dead time then first-order lag toward the command
        self._lag_buf.append(applied.copy())
        delayed = self._lag_buf[0]
        self._servo_state = (self._servo_state
                             + self._lag_alpha * (delayed - self._servo_state)).astype(np.float32)

        self.data.ctrl[self.non_arm_act] = 0.0
        self.data.ctrl[self.aadr] = self._servo_state
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self.data.qpos[0:7] = self._root_qpos       # rigid hold (seated/supported robot)
        self.data.qvel[0:6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.t += 1
        self.push_left -= 1
        if self.push_left <= 0:                     # kick the arms: teaches push-back
            self.data.qvel[self.vadr] += self.rng.normal(0, self.push_std, N_ARM)
            self.push_left = int(self.rng.integers(self.push_range[0], self.push_range[1] + 1))
        self.hold_left -= 1
        if self.hold_left <= 0:
            self._sample_target()

        jpos = self.data.qpos[self.qadr]
        err = jpos - self.target
        mse = float(np.mean(err ** 2))
        reward = self.w_track * float(np.exp(-self.track_bw * mse))
        reward += self.w_prec * float(np.exp(-self.prec_bw * mse))
        reward -= self.w_raw_rate * float(np.mean(d_raw ** 2))
        reward -= self.w_jvel * float(np.mean(self.data.qvel[self.vadr] ** 2))
        if np.all(np.abs(err) < self.hold_bonus_tol):
            reward += self.hold_bonus

        truncated = self.t >= self.max_steps
        return self._obs(), reward, False, truncated, {"track_err": float(np.abs(err).max())}


def make_arm_reach_env(config=None):
    return ArmReachEnv(config=config)
