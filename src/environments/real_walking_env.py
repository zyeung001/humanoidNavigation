# real_walking_env.py
"""Real-robot walking: PPO learns corrections on top of a phase-clocked reference gait.

Built on StandingEnv (the flex plant, the 228-dim proprioceptive obs, the real command chain:
clip -> residual box -> EMA -> dead time -> servo lag), so the body, sensing and action path are
the ones the deployed standing controller runs on. What it adds:

  * a REFERENCE gait, one cycle of joint targets q_ref(phase) and the sole heights it produced
    (`walk_reference`, an .npz from scripts/build_walk_reference.py). The residual box is centred
    on q_ref(phase) every step, so a network that outputs zero walks the reference (plus the
    feedback prior, if one is configured) and the policy's job is the feedback that keeps that
    gait upright on a plant it was not tuned for.
  * the phase as two obs dims, [sin, cos], appended after the history stack (the clock the robot
    would run).
  * a reward for following the reference (joint pose and sole heights) and for the forward speed
    the reference makes, plus upright, heading and smoothness terms.
  * a reward for real forward travel (both off by default, on in walk_step_to_v2):
      - `progress`: the REAR sole's new ground, ratcheted, so standing still, rocking, a
        split stance or a dive earn nothing;
      - `step`: a bonus at each touchdown of a foot that cleared `walk_step_clear_m` in the air
        and landed ahead of both its take-off and its furthest landing, so a slide or a hop in
        place earns nothing.
    and, with `walk_missed_step_terminate`, an episode ends (like a fall) when a step the
    reference lifts a foot for has not been landed `walk_step_grace_s` after its swing ends:
    without it, not stepping is a safe option a fall-averse policy takes (early termination, as
    in DeepMimic).
    WHY (11/14, walk_step_to-1113-2300): with only the speed and sole-height terms, a policy that
    slid its feet and never stepped kept ~97% of the reward (0.5 cm/s target: standing still paid
    95% of the speed term; the swing is 5% of the cycle), stayed up 24/24 held out, and walked
    0.11 cm/s instead of 0.5.

WHY A REFERENCE (11/13/2026, scratchpad walk/ feasibility). This body cannot walk statically: the
legs are hip roll/yaw/pitch + knee with no ankle, hip flexion stops at +0.35 rad and a bent knee
tips the foot toe-down. A trajectory search through the real command chain found that it CAN
step: lean onto one foot over ~1 s, lift the other for ~0.45 s (2-3 cm), land it 5 cm forward.
Plain PPO never found even standing feedback from scratch on this robot (scratch_base tied the
held pose), so the gait is given and RL learns to keep it on its feet.
"""
from typing import Optional

import numpy as np
from gymnasium.spaces import Box

from .standing_env import StandingEnv

PHASE_DIM = 3                  # sin(phase), cos(phase), gait running (0 while standing first)


def sole_boxes(model, foot_body_ids):
    """(geom id, its 8 box corners in the geom frame) for the collision box of each foot (R, L)."""
    corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], dtype=float)
    out = []
    for body in list(foot_body_ids)[:2]:
        g = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                 and (model.geom_contype[g] | model.geom_conaffinity[g]))
        out.append((g, corners * model.geom_size[g]))
    return out


def sole_heights(data, boxes) -> np.ndarray:
    """Lowest corner of each sole box above the floor (z = 0), m."""
    return np.array([float((data.geom_xpos[g] + c @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
                     for g, c in boxes], dtype=np.float32)


class GaitReference:
    """One periodic cycle of joint targets and sole heights, linearly interpolated in phase."""

    def __init__(self, t, q, foot_z, period_s, advance_m):
        self.t = np.asarray(t, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float32)
        self.foot_z = np.asarray(foot_z, dtype=np.float32)
        self.period = float(period_s)
        self.advance = float(advance_m)
        if self.q.ndim != 2 or len(self.t) != len(self.q) or len(self.t) != len(self.foot_z):
            raise ValueError("reference arrays t, q, foot_z must have the same length")
        if not (self.t[0] == 0.0 and np.all(np.diff(self.t) > 0) and self.t[-1] < self.period):
            raise ValueError("reference t must start at 0, increase, and end before period_s")

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["t"], z["q"], z["foot_z"], z["period_s"], z["advance_m"])

    @property
    def speed(self) -> float:
        """Forward speed the reference makes, m/s."""
        return self.advance / self.period

    def __call__(self, time_s: float):
        """(q_ref, foot_z_ref) at time_s, wrapping every period (the last sample joins the first)."""
        tt = float(time_s) % self.period
        t_ext = np.append(self.t, self.period)
        i = int(np.clip(np.searchsorted(t_ext, tt, side="right") - 1, 0, len(self.t) - 1))
        f = (tt - t_ext[i]) / (t_ext[i + 1] - t_ext[i])
        j = (i + 1) % len(self.t)
        return ((1 - f) * self.q[i] + f * self.q[j]).astype(np.float32), \
               ((1 - f) * self.foot_z[i] + f * self.foot_z[j]).astype(np.float32)


def swing_ends(t, z, clear_m) -> np.ndarray:
    """End times (s, within the cycle) of each stretch where a reference sole is above clear_m."""
    up = np.asarray(z) > clear_m
    ends = [t[i] for i in range(len(t)) if up[i] and not up[(i + 1) % len(t)]]
    return np.array([e + (t[1] - t[0]) for e in ends]) if len(t) > 1 else np.zeros(0)


def steps_due(gait_time_s, deadlines, period_s) -> int:
    """How many reference swings with these deadlines have passed by gait_time_s."""
    if gait_time_s < 0.0 or len(deadlines) == 0:
        return 0
    cycles, tt = divmod(gait_time_s, period_s)
    return int(cycles) * len(deadlines) + int(np.sum(np.asarray(deadlines) <= tt))


class StepTracker:
    """Real steps of each foot, from its sole height and position along the walking direction.

    A foot is in the air from when its sole rises above `lift_m` until it is back below `land_m`.
    A touchdown counts as a step when the sole cleared `clear_m` in between; its gain is how far
    it landed ahead of both where it took off and the furthest it had landed before (so stepping
    back and forth, or sliding forward and then hopping in place, gains nothing)."""

    def __init__(self, n_feet=2, lift_m=0.005, land_m=0.002, clear_m=0.015):
        self.n, self.lift_m, self.land_m, self.clear_m = n_feet, lift_m, land_m, clear_m
        self.reset(np.zeros(n_feet))

    def reset(self, x):
        self.best = np.asarray(x, dtype=np.float64).copy()
        self.air = np.zeros(self.n, dtype=bool)
        self.x_off = self.best.copy()
        self.peak = np.zeros(self.n)
        self.count = np.zeros(self.n, dtype=int)

    def update(self, z, x) -> np.ndarray:
        """Feed one control step; returns each foot's gain (m) if it just landed a step, else 0."""
        gain = np.zeros(self.n)
        for i in range(self.n):
            if not self.air[i]:
                if z[i] > self.lift_m:
                    self.air[i], self.x_off[i], self.peak[i] = True, x[i], z[i]
                continue
            self.peak[i] = max(self.peak[i], z[i])
            if z[i] < self.land_m:
                self.air[i] = False
                if self.peak[i] >= self.clear_m:
                    g = x[i] - max(self.x_off[i], self.best[i])
                    if g > 0.0:
                        gain[i] = g
                        self.count[i] += 1
                self.best[i] = max(self.best[i], x[i])
        return gain


class RealWalkingEnv(StandingEnv):
    def __init__(self, render_mode: Optional[str] = None, config=None):
        super().__init__(render_mode=render_mode, config=config)
        cfg = self.cfg
        if not np.any(self.residual_clamp > 0.0):
            raise ValueError("real_walking needs residual_clamp > 0: the policy acts around q_ref")
        self.ref = GaitReference.load(cfg["walk_reference"])
        if self.ref.q.shape[1] != self.env.action_space.shape[0]:
            raise ValueError(f"reference has {self.ref.q.shape[1]} joints, the robot "
                             f"{self.env.action_space.shape[0]}")
        # The servos trail the command by ~90 ms (dead time + lag) and the EMA adds ~70 ms, so
        # the joints are compared with the reference this much earlier.
        self.walk_track_lag_s = float(cfg.get("walk_track_lag_s", 0.15))
        self.w_pose = float(cfg.get("walk_pose_weight", 1.0))
        self.k_pose = float(cfg.get("walk_pose_k", 20.0))           # per rad^2, mean over joints
        self.w_foot = float(cfg.get("walk_foot_weight", 2.0))
        self.k_foot = float(cfg.get("walk_foot_k", 2000.0))         # per m^2, summed over feet
        self.w_speed = float(cfg.get("walk_speed_weight", 2.0))
        self.k_speed = float(cfg.get("walk_speed_k", 2000.0))       # per (m/s)^2
        self.speed_tau_s = float(cfg.get("walk_speed_tau_s", 1.0))  # EMA of forward COM velocity
        self.w_up = float(cfg.get("walk_upright_weight", 1.0))
        self.w_heading = float(cfg.get("walk_heading_weight", 0.5))
        self.w_alive = float(cfg.get("walk_alive_bonus", 0.5))
        self.w_rate = float(cfg.get("walk_action_rate_weight", 0.5))
        self.fall_penalty = float(cfg.get("walk_fall_penalty", 10.0))
        # Real forward travel (off by default; see the module docstring). progress pays
        # walk_progress_weight per reference cycle's advance of new rear-sole ground; step pays
        # walk_step_bonus per walk_step_ref_m a foot lands ahead (capped at 1.25 steps).
        self.w_progress = float(cfg.get("walk_progress_weight", 0.0))
        self.w_step = float(cfg.get("walk_step_bonus", 0.0))
        self.step_ref_m = float(cfg.get("walk_step_ref_m", 0.05))
        self.steps = StepTracker(clear_m=float(cfg.get("walk_step_clear_m", 0.015)))
        self._rear_best = 0.0
        self.rear_progress_m = 0.0
        self.missed_step_terminate = bool(cfg.get("walk_missed_step_terminate", False))
        grace = float(cfg.get("walk_step_grace_s", 0.5))
        # Deadlines in the cycle (s) by which each foot must have landed each reference swing.
        self.step_deadlines = [swing_ends(self.ref.t, self.ref.foot_z[:, i], self.steps.clear_m)
                               + grace for i in range(2)]
        # Each episode stands this long (s, drawn per episode) before the gait clock starts: the
        # reference was recorded from a settled stance, and the robot spawns a few mm up.
        ss = cfg.get("walk_stand_s", [1.0, 2.0])
        self.walk_stand_s = (float(ss), float(ss)) if np.isscalar(ss) else (float(ss[0]), float(ss[1]))
        self._stand_s = self.walk_stand_s[0]
        self._stand_z = np.zeros(2, dtype=np.float32)
        self._episode_offset = np.zeros_like(self.residual_baseline)
        self._speed_ema = 0.0
        self._yaw0 = 0.0
        self._foot_geoms = None

        base_dim = int(self.observation_space.shape[0])
        self.walking_obs_dim = base_dim + PHASE_DIM
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.walking_obs_dim,),
                                     dtype=np.float32)
        print(f"  RealWalkingEnv: reference {cfg['walk_reference']} (period {self.ref.period:.2f} s, "
              f"{100 * self.ref.advance:.1f} cm/cycle = {100 * self.ref.speed:.1f} cm/s); "
              f"obs {base_dim} + phase {PHASE_DIM} = {self.walking_obs_dim}")

    # ----- time and phase -----
    @property
    def time_s(self) -> float:
        return self.current_step * self._control_dt

    @property
    def gait_time_s(self) -> float:
        """Time on the gait clock: negative while the episode's opening stand lasts."""
        return self.time_s - self._stand_s

    def reference(self, gait_time_s: float):
        """(q_ref, sole heights, gait running) at a gait-clock time; the stance before it starts."""
        if gait_time_s < 0.0:
            return self.residual_baseline.astype(np.float32), self._stand_z, 0.0
        q, z = self.ref(gait_time_s)
        return q, z, 1.0

    def phase_features(self) -> np.ndarray:
        g = self.gait_time_s
        ph = 2.0 * np.pi * (max(g, 0.0) % self.ref.period) / self.ref.period
        return np.array([np.sin(ph), np.cos(ph), 1.0 if g >= 0.0 else 0.0], dtype=np.float32)

    # ----- geometry the reward reads -----
    def sole_heights(self) -> np.ndarray:
        """Lowest point of each sole box above the floor, m (R, L)."""
        if self._foot_geoms is None:
            self._foot_geoms = sole_boxes(self.env.unwrapped.model, self.model_spec.foot_body_ids)
        return sole_heights(self.env.unwrapped.data, self._foot_geoms)

    def sole_x(self) -> np.ndarray:
        """Forward (world x) position of each sole box's centre, m (R, L)."""
        self.sole_heights()                                  # builds the geom list once
        d = self.env.unwrapped.data
        return np.array([float(d.geom_xpos[g][0]) for g, _ in self._foot_geoms])

    @staticmethod
    def _yaw(quat) -> float:
        w, x, y, z = quat
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    # ----- obs -----
    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        base = super()._process_observation(obs)
        return np.concatenate([base, self.phase_features()]).astype(np.float32)

    # ----- step / reset -----
    def reset(self, seed: Optional[int] = None):
        # Back to the nominal centre first: StandingEnv.reset only redraws it when
        # residual_baseline_rand > 0, and otherwise it would keep the last episode's q_ref.
        self._episode_offset = np.zeros_like(self.residual_baseline)
        self._res_base_ep = self.residual_baseline.copy()
        out = super().reset(seed=seed)
        # residual_baseline_rand draws a per-episode offset of the box centre; keep it on top of
        # the moving reference instead of letting step() overwrite it.
        self._episode_offset = (self._res_base_ep - self.residual_baseline).astype(np.float32)
        self._speed_ema = 0.0
        self._yaw0 = self._yaw(self.env.unwrapped.data.qpos[3:7])
        self._stand_s = float(np.random.uniform(*self.walk_stand_s))
        x = self.sole_x()
        self.steps.reset(x)
        self._rear_best = float(x.min())
        self.rear_progress_m = 0.0
        # Recompute the first obs: its phase dims were built before this episode's stand length.
        obs, info = out
        return np.concatenate([obs[:-PHASE_DIM], self.phase_features()]).astype(np.float32), info

    def step(self, action):
        # As in standing, residual_baseline_rand moves the box, not the command: zero action
        # commands q_ref itself, and the box around it is shifted by this episode's offset.
        q_ref, _, _ = self.reference(self.gait_time_s)
        self._res_base_ep = (q_ref + self._episode_offset).astype(np.float32)
        return super().step(q_ref + np.asarray(action, dtype=np.float32))

    @property
    def episode_baseline(self) -> np.ndarray:
        """The pose a zero action commands now: the reference at this instant. (Scenarios start
        the robot here; until the gait clock starts that is the standing pose.)"""
        return self.reference(self.gait_time_s)[0].astype(np.float32)

    # ----- reward -----
    def _compute_task_reward(self, obs, base_reward, info, action):
        data = self.env.unwrapped.data
        height = self._get_height()
        upright_cos = -float(self._projected_gravity(data.qpos[3:7])[2])

        q_lag, z_ref, on = self.reference(self.gait_time_s - self.walk_track_lag_s)
        q = np.asarray(data.qpos[self.act_qposadr], dtype=np.float32)
        pose_err = float(np.mean(np.square(q - q_lag)))
        foot_err = float(np.sum(np.square(self.sole_heights() - z_ref)))   # the reference's soles

        a = 1.0 - np.exp(-self._control_dt / max(self.speed_tau_s, 1e-6))
        self._speed_ema += a * (float(data.qvel[0]) - self._speed_ema)
        speed_err = self._speed_ema - on * self.ref.speed
        # New ground under the rear foot, and steps landed (both paid only once the gait runs).
        z, x = self.sole_heights(), self.sole_x()
        step_gain = self.steps.update(z, x)
        rear_gain = max(float(x.min()) - self._rear_best, 0.0)
        self._rear_best += rear_gain
        if on:
            self.rear_progress_m += rear_gain
        else:
            step_gain[:] = 0.0
            self.steps.count[:] = 0
        heading_err = float(np.arctan2(np.sin(self._yaw(data.qpos[3:7]) - self._yaw0),
                                       np.cos(self._yaw(data.qpos[3:7]) - self._yaw0)))

        terms = {
            "pose": self.w_pose * np.exp(-self.k_pose * pose_err),
            "foot": self.w_foot * np.exp(-self.k_foot * foot_err),
            "speed": self.w_speed * np.exp(-self.k_speed * speed_err ** 2),
            "upright": self.w_up * np.exp(-8.0 * (1.0 - upright_cos) ** 2),
            "progress": on * self.w_progress * rear_gain / max(self.ref.advance, 1e-3),
            "step": self.w_step * float(np.minimum(step_gain / self.step_ref_m, 1.25).sum()),
            "heading": -self.w_heading * heading_err ** 2,
            "alive": self.w_alive,
            "action_rate": -self.w_rate * float(np.sum(np.square(self._last_action_rate))),
        }
        terminated = bool(height < 0.75 or height > 2.0 or upright_cos < 0.5)
        missed = self.missed_step_terminate and any(
            self.steps.count[i] < steps_due(self.gait_time_s, self.step_deadlines[i], self.ref.period)
            for i in range(2))
        if terminated or missed:
            terms["termination"] = -self.fall_penalty
            self._termination_cause = ("height_low" if height < 0.75 else
                                       "tilt" if upright_cos < 0.5 else
                                       "height_high" if terminated else "missed_step")
            terminated = True
        self._last_reward_terms = {k: float(v) for k, v in terms.items()}
        info.update({"walk_speed_ema": self._speed_ema, "walk_pose_err": pose_err,
                     "walk_foot_err": foot_err, "walk_heading_err": heading_err,
                     "upright_cos": upright_cos, "height": height,
                     "walk_steps_R": int(self.steps.count[0]), "walk_steps_L": int(self.steps.count[1]),
                     "walk_rear_progress_m": self.rear_progress_m})
        return float(sum(terms.values())), terminated


def make_real_walking_env(render_mode=None, config=None):
    return RealWalkingEnv(render_mode=render_mode, config=config)
