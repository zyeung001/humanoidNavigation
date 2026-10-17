"""ScenarioWrapper: play one Scenario per episode on a StandingEnv.

Training gives it a weighted mix and it draws a scenario at every reset. Evaluation names
the scenario through reset(options={"scenario": ...}). Either way the episode's info carries
`scenario`, so logs can report how each kind of episode is going.

With no mix and no scenario requested the wrapper does nothing at all: same RNG draws, same
observations as the bare env (tests/test_robustness.py holds it to that).
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import gymnasium as gym
import numpy as np

from .scenarios import Scenario, training_scenarios

# StandingEnv settings a scenario may override for one episode. Snapshotted at construction
# and restored before every reset, so one episode's overrides never leak into the next.
_SETTINGS = ("obs_noise", "push_enabled", "domain_rand", "rand_mass_range", "rand_friction_range",
             "actuator_rand", "actuator_rand_range", "actuator_lag_scale",
             "residual_baseline_rand", "actuator_profile_jitter")


def _u(rng, r) -> float:
    lo, hi = r
    return float(lo) if lo == hi else float(rng.uniform(lo, hi))


_GOLDEN = np.pi * (3.0 - np.sqrt(5.0))   # successive pushes in one episode spread evenly


def _horizontal(rng, stratum=None, phase: float = 0.0, k: int = 0) -> np.ndarray:
    """A random horizontal unit vector. With a stratum (i, n) -- set by the evaluator -- the
    angle is drawn from the i-th of n equal slices of the circle instead of the whole of it,
    so n episodes cover every direction once rather than clumping by chance. Direction
    decides most of whether a push or a lean is survivable, so this is most of the variance
    in a scenario's score. `phase` separates aspects (a lean and a gyro error should not
    always point the same way) and `k` spreads the k-th push of an episode by the golden
    angle."""
    if stratum is None:
        a = rng.uniform(0.0, 2.0 * np.pi)
    else:
        i, n = stratum
        a = 2.0 * np.pi * (i + rng.uniform()) / n + phase + k * _GOLDEN
    return np.array([np.cos(a), np.sin(a), 0.0])


def _rotation(axis, angle) -> np.ndarray:
    """Rodrigues: rotation by `angle` about unit `axis`."""
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


class ScenarioWrapper(gym.Wrapper):

    def __init__(self, env, mix: Optional[Dict[str, float]] = None,
                 scenarios: Optional[Dict[str, Scenario]] = None, mix_unit: str = "steps"):
        super().__init__(env)
        # WHAT A MIX WEIGHT MEANS. "steps" (default): the share of training STEPS a scenario
        # gets. "episodes": the share of episodes, which is what stand_pushes ran with on 10/8
        # -- and its pushed episodes fell after ~9 s against ~110 s for unpushed ones, so 40%
        # of episodes came to 6.8% of steps and 86% of its training was the control arm's.
        # For "steps" each scenario is drawn in proportion to weight / mean episode length,
        # with the lengths measured as training goes.
        if mix_unit not in ("steps", "episodes"):
            raise ValueError(f"mix_unit must be 'steps' or 'episodes', not {mix_unit!r}")
        self.mix_unit = mix_unit
        self._lengths: Dict[str, deque] = {}
        self.base = env
        while not hasattr(self.base, "place_robot"):     # find the StandingEnv inside
            self.base = self.base.env
        self._scenarios = dict(scenarios or training_scenarios())
        self._weights: Dict[str, float] = {}
        if mix:
            self.set_scenario_mix(mix)
        self._seed_streams(None)
        self._defaults = {k: getattr(self.base, k) for k in _SETTINGS}
        m = self.base.env.unwrapped.model
        # The env randomises these arrays in place and never puts them back when randomisation
        # is switched off, so the wrapper owns restoring them.
        self._nominal = {k: getattr(m, k).copy() for k in
                         ("body_mass", "body_inertia", "geom_friction", "actuator_gainprm",
                          "actuator_biasprm")}
        self._dt = float(self.base.env.unwrapped.dt)
        self._body_cache: Dict[str, int] = {}
        self._dirty = False
        self._stratum = None
        self.scenario: Optional[Scenario] = None
        self.disturbance_steps = []   # control steps at which a disturbance began this episode
        self._reset_schedule()

    # ------------------------------------------------------------------ mix
    def set_scenario_mix(self, weights: Dict[str, float]) -> None:
        """Replace the training mix. Called through VecEnv.env_method by adaptive training."""
        unknown = set(weights) - set(self._scenarios)
        if unknown:
            raise KeyError(f"unknown training scenario(s) {sorted(unknown)}; "
                           f"known: {sorted(self._scenarios)}")
        total = float(sum(max(0.0, float(w)) for w in weights.values()))
        if total <= 0.0:
            raise ValueError("scenario mix weights must not all be zero")
        self._weights = {k: max(0.0, float(w)) / total for k, w in weights.items()}

    def get_scenario_mix(self) -> Dict[str, float]:
        return dict(self._weights)

    def episode_probabilities(self) -> Dict[str, float]:
        """The chance each scenario is drawn at the next reset. A scenario not yet seen is
        assumed to run the full episode, so the first draws follow the weights."""
        if self.mix_unit == "episodes":
            return dict(self._weights)
        full = float(getattr(self.base, "max_episode_steps", 1000))
        raw = {n: w / max(1.0, float(np.mean(self._lengths[n])) if self._lengths.get(n) else full)
               for n, w in self._weights.items()}
        total = sum(raw.values()) or 1.0
        return {n: v / total for n, v in raw.items()}

    # ------------------------------------------------------------------ gym API
    def _seed_streams(self, seed):
        # One independent stream per aspect, so what one scenario draws for its sensors cannot
        # shift the pushes another one gets. With the same seed, episode i of push_5N and of
        # push_5N_gyro meet identical pushes at identical times; only the gyro differs.
        ss = np.random.SeedSequence(seed)
        (self._rng_mix, self._rng_plant, self._rng_sensors, self._rng_start,
         self._rng_disturb) = [np.random.default_rng(s) for s in ss.spawn(5)]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed_streams(seed)
        requested = (options or {}).get("scenario")
        if isinstance(requested, str):
            requested = self._scenarios[requested]
        if requested is None and self._weights:
            probs = self.episode_probabilities()
            names = list(probs)
            p = np.array([probs[n] for n in names])
            requested = self._scenarios[names[int(self._rng_mix.choice(len(names), p=p))]]
        self.scenario = requested
        self._stratum = (options or {}).get("stratum")
        self._reset_schedule()

        if requested is None:
            if self._dirty:
                self._restore()
                self.base.set_sensor_offsets()
                self._dirty = False
            return self.env.reset(seed=seed)

        self._dirty = True
        self._configure(requested)
        obs, info = self.env.reset(seed=seed)
        self.base.env.unwrapped.data.xfrc_applied[:] = 0.0
        if not requested.start.is_noop():
            obs = self._place(requested)
        d = requested.disturb
        if d.kind != "none":
            self._next_at = max(1, round(_u(self._rng_disturb, d.first_s) / self._dt))
        info["scenario"] = requested.name
        return obs, info

    def step(self, action):
        if self.scenario is not None and self.scenario.disturb.kind != "none":
            self._disturb()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        if self.scenario is not None and (terminated or truncated):
            info["scenario"] = self.scenario.name
            self._lengths.setdefault(self.scenario.name, deque(maxlen=50)).append(self._t)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ internals
    def _restore(self) -> None:
        m = self.base.env.unwrapped.model
        for k, v in self._nominal.items():
            getattr(m, k)[:] = v
        for k, v in self._defaults.items():
            setattr(self.base, k, v)

    def _configure(self, s: Scenario) -> None:
        """Settings for this episode: defaults, then whatever the scenario names."""
        b = self.base
        self._restore()

        if s.env_pushes is not None:
            b.push_enabled = bool(s.env_pushes)
        if s.env_randomisation is False:
            b.domain_rand = b.actuator_rand = False
            b.residual_baseline_rand = 0.0
            b.actuator_profile_jitter = 0.0
        if s.sensors.noise is not None:
            b.obs_noise = bool(s.sensors.noise)

        p = s.plant
        if p.mass_scale is not None or p.friction_scale is not None:
            keep = s.env_randomisation is not False   # unspecified axis: configured, or nominal
            b.domain_rand = True
            b.rand_mass_range = list(p.mass_scale or (self._defaults["rand_mass_range"]
                                                      if keep else (1.0, 1.0)))
            b.rand_friction_range = list(p.friction_scale or (self._defaults["rand_friction_range"]
                                                              if keep else (1.0, 1.0)))
        if p.servo_gain is not None:
            b.actuator_rand = True
            b.actuator_rand_range = list(p.servo_gain)
        b.actuator_lag_scale = self._defaults["actuator_lag_scale"] * _u(self._rng_plant, p.servo_lag)

        se, rng = s.sensors, self._rng_sensors
        gyro = imu = enc = None
        g = _u(rng, se.gyro_bias)
        if g > 0:                                # base x/y: the roll and pitch rates
            gyro = g * _horizontal(rng, self._stratum, phase=3.0)
        t = np.radians(_u(rng, se.imu_tilt_deg))
        if t > 0:
            imu = _rotation(_horizontal(rng, self._stratum, phase=4.0), t)
        e = _u(rng, se.encoder_bias)
        if e > 0:
            enc = rng.uniform(-e, e, size=b.n_joints)
        b.set_sensor_offsets(gyro=gyro, imu_rotation=imu, jpos=enc)

    def _place(self, s: Scenario):
        b, rng, st = self.base, self._rng_start, s.start
        data = b.env.unwrapped.data
        q = b.episode_baseline if st.at_baseline else np.asarray(
            data.qpos[b.act_qposadr], dtype=np.float32).copy()
        off = _u(rng, st.joint_offset)
        if off > 0:
            q = q + rng.uniform(-off, off, size=q.shape)
        lo, hi = b.env.action_space.low, b.env.action_space.high   # the joints' real ranges
        q = np.clip(q, lo, hi)
        tilt = np.radians(_u(rng, st.tilt_deg))
        w = _u(rng, st.ang_vel)
        v = _u(rng, st.lin_vel)
        sv = self._stratum
        return b.place_robot(
            joint_pos=q if (st.at_baseline or off > 0) else None,
            tilt_axis=_horizontal(rng, sv) if tilt > 0 else None, tilt_rad=tilt,
            base_ang_vel=w * _horizontal(rng, sv, phase=1.0) if w > 0 else None,
            base_lin_vel=v * _horizontal(rng, sv, phase=1.5) if v > 0 else None,
            smoothing_from_zero=st.ema_from_zero)

    def _reset_schedule(self):
        self._t = 0
        self._next_at = None
        self._push_until = None
        self._kick_sign = 1.0
        self.disturbance_steps = []

    def _disturb(self) -> None:
        d, rng = self.scenario.disturb, self._rng_disturb
        data = self.base.env.unwrapped.data
        if self._push_until is not None and self._t >= self._push_until:
            data.xfrc_applied[self._body(d.body), :] = 0.0
            self._push_until = None
        if self._next_at is None or self._t < self._next_at:
            return
        direction = _horizontal(rng, self._stratum, phase=2.0, k=len(self.disturbance_steps))
        mag = _u(rng, d.magnitude)
        if d.kind == "push":
            data.xfrc_applied[self._body(d.body), 0:3] = mag * direction
            self._push_until = self._t + max(1, round(_u(rng, d.duration_s) / self._dt))
        else:
            data.qvel[0:3] += mag * direction
            data.qvel[5] += self._kick_sign * _u(rng, d.yaw_rate)
            self._kick_sign = -self._kick_sign
        self.disturbance_steps.append(self._t)
        self._next_at = self._t + max(1, round(_u(rng, d.interval_s) / self._dt))

    def _body(self, name: str) -> int:
        """Body id for a push. 'torso' = the highest subtree holding >= 15% of the mass (the
        upper body, where a hand lands); 'pelvis' = the floating base."""
        if name in self._body_cache:
            return self._body_cache[name]
        import mujoco
        m, data = self.base.env.unwrapped.model, self.base.env.unwrapped.data
        if name == "pelvis":
            bid = 1
        elif name == "torso":
            heavy = [i for i in range(1, m.nbody)
                     if m.body_subtreemass[i] >= 0.15 * m.body_subtreemass[1]]
            bid = max(heavy, key=lambda i: data.subtree_com[i][2])
        else:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise KeyError(f"no body named {name!r} to push")
        self._body_cache[name] = int(bid)
        return int(bid)
