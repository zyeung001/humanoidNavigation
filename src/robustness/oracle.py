"""Oracle controllers: what the PLANT allows, whatever a policy can sense or learn.

Every standing controller tried on the flex plant (PPO v1-v3, LCP, an 18-point knee-law gain
grid) scored within ~1 point of the knee law on standing-v2. These tools ask whether that is the
controllers or the robot. Each one reads sim truth (full state, servo filter, dead-time buffer)
and so is an upper bound, not something to deploy:

- `LQRPolicy`: the best LINEAR full-state feedback. The whole env step (clip -> residual clamp ->
  EMA -> per-joint dead time -> first-order servo -> MuJoCo, 5 x 5 ms) is linearised by central
  differences at the settled stand (`linearize`) and a discounted DARE gives the gain (`lqr_gain`).
- `MPCPolicy`: sampling MPC (MPPI) whose model is the episode's own sim -- randomised masses,
  friction and servo speed included -- rolled out with `mujoco.rollout`. The env's command chain
  is replicated in numpy (`command_chain`), so a rollout is bit-for-bit an env step (tested).
  Future disturbances are NOT known: rollouts run with no external force.
- `AnkleServoPolicy`: not an oracle but a hardware what-if. The robot has no ankle servos; the
  foot hangs on the passive `ankle_flex` play spring. This drives that spring's rest angle
  through a lagged SCS15-like servo with a 2-gain ankle law, on top of the knee law.

`run_bound()` scores any of them on suite scenarios with the suite's own paired seeds, so the
results pair with `suite.evaluate` of the law. `scripts/ceiling.py` is the CLI.
"""

from __future__ import annotations

import contextlib
import io
import time
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------------- environment


def make_env(cfg: dict, seconds: float = 15.0):
    """A suite eval env (StandingEnv in a ScenarioWrapper), with its config print silenced."""
    from src.robustness import suite  # noqa: PLC0415
    with contextlib.redirect_stdout(io.StringIO()):
        return suite.make_eval_env(cfg, seconds)


def settle(wenv, seconds: float = 6.0) -> np.ndarray:
    """Reset into `quiet` (seed 0) and hold the baseline pose `seconds`; returns that pose."""
    from src.robustness.scenarios import SUITE  # noqa: PLC0415
    quiet = next(s for s in SUITE if s.name == "quiet")
    np.random.seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        wenv.reset(seed=0, options={"scenario": quiet, "stratum": None})
    u0 = wenv.base.episode_baseline.astype(np.float32).copy()
    for _ in range(int(seconds / wenv.base._control_dt)):
        _, _, term, _, _ = wenv.step(u0)
        if term:
            raise RuntimeError("fell while settling on the held baseline")
    return u0


def push_band(forces=(1.5, 2.0, 2.5, 3.0)):
    """One 0.2 s torso push per episode at each force: the band where standing in place CAN
    recover (capture-point shift 3-6 cm inside an 8 cm half-foot) but the law sometimes falls."""
    from src.robustness.scenarios import _eval, _push  # noqa: PLC0415
    return [_eval(f"p{f:g}", "push", f"one {f:g} N torso push", disturb=_push(f, every=(30.0, 30.0)))
            for f in forces]


# ------------------------------------------------------------------------------- command chain


def command_chain(base, U: np.ndarray) -> np.ndarray:
    """The actuator commands (N, H, nu) the env would send for raw actions U (N, H, nu), starting
    from its CURRENT filter state, without touching it. Mirrors StandingEnv._process_action +
    _apply_actuator_lag (no prior: add it to U first if the env has one)."""
    N, H, nu = U.shape
    low = base.env.action_space.low.astype(np.float64)
    high = base.env.action_space.high.astype(np.float64)
    clamp = np.asarray(base.residual_clamp, dtype=np.float64)
    centre = np.asarray(base._res_base_ep, dtype=np.float64)
    tau = float(np.clip(base.action_smoothing_tau, 0.0, 1.0)) if base.enable_action_smoothing else 1.0
    ema = np.tile(np.asarray(base.prev_action, np.float64), (N, 1))
    lagged = bool(base.actuator_lag)
    if lagged:
        buf = np.tile(np.asarray(list(base._lag_buffer), np.float64), (N, 1, 1))
        servo = np.tile(np.asarray(base._servo_state, np.float64), (N, 1))
        depth = buf.shape[1]
        idx = np.clip(depth - 1 - np.asarray(base._lag_delay_steps), 0, depth - 1)
        alpha = np.asarray(base._lag_alpha, dtype=np.float64)
        cols = np.arange(nu)
    out = np.empty((N, H, nu))
    for t in range(H):
        a = np.clip(U[:, t], low, high)
        if np.any(clamp > 0):
            a = np.clip(a, centre - clamp, centre + clamp)
        ema = np.clip((1.0 - tau) * ema + tau * a, low, high)
        if not lagged:
            out[:, t] = ema
            continue
        buf = np.concatenate([buf[:, 1:], ema[:, None]], axis=1)
        servo = servo + alpha * (buf[:, idx, cols] - servo)
        out[:, t] = np.clip(servo, low, high)
    return out


def rollout_states(base, U: np.ndarray, scratch) -> np.ndarray:
    """Sim states (N, H, nstate: time, qpos, qvel) at the end of each control step for raw actions
    U from the env's current state. `scratch` is an MjData of the env's model (its xfrc_applied
    stays zero: the planner does not know about pushes)."""
    import mujoco  # noqa: PLC0415
    from mujoco import rollout  # noqa: PLC0415
    m, d = base.env.unwrapped.model, base.env.unwrapped.data
    fs = base.env.unwrapped.frame_skip
    ctrl = np.repeat(command_chain(base, U), fs, axis=1)
    x0 = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(m, d, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    st, _ = rollout.rollout(m, scratch, x0[None], ctrl, initial_warmstart=d.qacc_warmstart[None].copy())
    return st[:, fs - 1::fs, :]


# -------------------------------------------------------------------------------------- MPC

MPC_DEFAULTS = dict(N=32, H=16, knots=4, sigma=0.05, lam=0.1,
                    w_tilt=100.0, w_cp=1000.0, w_ang=1.0, w_h=100.0, w_u=1.0, w_du=10.0, w_q=0.1,
                    w_term=5.0, fall_up=0.80, w_fall=100.0)


def stand_reference(base) -> dict:
    """Height, joint pose and capture-point offset from the feet of the stand the env is in now."""
    probe = MPCPolicy({})
    probe.bind(base, None)
    com, feet = probe._com_feet(base.env.unwrapped.data.qpos)
    d = base.env.unwrapped.data
    return dict(h=float(d.qpos[2]), q=d.qpos[7:].copy(), cp_off=(com - feet)[:2].copy())


class MPCPolicy:
    """MPPI over raw action sequences: N candidates over H control steps (knot noise around the
    shifted previous plan, plus the plain hold), scored on lean, capture point vs the feet, base
    rate, height, pose, effort and command rate. Use with an env WITHOUT a feedback prior (the
    plan is the whole command). Cost per control step ~ N * H * 5 mj_steps (~0.1 ms each here)."""

    def __init__(self, params: Optional[dict] = None, label: str = "mpc"):
        self.p = dict(MPC_DEFAULTS, **(params or {}))
        self.label = self.fingerprint = label

    def bind(self, base, ref: Optional[dict]):
        import mujoco  # noqa: PLC0415
        self.b = base
        self.m, self.d = base.env.unwrapped.model, base.env.unwrapped.data
        self.scratch, self.kd = mujoco.MjData(self.m), mujoco.MjData(self.m)
        self.feet = list(base.model_spec.foot_body_ids)[:2]
        self.ref = ref
        self.nu = base.env.action_space.shape[0]
        self.nq, self.nv = self.m.nq, self.m.nv
        self.rng = np.random.default_rng(0)

    def reset(self):
        self.u_base = np.asarray(self.b.episode_baseline, dtype=np.float64).copy()
        self.Ubar = np.tile(self.u_base, (self.p["H"], 1))
        self.rng = np.random.default_rng(int(np.abs(self.d.qpos).sum() * 1e6) % 2**32)

    def _com_feet(self, qpos):
        import mujoco  # noqa: PLC0415
        self.kd.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.kd)
        mujoco.mj_comPos(self.m, self.kd)
        return self.kd.subtree_com[1].copy(), 0.5 * (self.kd.xpos[self.feet[0]] + self.kd.xpos[self.feet[1]])

    def cost(self, S, U):
        p, nq, nv = self.p, self.nq, self.nv
        N, H, _ = S.shape
        qpos, qvel = S[..., 1:1 + nq], S[..., 1 + nq:1 + nq + nv]
        up = 1.0 - 2.0 * (qpos[..., 4] ** 2 + qpos[..., 5] ** 2)
        com = np.empty((N, H + 1, 3))
        feet = np.empty((N, H, 3))
        com[:, 0] = self._com_feet(self.d.qpos)[0]
        for i in range(N):
            for t in range(H):
                com[i, t + 1], feet[i, t] = self._com_feet(qpos[i, t])
        vcom = np.diff(com, axis=1) / self.b._control_dt
        z = np.maximum(com[:, 1:, 2] - feet[..., 2], 0.05)
        cp = com[:, 1:, :2] + vcom[..., :2] * np.sqrt(z / 9.81)[..., None]
        cpe = cp - feet[..., :2] - self.ref["cp_off"]
        prev = np.tile(np.asarray(self.b.prev_action, np.float64), (N, 1, 1))
        c = (p["w_tilt"] * 2.0 * (1.0 - up) + p["w_cp"] * np.sum(cpe ** 2, -1)
             + p["w_ang"] * np.sum(qvel[..., 3:6] ** 2, -1)
             + p["w_h"] * (qpos[..., 2] - self.ref["h"]) ** 2
             + p["w_q"] * np.sum((qpos[..., 7:] - self.ref["q"]) ** 2, -1)
             + p["w_u"] * np.sum((U - self.u_base) ** 2, -1)
             + p["w_du"] * np.sum(np.diff(np.concatenate([prev, U], 1), axis=1) ** 2, -1)
             + p["w_fall"] * (up < p["fall_up"]))
        c[:, -1] *= p["w_term"]
        return c.sum(1)

    def act(self, obs):
        p = self.p
        N, H, K = p["N"], p["H"], p["knots"]
        knots = self.rng.normal(0.0, p["sigma"], (N, K, self.nu))
        knots[0] = 0.0                                             # the previous plan, shifted
        kt, t = np.linspace(0, H - 1, K), np.arange(H)
        w = np.clip(np.searchsorted(kt, t, side="right") - 1, 0, K - 2)
        f = ((t - kt[w]) / (kt[w + 1] - kt[w]))[None, :, None]
        U = self.Ubar[None] + (1 - f) * knots[:, w] + f * knots[:, w + 1]
        U[1] = self.u_base                                         # the plain hold, always tried
        c = self.cost(rollout_states(self.b, U, self.scratch), U)
        wts = np.exp(-(c - c.min()) / (p["lam"] * max(np.median(c) - c.min(), 1e-9)))
        plan = np.einsum("n,nhj->hj", wts / wts.sum(), U)
        self.Ubar = np.concatenate([plan[1:], plan[-1:]], 0)
        return plan[0].astype(np.float32)


# -------------------------------------------------------------------------------------- LQR


class EnvStateIO:
    """Get / set / step the FULL state of a StandingEnv: MuJoCo integration state + the command
    chain (EMA, servo filter, dead-time buffer). x = [dq (nv, tangent), qvel, ema, servo, buffer]."""

    def __init__(self, base):
        self.b = base
        self.m, self.d = base.env.unwrapped.model, base.env.unwrapped.data
        self.nv, self.nu = self.m.nv, base.env.action_space.shape[0]

    @property
    def depth(self):                       # the dead-time buffer exists only after a reset
        return len(self.b._lag_buffer)

    @property
    def nx(self):
        return 2 * self.nv + 2 * self.nu + self.depth * self.nu

    def snapshot(self):
        import mujoco  # noqa: PLC0415
        s = np.empty(mujoco.mj_stateSize(self.m, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.m, self.d, s, mujoco.mjtState.mjSTATE_INTEGRATION)
        return dict(mj=s, qpos=self.d.qpos.copy(), ema=self.b.prev_action.copy(),
                    servo=self.b._servo_state.copy(), buf=[x.copy() for x in self.b._lag_buffer],
                    raw=self.b._prev_raw_action.copy())

    def restore(self, snap):
        import mujoco  # noqa: PLC0415
        mujoco.mj_setState(self.m, self.d, snap["mj"], mujoco.mjtState.mjSTATE_INTEGRATION)
        self.b.prev_action = snap["ema"].copy()
        self.b._servo_state = snap["servo"].copy()
        self.b._lag_buffer = deque([x.copy() for x in snap["buf"]], maxlen=len(snap["buf"]))
        self.b._prev_raw_action = snap["raw"].copy()
        mujoco.mj_forward(self.m, self.d)

    def x(self, qpos_ref):
        import mujoco  # noqa: PLC0415
        dq = np.zeros(self.nv)
        mujoco.mj_differentiatePos(self.m, dq, 1.0, qpos_ref, self.d.qpos)
        return np.concatenate([dq, self.d.qvel, self.b.prev_action, self.b._servo_state,
                               np.concatenate(list(self.b._lag_buffer))])

    def set_x(self, x, snap):
        import mujoco  # noqa: PLC0415
        self.restore(snap)
        nv, nu = self.nv, self.nu
        q = snap["qpos"].copy()
        mujoco.mj_integratePos(self.m, q, x[:nv], 1.0)
        self.d.qpos[:] = q
        self.d.qvel[:] = x[nv:2 * nv]
        o = 2 * nv
        self.b.prev_action = x[o:o + nu].astype(np.float32)
        self.b._servo_state = x[o + nu:o + 2 * nu].astype(np.float32)
        o += 2 * nu
        self.b._lag_buffer = deque([x[o + k * nu:o + (k + 1) * nu].astype(np.float32)
                                    for k in range(self.depth)], maxlen=self.depth)
        mujoco.mj_forward(self.m, self.d)

    def step(self, u):
        proc = self.b._process_action(np.asarray(u, dtype=np.float32))
        act = self.b._apply_actuator_lag(proc)
        self.b.env.unwrapped.do_simulation(act, self.b.env.unwrapped.frame_skip)


def linearize(io_: EnvStateIO, u0, eps: float = 1e-6):
    """(A, B, x0, snapshot) of one env step at the current state, by central differences."""
    snap = io_.snapshot()
    qref = snap["qpos"].copy()
    x0 = io_.x(qref)

    def f(x, u):
        io_.set_x(x, snap)
        io_.step(u)
        return io_.x(qref)

    A = np.zeros((io_.nx, io_.nx))
    B = np.zeros((io_.nx, io_.nu))
    for i in range(io_.nx):
        e = np.zeros(io_.nx)
        e[i] = eps
        A[:, i] = (f(x0 + e, u0) - f(x0 - e, u0)) / (2 * eps)
    for j in range(io_.nu):
        e = np.zeros(io_.nu)
        e[j] = eps
        B[:, j] = (f(x0, u0 + e) - f(x0, u0 - e)) / (2 * eps)
    io_.restore(snap)
    return A, B, x0, snap


def lqr_gain(A, B, nv: int, w: Dict[str, float], gamma: float = 0.99):
    """Discounted DARE gain. Base x, y, yaw carry no balance information with planted feet: zero
    weight and zero K columns (discounting keeps those marginal modes from breaking the DARE)."""
    import scipy.linalg  # noqa: PLC0415
    nx, nu = A.shape[0], B.shape[1]
    q = np.full(nx, w["internal"])
    q[0:3] = [0.0, 0.0, w["pos_z"]]
    q[3:6] = [w["tilt"], w["tilt"], 0.0]
    q[6:nv] = w["joint"]
    q[nv:nv + 3] = w["linvel"]
    q[nv + 3:nv + 6] = w["angvel"]
    q[nv + 6:2 * nv] = w["jointvel"]
    Q, R = np.diag(q), w["r"] * np.eye(nu)
    g = np.sqrt(gamma)
    P = scipy.linalg.solve_discrete_are(g * A, g * B, Q, R)
    K = np.linalg.solve(R + gamma * B.T @ P @ B, gamma * B.T @ P @ A)
    K[:, [0, 1, 5]] = 0.0
    return K, float(np.abs(np.linalg.eigvals(A - B @ K)).max())


class LQRPolicy:
    """u = u0 - K (x - x0) on sim truth. Designed for the nominal dead-time buffer depth: a
    slow_plant scenario lengthens the buffer and, like the law, the controller is not told (it
    reads the most recent nominal-depth commands, padding with the oldest if shorter)."""

    def __init__(self, K, x0, u0, qref, label: str = "lqr"):
        self.K, self.x0, self.u0, self.qref = K, x0, u0, qref
        self.label = self.fingerprint = label

    def bind(self, base, ref=None):
        self.io = EnvStateIO(base)

    def act(self, obs):
        nv, nu = self.io.nv, self.io.nu
        depth0 = (len(self.x0) - 2 * nv - 2 * nu) // nu
        x = self.io.x(self.qref)
        head, buf = x[:2 * nv + 2 * nu], x[2 * nv + 2 * nu:].reshape(-1, nu)
        buf = buf[-depth0:] if len(buf) >= depth0 else np.vstack([np.repeat(buf[:1], depth0 - len(buf), 0), buf])
        return self.u0 - self.K @ (np.concatenate([head, buf.ravel()]) - self.x0)


# ------------------------------------------------------------------------- ankle servo what-if

# ~50 ms dead time (the 8/14 bench sweep: 41-67 ms) and ~40 ms first-order lag, at 40 Hz. The gain
# is latency-sensitive: 3 steps (75 ms) gave +1.1 +- 0.3 where 2 steps give +1.7 +- 0.3 (778x48).
ANKLE_DELAY_STEPS, ANKLE_ALPHA = 2, 0.38


class AnkleServoPolicy:
    """Hold `pose` (the knee law, in the env, stays underneath) AND drive the rest angle of the
    passive ankle_flex joints like an added SCS15 ankle-pitch servo would:
        theta = lag(delay(clip(ka * pg_x + kd * gyro_y, +-0.5)))
    from the newest noisy IMU signals, exactly as the law reads them. The play spring stays in
    series: k = 7 N m/rad is the fitted play alone (optimistic); 4.4 = that play in series with
    a kp-12 servo (conservative). ka = kd = 0 holds the ankle at 0 (the servo, unused)."""

    def __init__(self, pose, k: float = 7.0, ka: float = 0.0, kd: float = 0.0, label: str = "",
                 delay_steps: int = ANKLE_DELAY_STEPS):
        self.pose = np.asarray(pose, np.float32).copy()
        self.k, self.ka, self.kd = float(k), float(ka), float(kd)
        self.delay_steps = int(delay_steps)
        self.label = self.fingerprint = label or f"ankle k{k:g} ka{ka:g} kd{kd:g}"

    def bind(self, base, ref=None):
        self.b = base
        self.m = base.env.unwrapped.model
        self.j = [self.m.joint(n).id for n in ("ankle_flex_R", "ankle_flex_L")]
        self.adr = [int(self.m.jnt_qposadr[j]) for j in self.j]
        self.sign = (1.0, -1.0)                 # mirrored joint axes: the same physical pitch
        self.k_nom = None

    def reset(self):
        # The env redraws flex stiffness at every reset when flex_rand_range is set (before this
        # runs), so scaling what it set is one scale per episode; otherwise scale the nominal.
        if self.b._flex_dofs is not None:
            self.m.jnt_stiffness[self.j] *= self.k / 7.0
        else:
            if self.k_nom is None:
                self.k_nom = self.m.jnt_stiffness[self.j].copy()
            self.m.jnt_stiffness[self.j] = self.k_nom * (self.k / 7.0)
        self.line = deque([0.0] * (self.delay_steps + 1), maxlen=self.delay_steps + 1)
        self.theta = 0.0
        for a in self.adr:
            self.m.qpos_spring[a] = 0.0

    def act(self, obs):
        s = self.b._prior_signals
        self.line.append(float(np.clip(self.ka * s[0] + self.kd * s[4], -0.5, 0.5)))
        self.theta += ANKLE_ALPHA * (self.line[0] - self.theta)     # the command delay_steps ago
        for a, sg in zip(self.adr, self.sign):
            self.m.qpos_spring[a] = sg * self.theta
        return self.pose.copy()


# ------------------------------------------------------------------------------------ scoring

_W: dict = {}


def _init(cfg, seconds, factory):
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415
    sys.stdout = open(os.devnull, "w")
    import torch  # noqa: PLC0415
    torch.set_num_threads(1)
    wenv = make_env(cfg, seconds)
    _W.update(wenv=wenv, pol=factory(wenv), steps=int(round(seconds / wenv.base._control_dt)))


def _run(task):
    from src.robustness import suite  # noqa: PLC0415
    s, sd, st = task
    return suite.run_episode(_W["pol"], _W["wenv"], s, sd, _W["steps"], st)


def run_bound(factory: Callable, cfg: dict, scenarios: List, seed: int, episodes: int,
              seconds: float = 15.0, workers: int = 11, label: str = "oracle", config_path: str = ""):
    """Score a policy that needs the env itself (`factory(wenv) -> policy`, picklable, e.g. a
    functools.partial of a module-level function) on `scenarios`, with the suite's paired seeds."""
    import multiprocessing as mp  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from src.robustness import suite  # noqa: PLC0415
    from src.robustness.scenarios import SUITE  # noqa: PLC0415
    t0 = time.time()
    tasks = [(s, suite.episode_seed(seed, i), (i, episodes)) for s in scenarios for i in range(episodes)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(cfg, seconds, factory)) as pool:
        flat = list(pool.map(_run, tasks, chunksize=1))
    res = [suite.ScenarioResult(s.name, s.family, s.about, flat[k * episodes:(k + 1) * episodes])
           for k, s in enumerate(scenarios)]
    return suite.SuiteResult(label=label, fingerprint=label, suite_version=suite.SUITE_VERSION, seed=seed,
                             seconds=seconds, episodes_per_scenario=episodes,
                             complete={s.name for s in scenarios} >= {s.name for s in SUITE},
                             config=config_path, git=suite.git_sha(), when=time.strftime("%Y-%m-%d %H:%M"),
                             step=None, wall_s=time.time() - t0, scenarios=res)


def families(r) -> Dict[str, float]:
    fam: Dict[str, list] = {}
    for s in r.scenarios:
        fam.setdefault(s.family, []).append(s.standing_pct)
    return {k: round(float(np.mean(v)), 1) for k, v in fam.items()}


# Factories (module-level, so they pickle into the worker processes).

def mpc_factory(params, label, wenv):
    settle(wenv)
    ref = stand_reference(wenv.base)
    pol = MPCPolicy(params, label)
    pol.bind(wenv.base, ref)
    return pol


def lqr_factory(K, x0, u0, qref, label, wenv):
    pol = LQRPolicy(K, x0, u0, qref, label)
    pol.bind(wenv.base)
    return pol


def ankle_factory(pose, k, ka, kd, wenv, delay_steps=ANKLE_DELAY_STEPS):
    pol = AnkleServoPolicy(pose, k, ka, kd, delay_steps=delay_steps)
    pol.bind(wenv.base)
    return pol
