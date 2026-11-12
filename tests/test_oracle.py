"""The oracles' models of the env must BE the env, or their ceilings say nothing."""
import copy

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.robustness import oracle, policies, suite  # noqa: E402
from src.robustness.scenarios import suite_scenarios  # noqa: E402
from src.training.experiment import load_experiment  # noqa: E402

CFG = "config/experiments/flex_plant_v2.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_experiment(CFG).task_cfg("standing")


@pytest.fixture(scope="module")
def settled(cfg):
    c = copy.deepcopy(cfg)
    c["feedback_prior"] = None
    wenv = oracle.make_env(c, 6.0)
    u0 = oracle.settle(wenv, seconds=1.0)
    return wenv, u0


def test_command_chain_and_rollout_match_the_env(settled):
    wenv, u0 = settled
    b = wenv.base
    rng = np.random.default_rng(1)
    U = u0 + np.cumsum(rng.normal(0, 0.01, (1, 20, len(u0))), axis=1)
    import mujoco
    S = oracle.rollout_states(b, U, mujoco.MjData(b.env.unwrapped.model))
    ctrl = oracle.command_chain(b, U)[0]
    for t in range(20):
        proc = b._process_action(U[0, t].astype(np.float32))
        act = b._apply_actuator_lag(proc)
        assert np.abs(act - ctrl[t]).max() < 1e-6
        b.env.unwrapped.do_simulation(act, b.env.unwrapped.frame_skip)
        q = b.env.unwrapped.data.qpos
        assert np.abs(q - S[0, t, 1:1 + len(q)]).max() < 1e-6, t


def test_linearisation_predicts_a_small_perturbation(settled):
    wenv, u0 = settled
    io_ = oracle.EnvStateIO(wenv.base)
    A, B, x0, snap = oracle.linearize(io_, u0)
    nv = io_.nv
    dx = np.zeros(io_.nx)
    dx[nv + 4] = 0.02                           # a small pitch-rate kick: contact-consistent
    io_.set_x(x0, snap)
    io_.step(u0)
    nominal = io_.x(snap["qpos"])
    io_.set_x(x0 + dx, snap)
    io_.step(u0)
    true = io_.x(snap["qpos"])
    io_.restore(snap)
    pred = nominal + A @ dx
    assert np.abs(pred - true).max() < 0.05 * np.abs(true - nominal).max()


def test_unused_ankle_servo_on_the_fitted_spring_is_the_law(cfg):
    """k = 7, ka = kd = 0 leaves the plant exactly as it is: the same episodes as the law."""
    scen = [s for s in suite_scenarios() if s.name == "push_5N"]
    import functools
    f = functools.partial(oracle.ankle_factory, cfg["residual_baseline"], 7.0, 0.0, 0.0)
    r = oracle.run_bound(f, cfg, scen, 777, 2, seconds=4.0, workers=2)
    law = suite.evaluate(policies.HoldPosePolicy(cfg["residual_baseline"], label="law"), cfg, scen, episodes=2,
                         seconds=4.0, seed=777, workers=2, progress=lambda *a, **k: None)
    for x, y in zip(r.scenarios[0].episodes, law.scenarios[0].episodes):
        assert x.steps == y.steps and x.fell == y.fell
        assert x.lean_deg == pytest.approx(y.lean_deg, abs=1e-9)


def test_ankle_servo_delays_its_command_by_delay_steps(settled):
    wenv, _ = settled
    b = wenv.base
    pol = oracle.AnkleServoPolicy(b.episode_baseline, k=7.0, ka=1.0, kd=0.0, delay_steps=2)
    pol.bind(b)
    pol.reset()
    adr = pol.adr[0]
    saved = b._prior_signals.copy()
    try:
        b._prior_signals[:] = 0.0
        b._prior_signals[0] = 0.1                      # a step in pg_x at t = 0
        seen = []
        for _ in range(4):
            pol.act(None)
            seen.append(float(b.env.unwrapped.model.qpos_spring[adr]))
    finally:
        b._prior_signals[:] = saved
        b.env.unwrapped.model.qpos_spring[pol.adr] = 0.0
    assert seen[0] == 0.0 and seen[1] == 0.0          # dead time: nothing for 2 steps
    assert seen[2] == pytest.approx(oracle.ANKLE_ALPHA * 0.1)
    assert 0.0 < seen[2] < seen[3] < 0.1              # then the first-order lag
