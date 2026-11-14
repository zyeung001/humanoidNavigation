"""RealWalkingEnv: the reference drives the command, the clock is in the obs, the box follows it."""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

from src.environments.real_walking_env import PHASE_DIM, GaitReference, make_real_walking_env
from src.training.experiment import load_experiment

REF = "config/gaits/step_to_v1.npz"


def _cfg(**over):
    cfg = load_experiment("config/experiments/walk_step_to.yaml").task_cfg("standing")
    cfg.update(debug_print_every=0, **over)
    return cfg


@pytest.fixture(scope="module")
def env():
    with contextlib.redirect_stdout(io.StringIO()):
        return make_real_walking_env(config=_cfg(walk_stand_s=0.5, residual_baseline_rand=0.0,
                                                 obs_noise=False))


def test_reference_interpolates_and_wraps_every_period():
    t = np.array([0.0, 1.0])
    q = np.array([[0.0] * 17, [1.0] * 17])
    ref = GaitReference(t, q, np.zeros((2, 2)), period_s=2.0, advance_m=0.1)
    assert ref(0.5)[0][0] == pytest.approx(0.5)
    assert ref(1.5)[0][0] == pytest.approx(0.5)          # last sample blends back into the first
    assert ref(2.5)[0][0] == pytest.approx(0.5)          # next cycle
    assert ref.speed == pytest.approx(0.05)
    with pytest.raises(ValueError):
        GaitReference(t, q, np.zeros((2, 2)), period_s=1.0, advance_m=0.0)   # t must end before period


def test_the_shipped_reference_matches_the_robot():
    ref = GaitReference.load(REF)
    assert ref.q.shape[1] == 17 and ref.period == pytest.approx(len(ref.t) * 0.025)
    assert ref.advance > 0.02                             # it walks forward
    assert ref.foot_z[:, 0].max() > 0.015                 # and lifts the right foot


def test_obs_carries_the_gait_clock_after_the_opening_stand(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs[-1] == 0.0 and obs[-2] == pytest.approx(1.0)     # standing: cos 1, gait off
    for _ in range(int(0.5 / env._control_dt) + 1):
        obs, *_ = env.step(np.zeros(17, np.float32))
    assert obs[-1] == 1.0                                       # the clock has started


def test_zero_action_commands_the_reference_and_the_box_moves_with_it(env):
    env.reset(seed=1)
    centres = []
    for _ in range(int(2.0 / env._control_dt)):
        q_ref, _, on = env.reference(env.gait_time_s)      # what this step should be centred on
        env.step(np.zeros(17, np.float32))
        np.testing.assert_allclose(env._res_base_ep, q_ref, atol=1e-6)
        centres.append(env._res_base_ep.copy())
    assert on == 1.0
    assert np.abs(centres[-1] - centres[0]).max() > 0.05     # the box moved with the gait


def test_soles_are_on_the_floor_when_standing(env):
    env.reset(seed=2)
    for _ in range(10):
        env.step(np.zeros(17, np.float32))
    z = env.sole_heights()
    assert np.all(np.abs(z) < 0.004)


def test_reward_terms_are_logged_by_name(env):
    env.reset(seed=3)
    _, r, _, _, info = env.step(np.zeros(17, np.float32))
    terms = {k[len("reward/"):]: v for k, v in info.items() if k.startswith("reward/")}
    assert {"pose", "foot", "speed", "upright", "heading", "alive", "action_rate"} <= set(terms)
    assert r == pytest.approx(sum(terms.values()), rel=1e-5)


def test_built_cycle_closes_on_itself():
    sys.path.insert(0, str(Path("scripts").resolve()))
    try:
        import build_walk_reference as bwr
    finally:
        sys.path.pop(0)
    a = np.zeros((10, 17), np.float32)
    b = np.ones((10, 17), np.float32)
    q = bwr.join([a, b], motion_steps=8, hold_s=0.05, blend_s=0.05)
    assert len(q) == 20
    np.testing.assert_allclose(q[-1], q[0])               # the end blends into the start


def test_phase_dims_are_appended_once(env):
    obs, _ = env.reset(seed=4)
    base = env.walking_obs_dim - PHASE_DIM
    assert obs.shape[0] == base + PHASE_DIM
