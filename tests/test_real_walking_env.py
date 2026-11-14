"""RealWalkingEnv: the reference drives the command, the clock is in the obs, the box follows it."""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

from src.environments.real_walking_env import (PHASE_DIM, GaitReference, StepTracker,
                                               make_real_walking_env, steps_due, swing_ends)
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


def _feed(tracker, path):
    """Feed (z, x) of the right foot (the left stays put) and return the summed gains."""
    return sum(tracker.update(np.array([z, 0.0]), np.array([x, 0.0]))[0] for z, x in path)


def test_a_step_is_a_lift_that_lands_ahead():
    t = StepTracker(clear_m=0.012)
    t.reset(np.zeros(2))
    lift = [(0.0, 0.0), (0.006, 0.0), (0.02, 0.02), (0.01, 0.04), (0.0, 0.04)]
    assert _feed(t, lift) == pytest.approx(0.04) and t.count[0] == 1


def test_slides_hops_and_back_and_forth_are_not_steps():
    t = StepTracker(clear_m=0.012)
    t.reset(np.zeros(2))
    assert _feed(t, [(0.001, 0.01 * k) for k in range(6)]) == 0.0            # slide 5 cm
    assert _feed(t, [(0.006, 0.05), (0.02, 0.05), (0.0, 0.05)]) == 0.0      # then hop in place
    assert _feed(t, [(0.006, 0.05), (0.004, 0.08), (0.0, 0.08)]) == 0.0     # a scuff (3 mm up)
    t.reset(np.zeros(2))
    back = [(0.006, 0.0), (0.02, -0.03), (0.0, -0.03)]
    fwd = [(0.006, -0.03), (0.02, 0.0), (0.0, 0.0)]
    assert _feed(t, back) == 0.0 and _feed(t, fwd) == 0.0                   # back to the start
    assert t.count[0] == 0


def test_missed_step_deadlines_follow_the_reference_swing():
    ref = GaitReference.load(REF)
    ends = swing_ends(ref.t, ref.foot_z[:, 0], 0.012)
    assert len(ends) == 1 and 1.5 < ends[0] < 3.0                          # one right step
    assert len(swing_ends(ref.t, ref.foot_z[:, 1], 0.012)) == 0              # the left slides
    assert [steps_due(g, [3.0], 9.0) for g in (-1.0, 2.9, 3.0, 11.9, 12.0)] == [0, 0, 1, 1, 2]


def test_v2_reward_pays_stepping_and_ends_standing_still():
    """The run-1 exploit: a policy that keeps its balance and never steps must not keep the
    reference's reward. Nominal plant, so the reference's first step lands."""
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = load_experiment("config/experiments/walk_step_to_v2.yaml").task_cfg("standing")
        cfg.update(debug_print_every=0, domain_rand=False, actuator_rand=False, obs_noise=False,
                   residual_baseline_rand=0.0, walk_stand_s=1.0)
        e = make_real_walking_env(config=cfg)
    stand = e.residual_baseline.astype(np.float32)
    out = {}
    for name in ("reference", "stand still"):
        e.reset(seed=0)
        ret, term = 0.0, False
        for _ in range(int(5.0 / e._control_dt)):
            q_ref = e.reference(e.gait_time_s)[0]
            a = np.zeros(17, np.float32) if name == "reference" else np.clip(stand - q_ref, -0.2, 0.2)
            _, r, term, _, info = e.step(a)
            ret += r
            if term:
                break
        out[name] = (ret, term, e._termination_cause if term else None, info["walk_steps_R"])
    ref, still = out["reference"], out["stand still"]
    assert not ref[1] and ref[3] == 1                                        # stepped, still up
    assert still[1] and still[2] == "missed_step" and still[3] == 0
    assert ref[0] - still[0] > cfg["walk_step_bonus"]                         # by more than a step
