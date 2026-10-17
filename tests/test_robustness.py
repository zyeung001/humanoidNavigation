"""Scenario mechanics: each test pins one thing a score depends on being true."""

import numpy as np
import pytest
import yaml

from src.environments import make_standing_env
from src.robustness import ScenarioWrapper, scenarios as sc, suite
from src.robustness.policies import HoldPosePolicy

CFG = yaml.safe_load(open("config/real_humanoid_stand.yaml"))["standing"]
FRAME = 57          # proprioceptive frame: pg 3 | angvel 3 | jpos 17 | jvel 17 | last_action 17


def _env(**overrides):
    cfg = dict(CFG, debug_print_every=0, **overrides)
    return make_standing_env(config=cfg)


@pytest.fixture(scope="module")
def wenv():
    return suite.make_eval_env(CFG, seconds=3.0)


def _lean_deg(env):
    return suite._lean_deg(env.unwrapped.data.qpos)


# ---------------------------------------------------------------- the wrapper adds nothing

def test_wrapper_without_a_scenario_changes_nothing():
    """No mix, no scenario: the wrapped env must produce the bare env's exact trajectory,
    or every training run that wraps it has silently changed."""
    bare, wrapped = _env(), ScenarioWrapper(_env())
    acts = np.random.default_rng(0).uniform(-0.2, 0.2, size=(30, 17)).astype(np.float32)
    traj = []
    for env in (bare, wrapped):
        np.random.seed(7)
        obs, _ = env.reset(seed=7)
        out = [obs]
        for a in acts:
            obs, r, *_ = env.step(a)
            out.append(np.append(obs, r))
        traj.append(np.concatenate(out))
    np.testing.assert_array_equal(traj[0], traj[1])


# ---------------------------------------------------------------- sensors

def test_sensor_offsets_reach_the_observation_exactly():
    env = _env(obs_noise=False)
    env.reset(seed=0)
    clean = env._proprioceptive_features()
    gyro = np.array([0.1, -0.05, 0.0])
    enc = np.linspace(-0.05, 0.05, 17)
    env.set_sensor_offsets(gyro=gyro, jpos=enc)
    off = env._proprioceptive_features()
    np.testing.assert_allclose(off[3:6] - clean[3:6], gyro, atol=1e-6)
    np.testing.assert_allclose(off[6:23] - clean[6:23], enc, atol=1e-6)
    np.testing.assert_array_equal(off[:3], clean[:3])
    env.set_sensor_offsets()
    np.testing.assert_array_equal(env._proprioceptive_features(), clean)


def test_imu_tilt_rotates_reported_gravity_by_the_requested_angle():
    env = _env(obs_noise=False)
    env.reset(seed=0)
    pg = env._proprioceptive_features()[:3]
    from src.robustness.wrapper import _rotation
    env.set_sensor_offsets(imu_rotation=_rotation(np.array([1.0, 0.0, 0.0]), np.radians(4.0)))
    pg2 = env._proprioceptive_features()[:3]
    angle = np.degrees(np.arccos(np.clip(np.dot(pg, pg2) / np.linalg.norm(pg) / np.linalg.norm(pg2), -1, 1)))
    assert abs(angle - 4.0) < 0.05


# ---------------------------------------------------------------- start state

def test_lean_start_leans_by_the_requested_angle_and_stands_on_the_floor(wenv):
    s = sc.suite_scenarios(names=["lean_6deg"])[0]
    np.random.seed(1)
    wenv.reset(seed=1, options={"scenario": s})
    base = wenv.base
    assert abs(_lean_deg(base) - 6.0) < 1.5          # reset noise is ~0.01 rad on top
    low = base._lowest_collision_z(base.env.unwrapped.model, base.env.unwrapped.data)
    assert abs(low - 1e-4) < 1e-6


def test_starting_on_the_pose_starts_the_servos_there_too(wenv):
    s = sc.Scenario("on_pose", "start", "", start=sc.Start(at_baseline=True))
    np.random.seed(2)
    wenv.reset(seed=2, options={"scenario": s})
    b = wenv.base
    # residual_baseline_rand can push a baseline past a joint stop (the elbows stop at 0);
    # the robot is placed on the nearest pose it can physically hold.
    pose = np.clip(b.episode_baseline, b.env.action_space.low, b.env.action_space.high)
    q = b.env.unwrapped.data.qpos[b.act_qposadr]
    np.testing.assert_allclose(q, pose, atol=1e-6)
    np.testing.assert_allclose(b.prev_action, pose, atol=1e-6)
    np.testing.assert_allclose(b._servo_state, pose, atol=1e-6)


def test_deploy_start_keeps_command_smoothing_at_zero(wenv):
    s = sc.suite_scenarios(names=["deploy_start"])[0]
    np.random.seed(3)
    wenv.reset(seed=3, options={"scenario": s})
    np.testing.assert_array_equal(wenv.base.prev_action, 0.0)


# ---------------------------------------------------------------- disturbances

def test_a_push_is_applied_for_its_duration_then_removed(wenv):
    s = sc.Scenario("p", "push", "", disturb=sc.Disturb(
        kind="push", magnitude=(5, 5), duration_s=(0.1, 0.1), first_s=(0.05, 0.05),
        interval_s=(10, 10)), env_pushes=False, env_randomisation=False)
    np.random.seed(4)
    wenv.reset(seed=4, options={"scenario": s})
    data = wenv.base.env.unwrapped.data
    torso = wenv._body("torso")
    applied = []
    for _ in range(10):
        wenv.step(wenv.base.episode_baseline)
        applied.append(float(np.linalg.norm(data.xfrc_applied[torso, :3])))
    # first_s 0.05 s = step 2; duration 0.1 s = 4 steps
    assert applied[:2] == [0.0, 0.0]
    np.testing.assert_allclose(applied[2:6], 5.0)
    assert applied[6:] == [0.0] * 4
    assert wenv.disturbance_steps == [2]


def test_paired_scenarios_get_the_same_pushes(wenv):
    """Episode i of push_5N and push_5N_gyro must differ only by the gyro."""
    forces = {}
    for name in ("push_5N", "push_5N_gyro"):
        s = sc.suite_scenarios(names=[name])[0]
        np.random.seed(5)
        wenv.reset(seed=5, options={"scenario": s})
        f = []
        for _ in range(110):
            wenv.step(wenv.base.episode_baseline)
            f.append(wenv.base.env.unwrapped.data.xfrc_applied[wenv._body("torso"), :3].copy())
        forces[name] = (np.array(f), list(wenv.disturbance_steps))
    np.testing.assert_array_equal(forces["push_5N"][0], forces["push_5N_gyro"][0])
    assert forces["push_5N"][1] == forces["push_5N_gyro"][1]


# ---------------------------------------------------------------- one episode never leaks into the next

def test_plant_overrides_are_undone_before_the_next_scenario(wenv):
    m = wenv.base.env.unwrapped.model
    nominal = m.body_mass.copy()
    odd = sc.suite_scenarios(names=["push_5N_oddbody"])[0]
    np.random.seed(6)
    wenv.reset(seed=6, options={"scenario": odd})
    assert not np.allclose(m.body_mass, nominal)
    wenv.reset(seed=6, options={"scenario": sc.suite_scenarios(names=["quiet"])[0]})
    np.testing.assert_array_equal(m.body_mass, nominal)
    assert wenv.base.actuator_lag_scale == 1.0


def test_an_episode_is_reproducible_from_its_seed(wenv):
    pol = HoldPosePolicy(CFG["residual_baseline"])
    s = sc.suite_scenarios(names=["push_5N_noise"])[0]
    a = suite.run_episode(pol, wenv, s, seed=11, steps=80)
    suite.run_episode(pol, wenv, sc.suite_scenarios(names=["push_5N_oddbody"])[0], seed=3, steps=80)
    b = suite.run_episode(pol, wenv, s, seed=11, steps=80)
    assert a == b


# ---------------------------------------------------------------- training mix

def test_training_mix_draws_scenarios_in_proportion():
    w = ScenarioWrapper(_env(), mix={"as_configured": 3, "wide_start": 1})
    w.reset(seed=0)
    names = []
    for _ in range(200):
        _, info = w.reset()
        names.append(info["scenario"])
    frac = names.count("wide_start") / len(names)
    assert 0.15 < frac < 0.35
    with pytest.raises(KeyError):
        w.set_scenario_mix({"no_such_scenario": 1.0})


# ---------------------------------------------------------------- definitions

def test_yaml_scenarios_accept_scalars_and_reject_typos():
    s = sc.Scenario.from_dict("x", {"family": "push", "disturb": {"kind": "push", "magnitude": 3}})
    assert s.disturb.magnitude == (3.0, 3.0)
    with pytest.raises(ValueError):
        sc.Scenario.from_dict("x", {"disturb": {"kind": "push", "magnitud": 3}})
    with pytest.raises(ValueError):
        sc.Scenario.from_dict("x", {"distrub": {}})


def test_suite_names_are_unique_and_every_family_is_scored():
    names = [s.name for s in sc.SUITE]
    assert len(names) == len(set(names))
    assert {s.family for s in sc.SUITE} >= {"quiet", "start", "push", "kick", "sensor", "plant"}


# ---------------------------------------------------------------- what a mix weight means (10/9)

def test_mix_weights_are_shares_of_training_steps_not_episodes():
    """stand_pushes (10/8): 40% of episodes were pushed, but they fell after ~9 s against
    ~110 s, so pushes got 6.8% of the steps. A weight must mean a share of steps."""
    from collections import deque
    w = ScenarioWrapper(_env(), mix={"as_configured": 0.5, "torso_pushes": 0.5})
    assert w.episode_probabilities() == pytest.approx({"as_configured": 0.5, "torso_pushes": 0.5})
    w._lengths = {"as_configured": deque([4400]), "torso_pushes": deque([350])}
    p = w.episode_probabilities()
    steps = {k: p[k] * {"as_configured": 4400, "torso_pushes": 350}[k] for k in p}
    total = sum(steps.values())
    assert steps["torso_pushes"] / total == pytest.approx(0.5)
    w.mix_unit = "episodes"
    assert w.episode_probabilities() == pytest.approx({"as_configured": 0.5, "torso_pushes": 0.5})


def test_the_wrapper_measures_each_scenarios_episode_length():
    w = ScenarioWrapper(_env(max_episode_steps=20), mix={"as_configured": 1.0})
    w.reset(seed=0)
    for _ in range(20):
        *_, truncated, _ = w.step(w.base.episode_baseline)
    assert truncated and list(w._lengths["as_configured"]) == [20]


def test_every_scenario_survives_the_round_trip_into_a_run_config():
    for s in [*sc.SUITE, *sc.TRAINING]:
        back = sc.Scenario.from_dict(s.name, yaml.safe_load(yaml.safe_dump(s.to_dict())))
        assert back == s, s.name
