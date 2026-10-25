"""The feedback prior must mean the same thing in training, in evaluation and on the robot."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.environments import feedback_prior, make_standing_env

CFG = yaml.safe_load(open("config/real_humanoid_stand.yaml"))["standing"]
NAMES = feedback_prior.joint_names()
PRIOR = {"R_hip_pitch": {"pg_x": 0.5, "gyro_y": 0.05}, "L_hip_pitch": {"pg_x": 0.5, "gyro_y": 0.05},
         "waist_roll": {"pg_y": -0.3}}


def _module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_an_env_with_a_prior_is_the_env_without_one_running_policy_plus_prior():
    """Bit for bit: clipping, smoothing and every penalty see policy + K @ s as one command, so
    a prior changes nothing except what the untrained policy already does."""
    k = feedback_prior.gain_matrix(PRIOR, NAMES)
    acts = np.random.default_rng(0).uniform(-0.2, 0.2, size=(40, 17)).astype(np.float32)
    traj = []
    for prior in (PRIOR, None):
        env = make_standing_env(config=dict(CFG, debug_print_every=0, feedback_prior=prior))
        np.random.seed(7)
        obs, _ = env.reset(seed=7)
        out = [obs]
        for a in acts:
            if prior is None:       # the same law, applied by the "policy" instead
                a = a + k @ obs[-57:-51]
            obs, r, *_ = env.step(a)
            out.append(np.append(obs, r))
        traj.append(np.concatenate(out))
    np.testing.assert_array_equal(traj[0], traj[1])


def test_the_prior_reads_the_frame_the_policy_saw_including_sensor_faults():
    env = make_standing_env(config=dict(CFG, debug_print_every=0, feedback_prior=PRIOR))
    env.reset(seed=1)
    env.set_sensor_offsets(gyro=np.array([0.0, 0.12, 0.0]))
    obs, *_ = env.step(np.zeros(17, np.float32))
    np.testing.assert_array_equal(env._prior_signals, obs[-57:-51])


def test_prior_config_names_joints_and_signals_and_rejects_typos():
    k = feedback_prior.gain_matrix(PRIOR, NAMES)
    assert k[NAMES.index("R_hip_pitch"), feedback_prior.SIGNALS.index("pg_x")] == np.float32(0.5)
    assert k[NAMES.index("waist_roll"), feedback_prior.SIGNALS.index("pg_y")] == np.float32(-0.3)
    assert np.count_nonzero(k) == 5
    np.testing.assert_array_equal(feedback_prior.gain_matrix(k.tolist(), NAMES), k)
    assert feedback_prior.gain_matrix(None, NAMES) is None
    with pytest.raises(ValueError):
        feedback_prior.gain_matrix({"R_hip_ptch": {"pg_x": 1}}, NAMES)
    with pytest.raises(ValueError):
        feedback_prior.gain_matrix({"R_hip_pitch": {"pgx": 1}}, NAMES)


def test_the_robot_bundle_carries_the_prior_and_applies_it_to_the_frame(tmp_path):
    """export_policy.py --prior-only: the network outputs zero, and the deploy loop's
    prior_term(frame) is the same K @ s the env adds."""
    export = _module("scripts/deploy/export_policy.py")
    numpy_policy = _module("scripts/deploy/numpy_policy.py")
    k = feedback_prior.gain_matrix(PRIOR, NAMES)
    out = tmp_path / "prior_only.npz"
    export.write_prior_only(str(out), k, "test")
    pol = numpy_policy.NumpyPolicy.load(str(out))
    frame = np.random.default_rng(1).normal(size=57).astype(np.float32)
    obs = np.tile(frame, 4)
    np.testing.assert_array_equal(pol.predict(obs), np.zeros(17, np.float32))
    np.testing.assert_array_equal(pol.prior_term(frame), (k @ frame[:6]).astype(np.float32))
    np.testing.assert_array_equal(pol.prior_gain, k)


def test_a_bundle_without_a_prior_adds_nothing():
    numpy_policy = _module("scripts/deploy/numpy_policy.py")
    pol = numpy_policy.NumpyPolicy([], np.zeros((17, 228), np.float32), np.zeros(17, np.float32),
                                   np.zeros(228, np.float32), np.ones(228, np.float32), 50.0, 1e-8)
    assert pol.prior_gain is None
    np.testing.assert_array_equal(pol.prior_term(np.ones(57, np.float32)), 0.0)


DEADBAND = {"gyro_y": 0.05}


def test_the_deadband_zeroes_small_signals_and_only_shaves_big_ones():
    db = feedback_prior.deadband_vector(DEADBAND)
    s = np.array([0.0, 0.0, -1.0, 0.0, 0.03, 0.0], np.float32)
    np.testing.assert_array_equal(feedback_prior.apply_deadband(s, db), [0, 0, -1, 0, 0, 0])
    s[4] = -0.45
    assert feedback_prior.apply_deadband(s, db)[4] == pytest.approx(-0.40)
    assert feedback_prior.deadband_vector(None) is None
    with pytest.raises(ValueError):
        feedback_prior.deadband_vector({"gyro": 0.05})


def test_a_deadbanded_prior_is_the_policy_applying_the_same_deadbanded_law():
    k = feedback_prior.gain_matrix(PRIOR, NAMES)
    db = feedback_prior.deadband_vector(DEADBAND)
    acts = np.random.default_rng(3).uniform(-0.2, 0.2, size=(40, 17)).astype(np.float32)
    traj = []
    for prior in (PRIOR, None):
        env = make_standing_env(config=dict(CFG, debug_print_every=0, feedback_prior=prior,
                                            feedback_prior_deadband=DEADBAND if prior else None))
        np.random.seed(11)
        obs, _ = env.reset(seed=11)
        out = [obs]
        for a in acts:
            if prior is None:
                a = a + k @ feedback_prior.apply_deadband(obs[-57:-51], db)
            obs, r, *_ = env.step(a)
            out.append(np.append(obs, r))
        traj.append(np.concatenate(out))
    np.testing.assert_array_equal(traj[0], traj[1])


def test_the_robot_bundle_carries_the_deadband_and_applies_it_like_the_sim(tmp_path):
    export = _module("scripts/deploy/export_policy.py")
    numpy_policy = _module("scripts/deploy/numpy_policy.py")
    k = feedback_prior.gain_matrix(PRIOR, NAMES)
    db = feedback_prior.deadband_vector(DEADBAND)
    out = tmp_path / "prior_db.npz"
    export.write_prior_only(str(out), k, "test", deadband=db)
    pol = numpy_policy.NumpyPolicy.load(str(out))
    rng = np.random.default_rng(2)
    for _ in range(20):
        frame = rng.normal(scale=0.1, size=57).astype(np.float32)
        np.testing.assert_allclose(pol.prior_term(frame),
                                   k @ feedback_prior.apply_deadband(frame[:6], db), atol=1e-7)
