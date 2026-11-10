"""Per-episode IMU mounting tilt and sensor delay (StandingEnv imu_tilt_rand_deg, obs_delay_ms)."""

import copy
from pathlib import Path

import numpy as np
import pytest

from src.environments.standing_env import StandingEnv
from src.training.experiment import load_experiment

ROOT = Path(__file__).resolve().parents[1]
CFG = load_experiment(ROOT / "config/experiments/flex_plant_v3.yaml").task_cfg("standing")


def _rollout(steps=8, **over):
    c = copy.deepcopy(CFG)
    c.update(over, obs_noise=False, privileged_critic=False)
    env = StandingEnv(config=c)
    np.random.seed(0)
    frames = [env.reset(seed=0)[0]]
    acts = np.random.default_rng(1).uniform(-0.05, 0.05, (steps,) + env.action_space.shape)
    for i, a in enumerate(acts):
        np.random.seed(100 + i)
        frames.append(env.step(a.astype(np.float32))[0])
    return env, np.array(frames)


@pytest.fixture(scope="module")
def plain():
    return _rollout()


def test_off_is_the_observation_it_always_was(plain):
    _, a = plain
    _, b = _rollout(imu_tilt_rand_deg=0.0, obs_delay_ms=None)
    assert np.array_equal(a, b)


def test_the_tilt_rotates_gravity_and_gyro_by_at_most_its_bound():
    env, _ = _rollout(steps=3, imu_tilt_rand_deg=2.0)
    tilt = env._imu_tilt
    assert np.allclose(tilt @ tilt.T, np.eye(3), atol=1e-5)
    assert 0 < np.degrees(np.arccos((np.trace(tilt) - 1) / 2)) <= 2.0 + 1e-3
    env._imu_tilt = None
    clean = env._proprioceptive_features()
    env._imu_tilt = tilt
    tilted = env._proprioceptive_features()                     # same state, obs_noise off
    assert np.allclose(tilted[0:3], tilt @ clean[0:3], atol=1e-6)    # gravity
    assert np.allclose(tilted[3:6], tilt @ clean[3:6], atol=1e-6)    # gyro
    assert np.allclose(tilted[6:], clean[6:])                    # joints and last_action untouched


def test_the_delay_reads_back_in_time_and_interpolates():
    env, _ = _rollout(steps=1, obs_delay_ms=[10.0, 50.0])
    assert 0.4 - 1e-9 <= env._obs_delay_steps <= 2.0 + 1e-9    # 10-50 ms at 25 ms per frame
    env._sensor_frames.clear()
    env._obs_delay_steps = 1.5
    outs = [env._delayed_sensors(np.full(3, float(t)))[0][0] for t in range(5)]
    assert outs[0] == 0.0 and outs[1] == 0.0                     # not enough history: oldest kept
    assert outs[2:] == pytest.approx([0.5, 1.5, 2.5])            # 1.5 frames late, interpolated
    env._sensor_frames.clear()
    env._obs_delay_steps = 1.0
    outs = [env._delayed_sensors(np.full(3, float(t)))[0][0] for t in range(4)]
    assert outs == pytest.approx([0.0, 0.0, 1.0, 2.0])


def test_the_delay_is_applied_to_the_sensors_and_not_to_last_action():
    env, _ = _rollout(steps=2, obs_delay_ms=[25.0, 25.0])        # exactly one frame
    nj = env.n_joints
    env.prev_action = np.full(nj, 0.123, dtype=np.float32)
    f = env._proprioceptive_features()
    assert np.allclose(f[6 + 2 * nj:6 + 3 * nj], 0.123)          # last_action is current
