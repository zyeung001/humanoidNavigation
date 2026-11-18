"""The sim's command clip must match what the robot's deploy map lets through."""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

from src.environments.servo_limits import real_command_limits

MAP = "config/joint_servo_map.yaml"


def test_limits_match_the_deploy_conversion():
    sys.path.insert(0, str(Path("scripts/deploy").resolve()))
    try:
        from sim_real_map import SimRealMap
    finally:
        sys.path.pop(0)
    m = SimRealMap(MAP)
    lo, hi = real_command_limits(MAP, m.n)
    x = np.random.default_rng(0).uniform(-2.0, 2.0, size=(500, m.n)).astype(np.float32)
    sent = np.array([m.units_to_rad(m.rad_to_units(v)) for v in x])     # what the robot gets
    np.testing.assert_allclose(sent, np.clip(x, lo, hi), atol=1.0 / m.units_per_rad)


def test_knees_cannot_be_commanded_backward():
    lo, hi = real_command_limits(MAP, 17)
    assert -0.06 < lo[3] < 0.0 and -0.06 < lo[7] < 0.0          # straight is the limit
    assert hi[8] < 0.4                                         # waist roll stops near +22 deg


def test_env_clips_commands_to_the_real_limits():
    from src.environments.real_walking_env import make_real_walking_env
    from src.training.experiment import load_experiment
    cfg = load_experiment("config/experiments/walk_step_to.yaml").task_cfg("standing")
    cfg.update(debug_print_every=0, clip_to_servo_limits=True, residual_clamp=3.0,
               feedback_prior=None, enable_action_smoothing=False)
    with contextlib.redirect_stdout(io.StringIO()):
        e = make_real_walking_env(config=cfg)
    e.reset(seed=0)
    lo, hi = real_command_limits(MAP, 17)
    a = np.zeros(17, np.float32)
    a[3] = a[7] = -1.0                                         # ask both knees to bend backward
    a[8] = 1.0                                                 # and the waist roll far past +22
    e.step(a - e.reference(e.gait_time_s)[0])                   # walking env adds q_ref back
    assert e.prev_action[3] >= lo[3] - 1e-6 and e.prev_action[7] >= lo[7] - 1e-6
    assert e.prev_action[8] <= hi[8] + 1e-6
