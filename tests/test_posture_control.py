"""The slow posture controller (scripts/deploy/posture_control.py): one file for the robot and the sim."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.robustness import policies

PATH = Path("scripts/deploy/posture_control.py")
CFG = dict(gain=1.0, tau_s=1.0, zero_deg=None, min_deg=-3.0, max_deg=15.0, self_tilt=-1.34,
           settle_s=1.0, window_s=1.5, max_spread_deg=1.5)
DT = 0.025


def _module():
    spec = importlib.util.spec_from_file_location("posture_control", PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pg(pitch_deg):
    a = np.radians(pitch_deg)
    return np.array([np.sin(a), 0.0, -np.cos(a)], dtype=np.float32)


def _run(ctrl, pitches):
    return [float(np.degrees(ctrl.offsets(_pg(p))[3])) for p in pitches]


def test_no_correction_until_it_has_seen_a_still_stretch_then_the_zero_is_that_stretch():
    ctrl = _module().PostureController.from_config(CFG, dt=DT)
    knees = _run(ctrl, [2.0] * int(2.4 / DT))
    assert not ctrl.engaged and max(abs(k) for k in knees) == 0.0
    _run(ctrl, [2.0] * int(0.2 / DT))
    assert ctrl.engaged and ctrl.zero_deg == pytest.approx(2.0, abs=1e-6)


def test_moving_while_it_measures_restarts_the_window():
    ctrl = _module().PostureController.from_config(dict(CFG, tau_s=0.05), dt=DT)
    _run(ctrl, [0.0] * int(1.8 / DT) + [4.0] * 10)       # a jolt 0.8 s into the window
    assert not ctrl.engaged
    _run(ctrl, [4.0] * int(1.6 / DT))
    assert ctrl.engaged and ctrl.zero_deg == pytest.approx(4.0, abs=0.05)


def test_a_slow_forward_lean_bends_the_knees_by_gain_times_lean_once_its_own_tilt_is_removed():
    mod = _module()
    ctrl = mod.PostureController.from_config(CFG, dt=DT)
    _run(ctrl, [0.0] * int(3 / DT))
    assert ctrl.engaged
    # the plant: knee correction c pitches the pelvis back by 1.34 c on top of a 4 deg lean
    corr = 0.0
    for _ in range(int(10 / DT)):
        corr = float(np.degrees(ctrl.offsets(_pg(4.0 - 1.34 * corr))[3]))
    assert corr == pytest.approx(4.0, abs=0.05)                 # gain 1 x the 4 deg lean
    assert np.degrees(ctrl.offsets(_pg(4.0 - 1.34 * corr))[7]) == pytest.approx(corr, abs=1e-4)


def test_the_correction_stops_at_its_limits():
    ctrl = _module().PostureController.from_config(dict(CFG, zero_deg=0.0), dt=DT)
    assert max(_run(ctrl, [40.0] * 400)) == pytest.approx(15.0)
    ctrl.reset()
    assert min(_run(ctrl, [-40.0] * 400)) == pytest.approx(-3.0)


def _shipped(**over):
    import yaml
    cfg = yaml.safe_load(open("config/stances/posture.yaml"))["posture"]
    ctrl = _module().PostureController.from_config(dict(cfg, **over), dt=DT)
    _run(ctrl, [0.0] * int(3 / DT))
    assert ctrl.engaged
    return ctrl


def test_a_push_barely_moves_the_knees_but_a_held_lean_does():
    ctrl = _shipped()
    push = _run(ctrl, [20.0] * int(1.0 / DT) + [-10.0] * int(0.5 / DT) + [0.0] * int(3 / DT))
    assert max(abs(k) for k in push) < 1.0          # the 10/24 settings went to the +12 deg limit
    held = _run(ctrl, [4.0] * int(10 / DT))
    assert held[-1] == pytest.approx(ctrl.gain * (4.0 - ctrl.deadband_deg), abs=0.05)


def test_no_correction_inside_the_deadband_and_the_knees_never_jump():
    ctrl = _shipped()
    assert max(abs(k) for k in _run(ctrl, [0.9] * int(10 / DT))) == 0.0
    ks = _run(ctrl, [40.0] * int(10 / DT))
    assert max(np.diff(ks)) <= ctrl.max_rate_deg_s * DT + 1e-6
    assert ks[-1] == pytest.approx(ctrl.max_deg)


def test_the_sim_policy_is_the_stance_plus_the_robot_controller_and_rebuilds_in_workers():
    stance = np.linspace(-0.1, 0.1, 17).astype(np.float32)
    pol = policies.PosturePolicy(stance, dict(CFG, zero_deg=0.0), dt=DT, label="p")
    ref = _module().PostureController.from_config(dict(CFG, zero_deg=0.0), dt=DT)
    obs = np.zeros(228, dtype=np.float32)
    obs[-57:-54] = _pg(3.0)
    np.testing.assert_allclose(pol.act(obs), stance + ref.offsets(_pg(3.0)), atol=1e-7)
    twin = policies.rebuild(pol.recipe)
    twin.reset()
    pol.reset()
    np.testing.assert_allclose(twin.act(obs), pol.act(obs), atol=0)


def _hold_npz(path, stance, obs_dim=228):
    """A hold bundle (what models/real_standing_hold_knee2.npz is): no network, outputs `stance`.
    Built here because models/*.npz is gitignored, so CI never has the shipped one."""
    stance = np.asarray(stance, np.float32)
    np.savez(path, n_hidden=np.array(0, dtype=np.int64),
             out_w=np.zeros((len(stance), obs_dim), np.float32), out_b=stance,
             obs_mean=np.zeros(obs_dim, np.float32), obs_var=np.ones(obs_dim, np.float32),
             clip_obs=np.array(50.0, np.float32), epsilon=np.array(1e-8, np.float32))
    return path


def test_deploy_loads_the_shipped_posture_config(tmp_path):
    import subprocess
    import sys

    import yaml
    with open("config/stances/knee2.yaml") as f:
        stance = yaml.safe_load(f)["standing"]["residual_baseline"]
    npz = _hold_npz(tmp_path / "hold_knee2.npz", stance)
    out = subprocess.run([sys.executable, "scripts/deploy/deploy_standing.py", "--dry-run",
                          "--policy-npz", str(npz),
                          "--residual-config", "config/stances/knee2.yaml",
                          "--posture", "config/stances/posture.yaml"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "POSTURE:" in out.stdout and "[dry-run] OK" in out.stdout
