"""A policy bundle must never silently lose a feature on a Pi whose numpy_policy.py is older.

export_policy.py writes `requires` (the runtime features the bundle's optional keys need) and a
format version; numpy_policy.NumpyPolicy.load refuses anything it cannot fully run."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.environments import feedback_prior

NAMES = feedback_prior.joint_names()
PRIOR = {"R_knee": {"pg_x": 0.5, "gyro_y": 0.8}, "L_knee": {"pg_x": 0.5, "gyro_y": 0.8}}


def _module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


numpy_policy = _module("scripts/deploy/numpy_policy.py")
export = _module("scripts/deploy/export_policy.py")


def _mlp(obs_dim=12, act_dim=17, hidden=(8, 6), seed=0):
    """The arrays export_policy.py main() builds from an SB3 model, without SB3."""
    rng = np.random.default_rng(seed)
    arrays = {"n_hidden": np.array(len(hidden), dtype=np.int64)}
    fan_in = obs_dim
    for i, h in enumerate(hidden):
        arrays[f"h{i}_w"] = rng.normal(size=(h, fan_in)).astype(np.float32)
        arrays[f"h{i}_b"] = rng.normal(size=h).astype(np.float32)
        fan_in = h
    arrays["out_w"] = rng.normal(size=(act_dim, fan_in)).astype(np.float32)
    arrays["out_b"] = rng.normal(size=act_dim).astype(np.float32)
    arrays["obs_mean"] = rng.normal(size=obs_dim).astype(np.float32)
    arrays["obs_var"] = rng.uniform(0.5, 2.0, size=obs_dim).astype(np.float32)
    arrays["clip_obs"] = np.array(10.0, np.float32)
    arrays["epsilon"] = np.array(1e-8, np.float32)
    return arrays


def test_export_then_load_round_trips_the_network_and_names_its_features(tmp_path):
    arrays = _mlp()
    arrays["prior_gain"] = feedback_prior.gain_matrix(PRIOR, NAMES)
    arrays["prior_deadband"] = feedback_prior.deadband_vector({"gyro_y": 0.05})
    out = tmp_path / "policy.npz"
    export.save_bundle(str(out), arrays)

    with np.load(out) as d:
        assert int(d["format_version"]) == numpy_policy.BUNDLE_FORMAT
        assert list(d["requires"]) == ["feedback_prior", "prior_deadband"]
    pol = numpy_policy.NumpyPolicy.load(str(out))
    assert pol.requires == ["feedback_prior", "prior_deadband"]
    obs = np.random.default_rng(1).normal(size=12).astype(np.float32)
    x = np.clip((obs - arrays["obs_mean"]) / np.sqrt(arrays["obs_var"] + 1e-8), -10, 10)
    for i in range(2):
        x = numpy_policy.silu(x @ arrays[f"h{i}_w"].T + arrays[f"h{i}_b"])
    np.testing.assert_allclose(pol.predict(obs), x @ arrays["out_w"].T + arrays["out_b"], rtol=1e-6)
    np.testing.assert_array_equal(pol.prior_gain, arrays["prior_gain"])
    np.testing.assert_array_equal(pol.prior_deadband, arrays["prior_deadband"])


def test_a_plain_mlp_requires_nothing(tmp_path):
    out = tmp_path / "plain.npz"
    export.save_bundle(str(out), _mlp())
    assert numpy_policy.NumpyPolicy.load(str(out)).requires == []
    assert numpy_policy.describe(str(out)) == {"format": 2, "requires": [], "legacy": False}


def test_the_prior_only_bundle_requires_the_prior_and_its_deadband(tmp_path):
    k = feedback_prior.gain_matrix(PRIOR, NAMES)
    out = tmp_path / "law.npz"
    export.write_prior_only(str(out), k, "test", deadband=feedback_prior.deadband_vector({"gyro_y": 0.05}))
    assert numpy_policy.NumpyPolicy.load(str(out)).requires == ["feedback_prior", "prior_deadband"]


def test_a_required_feature_this_version_lacks_is_refused_by_name(tmp_path):
    """What an old Pi copy sees: the 10/25 bundle needs prior_deadband, its loader predates it."""
    out = tmp_path / "future.npz"
    arrays = numpy_policy.stamp(_mlp())
    arrays["requires"] = np.array(["obs_history8"], dtype=str)
    np.savez(out, **arrays)
    with pytest.raises(numpy_policy.BundleError, match="obs_history8") as e:
        numpy_policy.NumpyPolicy.load(str(out))
    assert "numpy_policy.py" in str(e.value) and "sync" in str(e.value)


def test_an_old_loader_refuses_a_bundle_needing_a_feature_it_predates(tmp_path):
    """The same check from the other side: today's bundle, a loader from before the deadband."""
    out = tmp_path / "kneelaw_db.npz"
    export.write_prior_only(str(out), feedback_prior.gain_matrix(PRIOR, NAMES), "test",
                            deadband=feedback_prior.deadband_vector({"gyro_y": 0.05}))
    old = _module("scripts/deploy/numpy_policy.py")
    old.FEATURE_KEYS = {"prior_gain": "feedback_prior"}
    old.SUPPORTED_FEATURES = frozenset(old.FEATURE_KEYS.values())
    with pytest.raises(old.BundleError, match="prior_deadband"):
        old.NumpyPolicy.load(str(out))


def test_an_unknown_key_without_requires_is_refused(tmp_path):
    """A bundle written before `requires` existed by a newer exporter: no list to check, but the
    key itself would be ignored -- refuse it."""
    out = tmp_path / "legacy_new_key.npz"
    np.savez(out, **_mlp(), prior_gain=np.zeros((17, 6), np.float32),
             posture_gain=np.ones(3, np.float32))
    with pytest.raises(numpy_policy.BundleError, match="posture_gain"):
        numpy_policy.NumpyPolicy.load(str(out))


def test_a_stray_hidden_layer_is_an_unknown_key(tmp_path):
    out = tmp_path / "extra_layer.npz"
    arrays = _mlp(hidden=(8,))
    arrays["h1_w"] = np.zeros((4, 8), np.float32)
    np.savez(out, **arrays)
    with pytest.raises(numpy_policy.BundleError, match="h1_w"):
        numpy_policy.NumpyPolicy.load(str(out))


def test_a_newer_bundle_format_is_refused(tmp_path):
    out = tmp_path / "format9.npz"
    arrays = numpy_policy.stamp(_mlp())
    arrays["format_version"] = np.array(9, dtype=np.int64)
    np.savez(out, **arrays)
    with pytest.raises(numpy_policy.BundleError, match="format 9"):
        numpy_policy.NumpyPolicy.load(str(out))


def test_requires_that_disagrees_with_the_keys_is_refused(tmp_path):
    out = tmp_path / "liar.npz"
    arrays = numpy_policy.stamp(_mlp())
    arrays["prior_gain"] = np.ones((17, 6), np.float32)     # carried, but not required
    np.savez(out, **arrays)
    with pytest.raises(numpy_policy.BundleError, match="inconsistent"):
        numpy_policy.NumpyPolicy.load(str(out))


@pytest.mark.parametrize("extra", [{}, {"prior_gain": np.ones((17, 6), np.float32)},
                                   {"prior_gain": np.ones((17, 6), np.float32),
                                    "prior_deadband": np.full(6, 0.05, np.float32)}])
def test_a_legacy_bundle_with_only_understood_keys_still_loads(tmp_path, extra):
    """Every bundle on the laptop and the Pi today has no `requires`: they keep working."""
    out = tmp_path / "legacy.npz"
    np.savez(out, **_mlp(), **extra)
    pol = numpy_policy.NumpyPolicy.load(str(out))
    assert pol.requires == numpy_policy.bundle_features(extra)
    assert numpy_policy.describe(str(out))["legacy"] is True
    assert (pol.prior_gain is None) == ("prior_gain" not in extra)
