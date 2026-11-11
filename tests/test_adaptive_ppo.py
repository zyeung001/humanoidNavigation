"""KL-adaptive PPO and the privileged critic (src/training/adaptive_kl_ppo.py, asymmetric_policy.py)."""

import copy
import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from src.training.adaptive_kl_ppo import AdaptiveKLPPO
from src.training.asymmetric_policy import AsymmetricActorCriticPolicy

ROOT = Path(__file__).resolve().parents[1]
ACTOR = 2                                    # Pendulum obs = 3: actor reads 2, critic all 3


def _model(**kw):
    env = gym.make("Pendulum-v1")
    return AdaptiveKLPPO(AsymmetricActorCriticPolicy, env, n_steps=64, batch_size=32, n_epochs=2,
                         learning_rate=1e-3, seed=0, device="cpu", verbose=0,
                         policy_kwargs={"actor_obs_dim": ACTOR, "net_arch": dict(pi=[16], vf=[16]),
                                        "activation_fn": torch.nn.SiLU}, **kw)   # SiLU: what export assumes


def test_the_lr_follows_the_rsl_rl_rule_and_its_clamps():
    m = _model(desired_kl=0.01, lr_min=1e-4, lr_max=2e-3)
    assert m.adaptive_lr == pytest.approx(1e-3)
    assert m.adapt_lr(0.03) == pytest.approx(1e-3 / 1.5)          # > 2x desired: smaller
    assert m.adapt_lr(0.015) == pytest.approx(1e-3 / 1.5)         # inside the band: unchanged
    assert m.adapt_lr(0.001) == pytest.approx(1e-3)               # < half desired: larger
    assert m.adapt_lr(0.0) == pytest.approx(1e-3)                 # frozen actor (kl 0): unchanged
    for _ in range(20):
        m.adapt_lr(0.001)
    assert m.adaptive_lr == pytest.approx(2e-3)                   # lr_max
    for _ in range(40):
        m.adapt_lr(1.0)
    assert m.adaptive_lr == pytest.approx(1e-4)                   # lr_min


def test_target_kl_is_refused():
    with pytest.raises(ValueError, match="target_kl"):
        _model(target_kl=0.02)


def test_the_actor_never_sees_the_privileged_dims_and_the_critic_does():
    pol = _model().policy
    obs = torch.randn(8, 3)
    other = obs.clone()
    other[:, ACTOR:] += 5.0
    with torch.no_grad():
        a = pol.get_distribution(obs).distribution.mean
        b = pol.get_distribution(other).distribution.mean
        assert torch.equal(a, b)
        assert not torch.allclose(pol.predict_values(obs), pol.predict_values(other))
    obs.requires_grad_(True)
    pol.get_distribution(obs).distribution.mean.sum().backward()
    assert torch.all(obs.grad[:, ACTOR:] == 0) and torch.any(obs.grad[:, :ACTOR] != 0)


def test_learn_logs_kl_and_lr_and_a_saved_model_keeps_its_actor_width(tmp_path):
    m = _model()
    m.learn(128)
    vals = m.logger.name_to_value
    assert "train/approx_kl" in vals and "train/learning_rate" in vals and "train/approx_kl_max" in vals
    assert vals["train/learning_rate"] == pytest.approx(m.adaptive_lr)
    m.save(tmp_path / "m.zip")
    back = AdaptiveKLPPO.load(tmp_path / "m.zip", device="cpu")
    assert back.policy.actor_obs_dim == ACTOR and back.adaptive_lr == pytest.approx(m.adaptive_lr)
    assert back.policy.mlp_extractor.policy_net[0].in_features == ACTOR


def test_the_env_appends_the_sim_truth_after_an_unchanged_actor_observation():
    from src.environments.standing_env import StandingEnv
    from src.training.experiment import load_experiment
    cfg = load_experiment(ROOT / "config/experiments/flex_plant_v3.yaml").task_cfg("standing")

    def rollout(priv):
        c = copy.deepcopy(cfg)
        c["privileged_critic"] = priv
        env = StandingEnv(config=c)
        np.random.seed(0)
        out = [env.reset(seed=0)[0]]
        act = np.random.default_rng(1).uniform(-0.05, 0.05, (10,) + env.action_space.shape)
        for i, a in enumerate(act):
            np.random.seed(100 + i)
            out.append(env.step(a.astype(np.float32))[0])
        return env, np.array(out)
    off_env, off = rollout(False)
    on_env, on = rollout(True)
    assert off_env.observation_space.shape == (228,) and on_env.actor_obs_dim == 228
    assert on.shape[1] == on_env.observation_space.shape[0] == 228 + on_env.privileged_dim
    assert np.array_equal(off, on[:, :228])
    tail = on[:, 228:]
    assert np.allclose(tail[:, 2], -1.0, atol=0.05)                 # clean gravity z, upright
    na = on_env.action_space.shape[0]
    kp = tail[0, 12:12 + na]
    assert np.all((kp > 0.5) & (kp < 1.5)) and kp.std() > 0         # this episode's kp scales


def test_export_ships_the_actor_only_and_matches_sb3(tmp_path, monkeypatch):
    m = _model()
    m.save(tmp_path / "m.zip")
    vn = SimpleNamespace(obs_rms=SimpleNamespace(mean=np.array([0.1, -0.2, 3.0]),
                                                 var=np.array([1.5, 0.5, 9.0])),
                         clip_obs=10.0, epsilon=1e-8)
    with open(tmp_path / "vn.pkl", "wb") as f:
        pickle.dump(vn, f)
    spec = importlib.util.spec_from_file_location("export_policy", ROOT / "scripts/deploy/export_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["export_policy.py", "--model", str(tmp_path / "m.zip"),
                                      "--vecnorm", str(tmp_path / "vn.pkl"), "--out", str(tmp_path / "p.npz")])
    mod.main()
    d = np.load(tmp_path / "p.npz")
    assert d["obs_mean"].shape == (ACTOR,) and np.allclose(d["obs_var"], [1.5, 0.5])

    spec = importlib.util.spec_from_file_location("numpy_policy", ROOT / "scripts/deploy/numpy_policy.py")
    npmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(npmod)
    npol = npmod.NumpyPolicy.load(str(tmp_path / "p.npz"))
    raw = np.array([0.3, -0.1, 2.0], np.float32)
    z = np.clip((raw - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8), -10, 10).astype(np.float32)
    with torch.no_grad():
        want = m.policy.get_distribution(torch.as_tensor(z[None])).distribution.mean.numpy()[0]
    assert np.allclose(npol.predict(raw[:ACTOR]), want, atol=1e-5)

    from src.robustness.policies import NpzPolicy
    assert np.allclose(NpzPolicy(tmp_path / "p.npz").act(raw), want, atol=1e-5)   # full obs, sliced


# ---------------------------------------------------------------- LCP smoothness (step 2)

def test_the_lcp_estimate_is_the_jacobian_norm_on_average():
    from src.training.adaptive_kl_ppo import lcp_penalty
    pol = _model().policy
    obs = torch.randn(4, 3)
    exact = 0.0
    for o in obs:
        jac = torch.autograd.functional.jacobian(
            lambda x: pol.get_distribution(x[None]).distribution.mean[0], o)
        exact += float(jac.pow(2).sum()) / len(obs)
    torch.manual_seed(0)
    est = np.mean([float(lcp_penalty(pol, obs)) for _ in range(4000)])
    assert est == pytest.approx(exact, rel=0.05)
    assert lcp_penalty(pol, obs).requires_grad                      # it trains the actor


def test_lcp_is_logged_when_on_and_absent_when_off():
    on = _model(lcp_lambda=1e-3)
    on.learn(128)
    assert on.logger.name_to_value["train/lcp_penalty"] > 0
    off = _model()
    off.learn(128)
    assert "train/lcp_penalty" not in off.logger.name_to_value


def test_smoothness_separates_sway_from_twitch():
    from src.robustness.suite import HF_HZ, _smoothness
    dt, t = 0.025, np.arange(400) * 0.025
    sway = [np.full(17, 0.02 * np.sin(2 * np.pi * 1.0 * x)) for x in t]
    twitch = [np.full(17, 0.02 * np.sin(2 * np.pi * 8.0 * x)) for x in t]
    rate_s, hf_s = _smoothness(sway, dt)
    rate_t, hf_t = _smoothness(twitch, dt)
    assert HF_HZ == 3.0 and hf_s < 0.01 and hf_t > 0.99
    assert rate_t == pytest.approx(8 * rate_s, rel=0.05)           # same size, 8x the frequency
    assert np.isnan(_smoothness(sway[:10], dt)[0])


def test_an_eval_saved_before_the_smoothness_metrics_still_loads(tmp_path):
    import json
    from src.robustness.suite import SuiteResult
    old = {"label": "old", "fingerprint": "f", "suite_version": "v", "seed": 1, "seconds": 15.0,
           "episodes_per_scenario": 1, "complete": True, "config": "", "git": "", "when": "",
           "step": None, "wall_s": 0.0,
           "scenarios": [{"name": "quiet", "family": "quiet", "about": "", "episodes": [
               {"seed": 0, "steps": 600, "horizon": 600, "fell": False, "lean_deg": 1.0, "sway_deg": 0.1,
                "jitter": 0.001, "excursion2": 0.0, "recover_s": [], "dt": 0.025}]}]}
    (tmp_path / "old.json").write_text(json.dumps(old))
    r = SuiteResult.load(tmp_path / "old.json")
    assert r.score == 100.0 and r.scenarios[0].episodes[0].act_rate is None
    assert np.isnan(r.metrics()["eval/quality/act_rate"])                # unmeasured reads nan


def test_lcp_batch_subsamples_the_minibatch_and_survives_save_load(tmp_path):
    m = _model(lcp_lambda=1e-3, lcp_batch=8)
    seen = []
    import src.training.adaptive_kl_ppo as mod
    real = mod.lcp_penalty
    mod.lcp_penalty = lambda pol, obs: (seen.append(len(obs)), real(pol, obs))[1]
    try:
        m.learn(128)
    finally:
        mod.lcp_penalty = real
    assert seen and set(seen) == {8}                                 # 8 of each 32-sample minibatch
    m.save(tmp_path / "m.zip")
    assert AdaptiveKLPPO.load(tmp_path / "m.zip", device="cpu").lcp_batch == 8
