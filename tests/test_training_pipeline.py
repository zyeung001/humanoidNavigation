"""Experiment files, training logs, scoring maths and the comparison: one test per claim."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src.robustness.suite import EpisodeResult, ScenarioResult, SuiteResult
from src.training.experiment import load_experiment
from src.training.run_logger import MetricsSink, TrainingLogger

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


# ---------------------------------------------------------------- experiments

def test_an_experiment_is_its_base_plus_only_what_it_names(tmp_path):
    base = _write(tmp_path / "base.yaml", {"standing": {"a": 1, "nested": {"x": 1, "y": 2}, "lst": [1, 2]}})
    exp = _write(tmp_path / "e.yaml", {"experiment": {"name": "e", "base": str(base)},
                                       "overrides": {"standing": {"nested": {"y": 5}, "lst": [9]}}})
    e = load_experiment(exp)
    assert e.task_cfg() == {"a": 1, "nested": {"x": 1, "y": 5}, "lst": [9]}
    assert e.changes_flat() == {"standing.nested.y": 5, "standing.lst": [9]}


def test_chained_experiments_accumulate_changes_and_record_lineage(tmp_path):
    base = _write(tmp_path / "base.yaml", {"standing": {"a": 1, "b": 1}})
    e1 = _write(tmp_path / "e1.yaml", {"experiment": {"base": str(base), "resume": {"model": "m.zip"}},
                                       "overrides": {"standing": {"a": 2}}})
    e2 = _write(tmp_path / "e2.yaml", {"experiment": {"base": str(e1)}, "overrides": {"standing": {"b": 3}}})
    e = load_experiment(e2)
    assert e.task_cfg() == {"a": 2, "b": 3}
    assert e.changes_flat() == {"standing.a": 2, "standing.b": 3}
    assert len(e.lineage) == 2
    assert e.resume == {"model": "m.zip"}          # continues what its base continues


def test_experiment_files_reject_config_keys_outside_overrides_and_loops(tmp_path):
    base = _write(tmp_path / "base.yaml", {"standing": {}})
    bad = _write(tmp_path / "bad.yaml", {"experiment": {"base": str(base)}, "standing": {"a": 1}})
    with pytest.raises(ValueError, match="overrides"):
        load_experiment(bad)
    a = tmp_path / "a.yaml"
    b = _write(tmp_path / "b.yaml", {"experiment": {"base": str(a)}})
    _write(a, {"experiment": {"base": str(b)}})
    with pytest.raises(ValueError, match="loops"):
        load_experiment(a)


def test_a_saved_run_config_reloads_to_the_same_config_and_does_not_auto_resume(tmp_path):
    base = _write(tmp_path / "base.yaml", {"standing": {"a": 1}})
    exp = _write(tmp_path / "e.yaml", {"experiment": {"base": str(base), "resume": {"model": "m.zip"}},
                                       "overrides": {"standing": {"a": 2}}})
    e = load_experiment(exp)
    saved = tmp_path / "run_config.yaml"
    saved.write_text(e.resolved_yaml(task="standing", task_block={"a": 2, "filled": True}))
    again = load_experiment(saved)
    assert again.task_cfg() == {"a": 2, "filled": True}
    assert again.resume == {}
    assert again.changes_flat() == {"standing.a": 2}


def test_every_shipped_experiment_loads():
    for path in sorted((ROOT / "config" / "experiments").glob("*.yaml")):
        e = load_experiment(path)
        assert e.base, path
        assert e.task_cfg()["xml_file"]


# ---------------------------------------------------------------- training logger

def _fake_model(kl=0.01, ev=0.9):
    return SimpleNamespace(
        logger=SimpleNamespace(name_to_value={"train/approx_kl": kl, "train/explained_variance": ev,
                                              "train/clip_fraction": 0.1}),
        ent_coef=0.003, policy=SimpleNamespace(log_std=torch.full((17,), -2.0)), num_timesteps=0)


def _logger(tmp_path, model):
    cb = TrainingLogger(MetricsSink(tmp_path / "m.jsonl"), every=10, control_dt=0.025, verbose=0)
    cb.model = model
    cb.num_timesteps = 0
    cb._on_training_start()
    return cb


def test_training_logger_reports_falls_reward_shares_and_health(tmp_path):
    cb = _logger(tmp_path, _fake_model(kl=0.05))
    infos = [{"reward/height": 40.0, "reward/action_rate": -10.0},
             {"reward/height": 40.0, "reward/action_rate": -10.0,
              "episode": {"r": 100.0, "l": 400}, "termination_cause": "tilt"},
             {"episode": {"r": 50.0, "l": 200}, "TimeLimit.truncated": True, "scenario": "wide_start"}]
    cb.locals = {"infos": infos}
    cb.num_timesteps = 10
    cb._on_step()
    rec = json.loads((tmp_path / "m.jsonl").read_text().splitlines()[-1])
    assert rec["train/fall_rate"] == 0.5
    assert rec["train/ep_len_s"] == pytest.approx((10.0 + 5.0) / 2)
    assert rec["train/fall_cause/tilt"] == 0.5
    assert rec["train/scenario/wide_start/fall_rate"] == 0.0
    assert rec["reward/share/height"] == pytest.approx(0.8)
    assert rec["health/approx_kl_high"] == 1.0
    assert rec["health/value_not_tracking"] == 0.0
    assert rec["ppo/std_at_floor"] == 1.0


def test_entropy_coefficient_stays_positive_when_resuming():
    """The 175M -> 195M fine-tune ran at ent_coef ~ -0.024: the schedule divided cumulative
    steps by this call's steps. It must follow SB3's progress instead."""
    spec = importlib.util.spec_from_file_location("train_standing", ROOT / "scripts" / "train_standing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cb = mod.EntropyScheduleCallback(0.004, 0.001)
    cb.model = SimpleNamespace(_current_progress_remaining=1 - 175 / 195, ent_coef=None, num_timesteps=175_000_000)
    cb._on_step()
    assert 0.001 <= cb.model.ent_coef <= 0.004


# ---------------------------------------------------------------- scoring and comparison

def _result(label, fracs_by_scenario, fp=None):
    scen = [ScenarioResult(name, "push", "", [
        EpisodeResult(seed=i, steps=int(f * 600), horizon=600, fell=f < 1, lean_deg=1, sway_deg=0.1,
                      jitter=0.01, excursion2=0.0) for i, f in enumerate(fr)])
        for name, fr in fracs_by_scenario.items()]
    return SuiteResult(label=label, fingerprint=fp or label, suite_version="v", seed=1, seconds=15,
                       episodes_per_scenario=len(next(iter(fracs_by_scenario.values()))), complete=True,
                       config="", git="", when="", step=None, wall_s=0, scenarios=scen)


def test_score_is_the_mean_share_of_time_upright_over_scenarios():
    r = _result("a", {"s1": [1.0, 0.5], "s2": [0.0, 0.5]})
    assert r.score == pytest.approx(50.0)
    assert r.survival_pct == pytest.approx(25.0)
    assert r.families() == {"push": pytest.approx(50.0)}
    assert r.worst(1)[0].name == "s2"


def test_results_survive_a_json_round_trip(tmp_path):
    r = _result("a", {"s1": [1.0, 0.25]})
    back = SuiteResult.load(r.save(tmp_path / "r.json"))
    assert back.score == r.score and back.scenarios[0].episodes == r.scenarios[0].episodes


def test_paired_comparison_counts_episodes_won_and_lost():
    spec = importlib.util.spec_from_file_location("compare_models", ROOT / "scripts" / "compare_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = _result("a", {"s": [1.0, 1.0, 0.5, 0.5]})
    b = _result("b", {"s": [0.5, 0.5, 0.5, 1.0]})
    mean, se, won, lost = mod.paired(a, b)
    assert mean == pytest.approx(100 * (0.5 + 0.5 + 0 - 0.5) / 4)
    assert (won, lost) == (2, 1)
    assert se > 0


# ---------------------------------------------------------------- the env reports what it rewards

def test_reward_terms_in_info_add_up_to_the_reward():
    from src.environments import make_standing_env
    cfg = dict(yaml.safe_load(open(ROOT / "config" / "real_humanoid_stand.yaml"))["standing"],
               debug_print_every=0)
    env = make_standing_env(config=cfg)
    env.reset(seed=0)
    for _ in range(5):
        _, r, terminated, _, info = env.step(env.episode_baseline)
    terms = {k: v for k, v in info.items() if k.startswith("reward/")}
    assert len(terms) == 15
    assert sum(terms.values()) == pytest.approx(r)
    assert np.isfinite(r)


def test_the_turning_env_survives_a_fall_with_its_own_reward():
    """RealTurningEnv overrides the reward, so the reward-term logging must not assume it ran."""
    from src.environments import make_real_turning_env
    cfg = yaml.safe_load(open(ROOT / "config" / "turning" / "heading_pelvis.yaml"))
    cfg = dict(cfg.get("turning") or cfg["standing"], debug_print_every=0)
    env = make_real_turning_env(config=cfg)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(400):
        _, _, terminated, truncated, info = env.step(rng.uniform(-1.5, 1.5, size=17).astype(np.float32))
        if terminated:
            assert info["termination_cause"]
            break
    else:
        pytest.skip("random actions did not topple the robot in 400 steps")


# ---------------------------------------------------------------- nothing on the side may kill a run
# 10/9: Windows ran short of memory while two runs evaluated at once, W&B's service was shut
# down, and the next wandb call raised inside the training loop. Each test below is one way
# that night could have gone, and must not stop training.

def test_a_dead_wandb_is_dropped_and_the_jsonl_keeps_logging(tmp_path, capsys):
    class DeadRun:
        summary = {}

        def log(self, *a, **k):
            raise ConnectionAbortedError("WinError 10053")

        def finish(self):
            raise ConnectionAbortedError("WinError 10053")
    sink = MetricsSink(tmp_path / "m.jsonl", DeadRun())
    sink.log({"eval/score": 70.0}, 1)
    sink.log({"eval/score": 71.0}, 2)
    sink.summary({"best/score": 71.0})
    sink.close()
    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    assert [json.loads(x)["eval/score"] for x in lines] == [70.0, 71.0]
    assert capsys.readouterr().out.count("W&B log failed") == 1


def test_eval_workers_are_capped_by_free_memory(monkeypatch):
    from src.robustness import suite
    monkeypatch.setattr(suite, "available_memory_gb", lambda: 12.0)
    assert suite.memory_safe_workers(11) == 2          # (12 - 8 reserve) // 2 GB each
    monkeypatch.setattr(suite, "available_memory_gb", lambda: 5.0)
    assert suite.memory_safe_workers(11) == 1
    monkeypatch.setattr(suite, "available_memory_gb", lambda: None)
    assert suite.memory_safe_workers(11) == 11


def test_a_broken_worker_pool_falls_back_to_the_same_answer(monkeypatch):
    from src.robustness import scenarios, suite
    from src.robustness.policies import HoldPosePolicy
    cfg = yaml.safe_load(open(ROOT / "config" / "real_humanoid_stand.yaml"))["standing"]
    sc = scenarios.suite_scenarios(names=["push_5N"])
    good = suite.evaluate(HoldPosePolicy(cfg["residual_baseline"]), cfg, sc, episodes=2, seconds=3.0,
                          workers=1, progress=None)
    broken = HoldPosePolicy(cfg["residual_baseline"])
    broken.recipe = ("no-such-policy-kind",)           # every worker fails to start
    monkeypatch.setattr(suite, "memory_safe_workers", lambda n: 2)
    notes = []
    r = suite.evaluate(broken, cfg, sc, episodes=2, seconds=3.0, workers=2, progress=notes.append)
    assert any("parallel evaluation failed" in n for n in notes)
    same = lambda e: (e.seed, e.steps, e.fell, e.lean_deg, e.sway_deg)   # noqa: E731 (nan != nan)
    assert [same(e) for e in r.scenarios[0].episodes] == [same(e) for e in good.scenarios[0].episodes]


def _fake_trainer():
    class Env:
        def save(self, p):
            Path(p).write_bytes(b"vecnorm")
    return SimpleNamespace(num_timesteps=210_000_000, get_env=lambda: Env(),
                           save=lambda p: Path(p).write_bytes(b"model"))


def test_a_failed_eval_keeps_the_checkpoint_and_lets_training_continue(tmp_path, monkeypatch):
    from src.robustness import suite
    from src.training.experiment import RunDir
    from src.training.robustness_eval import RobustnessEvalCallback
    run = RunDir.create("r", base=tmp_path)

    def boom(*a, **k):
        raise MemoryError("paging file is too small")
    monkeypatch.setattr(suite, "evaluate", boom)
    monkeypatch.setattr("src.robustness.policies.SB3Policy.load", lambda *a, **k: None)
    sink = MetricsSink(run.metrics)
    cb = RobustnessEvalCallback({}, run, sink, every=5_000_000, at_start=False)
    cb.model, cb.num_timesteps = _fake_trainer(), 210_000_000
    assert cb.evaluate() is None
    ck = run.checkpoint(210_000_000)
    assert (ck / "model.zip").exists() and (ck / "vecnorm.pkl").exists()
    assert json.loads(run.metrics.read_text().splitlines()[-1])["eval/failed"] == 1.0


def test_resuming_finds_the_furthest_trained_weights_and_keeps_the_best(tmp_path):
    import zipfile
    from src.training.experiment import RunDir
    from src.training.robustness_eval import RobustnessEvalCallback
    run = RunDir.create("r", base=tmp_path)
    (run.root / "config.yaml").write_text("experiment: {resolved: true}\nstanding: {}\n")

    def weights(folder, steps):
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(folder / "model.zip", "w") as z:
            z.writestr("data", json.dumps({"num_timesteps": steps}))
        (folder / "vecnorm.pkl").write_bytes(b"")
    weights(run.checkpoint(205_000_000), 205_000_000)
    weights(run.checkpoint(210_000_000), 210_000_000)
    weights(run.root / "crashed", 212_345_678)
    assert RunDir.open(run.root).latest_weights() == run.root / "crashed"

    scored = _result("r@205M", {"push_5N": [0.5, 0.5]})
    scored.save(run.checkpoint(205_000_000) / "eval.json")
    scored.save(run.best / "eval.json")
    cb = RobustnessEvalCallback({}, run, MetricsSink(run.metrics), at_start=True)
    assert cb.best.score == scored.score and cb.previous.score == scored.score
    cb.model, cb.num_timesteps = _fake_trainer(), 205_000_000
    cb._on_training_start()                     # this step is already scored: no re-eval
    assert cb._last_step == 205_000_000
