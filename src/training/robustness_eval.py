"""Score the policy on the robustness suite while it trains, and keep the best one.

Every `every` steps (and once at the start, so a fine-tune is measured against where it
began) this saves a checkpoint, runs the suite on it, logs eval/* and prints the table. The
checkpoint with the highest score is copied to runs/<name>/best/ -- the answer to "which
model was best" is a directory, not a guess about which scoreboard row named which weights.

ADAPTIVE MIX (training_scenarios.adaptive). After each eval the training scenario weights are
reset to base_weight * (floor + failure rate of that scenario's family), so training spends
its episodes where the policy is currently failing -- the push family when pushes knock it
over, starts when it cannot be let go leaning. The floor keeps every scenario in the mix so a
solved one is not forgotten. Off by default: it changes the training distribution during the
run, which makes two runs harder to compare.
"""

from __future__ import annotations

import json
import shutil
from typing import Dict, Optional

from stable_baselines3.common.callbacks import BaseCallback

from src.robustness import policies, suite
from src.robustness.scenarios import training_scenarios


class RobustnessEvalCallback(BaseCallback):

    def __init__(self, env_cfg: dict, run_dir, sink, every: int = 5_000_000, episodes: int = 12,
                 seconds: float = 15.0, workers: int = 1, seed: int = 12345, keep: int = 5,
                 at_start: bool = True, mix: Optional[Dict[str, float]] = None,
                 adaptive_floor: Optional[float] = None, verbose: int = 1):
        super().__init__(verbose)
        self.env_cfg = dict(env_cfg)
        self.run = run_dir
        self.sink = sink
        self.every = int(every)
        self.episodes, self.seconds, self.workers, self.seed = episodes, seconds, workers, seed
        self.keep = int(keep)
        self.at_start = at_start
        self.base_mix = dict(mix or {})
        self.adaptive_floor = adaptive_floor
        self.family = {n: s.family for n, s in training_scenarios(
            (env_cfg.get("training_scenarios") or {}).get("define")).items()}
        self.best: Optional[suite.SuiteResult] = None
        self.previous: Optional[suite.SuiteResult] = None
        self._next = None
        self._last_step = None
        self._saved = []

    def _on_training_start(self) -> None:
        self._next = (self.num_timesteps // self.every + 1) * self.every
        if self.at_start:
            self.evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self.evaluate()
        return True

    def _on_training_end(self) -> None:
        if self._last_step != self.num_timesteps:
            self.evaluate()

    def evaluate(self) -> suite.SuiteResult:
        step = int(self.num_timesteps)
        ck = self.run.checkpoint(step)
        ck.mkdir(parents=True, exist_ok=True)
        self.model.save(ck / "model.zip")
        self.model.get_env().save(str(ck / "vecnorm.pkl"))
        pol = policies.SB3Policy.load(ck / "model.zip", ck / "vecnorm.pkl",
                                      label=f"{self.run.name}@{step / 1e6:.2f}M")
        print(f"\n[robustness] scoring {step / 1e6:.2f}M on {len(suite.SUITE)} scenarios "
              f"x {self.episodes} episodes ({self.workers} workers)...", flush=True)
        r = suite.evaluate(pol, self.env_cfg, episodes=self.episodes, seconds=self.seconds,
                           seed=self.seed, step=step, workers=self.workers, progress=None,
                           config_path=str(self.run.root / "config.yaml"))
        r.save(ck / "eval.json")
        with open(self.run.evals, "a", encoding="utf-8") as f:
            f.write(json.dumps({"timesteps": step, "checkpoint": str(ck), **r.metrics("")}) + "\n")
        self.sink.log(r.metrics("eval/"), step)
        self.sink.table("eval/scenarios", ["scenario", "family", "upright_pct", "never_fell",
                                           "about"],
                        [(s.name, s.family, round(s.standing_pct, 1),
                          f"{s.survived}/{len(s.episodes)}", s.about) for s in r.scenarios], step)

        print(suite.format_result(r, previous=self.previous))
        if self.best is None or r.score > self.best.score:
            gain = "" if self.best is None else f" (+{r.score - self.best.score:.1f} over {self.best.step / 1e6:.2f}M)"
            self.best = r
            self.run.best.mkdir(parents=True, exist_ok=True)
            for name in ("model.zip", "vecnorm.pkl", "eval.json"):
                shutil.copy2(ck / name, self.run.best / name)
            self.sink.summary({"best/score": r.score, "best/score_se": r.score_se,
                               "best/step": step, "best/checkpoint": str(self.run.best)})
            print(f"  NEW BEST {r.score:.1f}{gain} -> {self.run.best}")
        else:
            print(f"  best is still {self.best.score:.1f} @ {self.best.step / 1e6:.2f}M "
                  f"({r.score - self.best.score:+.1f})")
        self.previous = r
        self._last_step = step
        self._prune(ck)
        if self.adaptive_floor is not None and self.base_mix:
            self._adapt(r, step)
        return r

    def _prune(self, ck) -> None:
        self._saved.append(ck)
        while len(self._saved) > self.keep:
            shutil.rmtree(self._saved.pop(0), ignore_errors=True)

    def _adapt(self, r: suite.SuiteResult, step: int) -> None:
        fam = r.families()
        w = {}
        for name, base in self.base_mix.items():
            fail = 1.0 - fam.get(self.family.get(name, ""), 100.0) / 100.0
            w[name] = base * (self.adaptive_floor + fail)
        total = sum(w.values()) or 1.0
        w = {k: v / total for k, v in w.items()}
        self.training_env.env_method("set_scenario_mix", w)
        self.sink.log({f"train/mix/{k}": v for k, v in w.items()}, step)
        print("  training mix now: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in
                                                 sorted(w.items(), key=lambda kv: -kv[1])))
