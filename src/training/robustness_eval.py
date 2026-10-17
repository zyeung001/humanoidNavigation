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


def _notes_only(line: str) -> None:
    """Pass the evaluator's notes (worker count cut, pool fallback) through, but not its
    per-scenario progress lines: the full table is printed right after."""
    if not line.startswith("  ") or line.lstrip().startswith(("(", "parallel")):
        print(line)


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
        self._pick_up_where_the_run_left_off()

    def _pick_up_where_the_run_left_off(self) -> None:
        """A resumed run (--resume-run) keeps its best, its last eval and its checkpoint list,
        so "best" still means best over the whole run and pruning still keeps the newest."""
        if (self.run.best / "eval.json").exists():
            self.best = suite.SuiteResult.load(self.run.best / "eval.json")
        scored = sorted(self.run.root.glob("checkpoints/*/eval.json"))
        if scored:
            self.previous = suite.SuiteResult.load(scored[-1])
        self._saved = [p.parent for p in scored]

    def _on_training_start(self) -> None:
        self._next = (self.num_timesteps // self.every + 1) * self.every
        # Skip the starting eval if this exact step was already scored (resuming from it).
        already = (self.run.checkpoint(int(self.num_timesteps)) / "eval.json").exists()
        if already:
            self._last_step = int(self.num_timesteps)
        elif self.at_start:
            self.evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self.evaluate()
        return True

    def _on_training_end(self) -> None:
        # The last rollout overshoots the target by up to one rollout (215.00M, then training
        # stops at 215.014M), so a step-based check scored nearly the same weights twice.
        if self._last_step is None or self.num_timesteps - self._last_step >= self.every // 10:
            self.evaluate()

    def evaluate(self) -> Optional[suite.SuiteResult]:
        """Checkpoint, then score. The checkpoint is written FIRST, so if scoring fails the
        weights are still on disk; a failed score is reported and training carries on."""
        step = int(self.num_timesteps)
        ck = self.run.checkpoint(step)
        ck.mkdir(parents=True, exist_ok=True)
        self.model.save(ck / "model.zip")
        self.model.get_env().save(str(ck / "vecnorm.pkl"))
        self._last_step = step
        print(f"\n[robustness] scoring {step / 1e6:.2f}M on {len(suite.SUITE)} scenarios "
              f"x {self.episodes} episodes...", flush=True)
        try:
            pol = policies.SB3Policy.load(ck / "model.zip", ck / "vecnorm.pkl",
                                          label=f"{self.run.name}@{step / 1e6:.2f}M")
            r = suite.evaluate(pol, self.env_cfg, episodes=self.episodes, seconds=self.seconds,
                               seed=self.seed, step=step, workers=self.workers,
                               progress=_notes_only, config_path=str(self.run.root / "config.yaml"))
            r.save(ck / "eval.json")
        except Exception as e:  # noqa: BLE001 -- an eval must never cost the training run
            print(f"  EVAL FAILED at {step / 1e6:.2f}M ({type(e).__name__}: {e}). Training "
                  f"continues. The checkpoint is saved; score it later with\n"
                  f"    python scripts/eval_standing.py --run {self.run.root} "
                  f"--checkpoint checkpoints/{ck.name}")
            self.sink.log({"eval/failed": 1.0}, step)
            return None
        with open(self.run.evals, "a", encoding="utf-8") as f:
            f.write(json.dumps({"timesteps": step, "checkpoint": str(ck), **r.metrics("")}) + "\n")
        self.sink.log(r.metrics("eval/"), step)

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
        self._prune(ck)
        if self.adaptive_floor is not None and self.base_mix:
            self._adapt(r, step)
        return r

    def _prune(self, ck) -> None:
        """Keep the newest `keep` SCORED checkpoints. One whose eval failed never gets here, so
        it is never deleted: it is the only copy of those weights until someone scores it."""
        if ck not in self._saved:
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
