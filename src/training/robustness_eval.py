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

EARLY STOP (early_stop). The run's first eval (step 0 when robustness_eval_at_start) is its
START: it is never pruned, and a resumed run finds it again through start_eval.json. Once the
run is `after` steps past it, every eval asks whether the best checkpoint beats the start by
at least `k_se` paired standard errors (suite.paired, same seeds, clustered by episode). If it
does not, the run has learned nothing it can show: 'warn' says so and logs it, 'stop' ends
training (learn() returns normally, so final/ and the end-of-run steps still happen), 'off'
skips the check. 2 of the last 3 scratch runs had stopped improving by 15M and ran 7-8 h more.
The false-kill rate is untested, hence 'warn' by default.
"""

from __future__ import annotations

import json
import shutil
from typing import Dict, Optional

EARLY_STOP_DEFAULTS = {"mode": "warn", "after": 15_000_000, "k_se": 2.0}

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
                 adaptive_floor: Optional[float] = None, early_stop: Optional[dict] = None,
                 verbose: int = 1):
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
        self.early_stop = {**EARLY_STOP_DEFAULTS, **(early_stop or {})}
        if self.early_stop["mode"] not in ("warn", "stop", "off"):
            raise ValueError(f"early_stop.mode must be warn, stop or off, not {self.early_stop['mode']!r}")
        self.best: Optional[suite.SuiteResult] = None
        self.previous: Optional[suite.SuiteResult] = None
        self.start: Optional[suite.SuiteResult] = None
        self.start_ck = None
        self.stopped_early: Optional[dict] = None
        self._next = None
        self._last_step = None
        self._saved = []
        self._pick_up_where_the_run_left_off()

    @property
    def _start_pointer(self):
        return self.run.root / "start_eval.json"

    def _pick_up_where_the_run_left_off(self) -> None:
        """A resumed run (--resume-run) keeps its best, its last eval and its checkpoint list,
        so "best" still means best over the whole run and pruning still keeps the newest."""
        if (self.run.best / "eval.json").exists():
            self.best = suite.SuiteResult.load(self.run.best / "eval.json")
        if self._start_pointer.exists():
            ck = self.run.root / "checkpoints" / json.loads(self._start_pointer.read_text())["checkpoint"]
            if (ck / "eval.json").exists():
                self.start, self.start_ck = suite.SuiteResult.load(ck / "eval.json"), ck
        scored = sorted(self.run.root.glob("checkpoints/*/eval.json"))
        if scored:
            self.previous = suite.SuiteResult.load(scored[-1])
        self._saved = [p.parent for p in scored if p.parent != self.start_ck]

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
        return self.stopped_early is None

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
        if self.start is None:
            self.start, self.start_ck = r, ck
            self._start_pointer.write_text(json.dumps({"checkpoint": ck.name, "step": step}))
            print(f"  START {r.score:.1f} @ {step / 1e6:.2f}M: what the run must beat "
                  f"(kept, never pruned)")
        self._prune(ck)
        self._check_early_stop(step)
        if self.adaptive_floor is not None and self.base_mix:
            self._adapt(r, step)
        return r

    def _check_early_stop(self, step: int) -> None:
        """Has the best checkpoint beaten the start by k_se paired SEs? See the module doc."""
        es = self.early_stop
        if es["mode"] == "off" or self.start is None or self.best is None:
            return
        pr = suite.paired(self.best, self.start)        # best IS start -> a gain of exactly 0
        if pr is None:
            print("  early stop: best and start were not scored on the same episodes; no check")
            return
        gain, se = pr[0], pr[1]
        self.sink.log({"early_stop/gain_vs_start": gain, "early_stop/gain_vs_start_se": se}, step)
        since = step - int(self.start.step or 0)
        if since < int(es["after"]):
            return
        if gain >= float(es["k_se"]) * se and gain > 0:
            print(f"  early stop: best beats the start by {gain:+.1f} +- {se:.1f} "
                  f"(>= {es['k_se']:g} SE): keep training")
            return
        msg = (f"best ({self.best.score:.1f} @ {self.best.step / 1e6:.2f}M) beats the start "
               f"({self.start.score:.1f}) by only {gain:+.1f} +- {se:.1f}, under {es['k_se']:g} SE, "
               f"{since / 1e6:.1f}M steps in")
        self.sink.log({"early_stop/fired": 1.0}, step)
        if es["mode"] == "warn":
            print(f"  EARLY STOP (warn only): {msg}. Training continues "
                  f"(early_stop.mode: stop would end it here).")
            return
        self.stopped_early = {"step": step, "gain": gain, "se": se, "k_se": float(es["k_se"]),
                              "after": int(es["after"]), "best_step": self.best.step,
                              "start_step": self.start.step}
        (self.run.root / "early_stop.json").write_text(json.dumps(self.stopped_early, indent=1))
        print(f"  EARLY STOP: {msg}. Ending training; final/ is still written. To keep going:\n"
              f"    python scripts/train_standing.py --resume-run {self.run.root} --no-early-stop")

    def _prune(self, ck) -> None:
        """Keep the newest `keep` SCORED checkpoints. One whose eval failed never gets here, so
        it is never deleted: it is the only copy of those weights until someone scores it. The
        run's START checkpoint is never in the list either: early stop compares against it."""
        if ck == self.start_ck:
            return
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
