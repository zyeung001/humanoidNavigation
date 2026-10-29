"""Training logs you can act on: where every number goes, and what the trainer prints.

THE DASHBOARD, by W&B section. Read them in this order:

  eval/      THE ANSWER. Robustness score on the fixed scenario suite (src/robustness), at
             fixed seeds. Comparable across runs even when the reward, the network or the
             observation changed -- the only section that is. eval/score is the headline;
             eval/family/* and eval/scenario/* say where it comes from. Logged by
             RobustnessEvalCallback every robustness_eval_every steps.
  train/     What training episodes look like: how long they last (seconds), how often they
             end in a fall and why, per training scenario. Comparable within a run.
  reward/    Each reward term's mean per step, and its share of the total. When one term is
             most of the reward, that term is what the policy is optimising.
  ppo/       Algorithm health: approx_kl, clip_fraction, explained_variance, entropy, std.
  health/    The rules below, applied to ppo/*: 1 = tripped. The console says which and why.
  perf/      Throughput.
  validation/  Once, at the end: validation/video (the best policy on the fixed showcase
             timeline -- pushes, a kick, a gyro fault -- with arrows, support map and HUD),
             beats_held and falls. To SEE what the score measured; scripts/train_standing.py.

Every record also goes to runs/<name>/metrics.jsonl, so nothing depends on W&B being up.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# (key, test, message). Thresholds are for PPO with clip_range 0.1-0.2 on this task; they are
# warnings, not stops -- the point is that a bad sign is SAID, not buried in a table.
HEALTH_RULES = [
    ("approx_kl_high", lambda p: p.get("approx_kl", 0) > 0.03,
     "approx_kl {approx_kl:.3f} > 0.03: updates too big -- lower learning_rate or n_epochs"),
    ("clip_fraction_high", lambda p: p.get("clip_fraction", 0) > 0.3,
     "clip_fraction {clip_fraction:.2f} > 0.3: most samples clipped -- same cure as above"),
    ("value_not_tracking", lambda p: p.get("explained_variance", 1) < 0.5,
     "explained_variance {explained_variance:.2f} < 0.5: the critic is not predicting returns, "
     "so advantages are mostly noise"),
]


def _scalar(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


class MetricsSink:
    """One dict, two destinations: the run's JSONL, and W&B when a run is attached.

    W&B is a mirror, never a dependency. On 10/9 Windows ran short of memory, both runs' W&B
    services were shut down, and the next wandb.log raised inside the training loop and took
    a 15M-step fine-tune down with it. Now the first W&B failure is reported once and W&B is
    dropped for the rest of the run; the JSONL keeps every number either way.
    """

    def __init__(self, jsonl_path: Path, wandb_run=None):
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        self.wandb = wandb_run

    def _wandb_call(self, what: str, fn) -> None:
        if self.wandb is None:
            return
        try:
            fn(self.wandb)
        except Exception as e:  # noqa: BLE001 -- any W&B failure, the point is to survive it
            print(f"\n  WARNING: W&B {what} failed ({type(e).__name__}: {e}). Training continues; "
                  f"W&B is off for the rest of this run, every metric is still in {self.path}\n")
            self.wandb = None

    def log(self, metrics: Dict[str, float], step: int) -> None:
        rec = {"timesteps": int(step), "wall_time": time.time(),
               **{k: _scalar(v) for k, v in metrics.items()}}
        self._fh.write(json.dumps(rec, default=float) + "\n")
        data = {k: v for k, v in rec.items() if k != "wall_time"}
        self._wandb_call("log", lambda run: run.log(data, step=int(step)))

    def log_video(self, key: str, path, step: int, caption: str = "") -> None:
        """A video FILE (mp4/webm) to W&B under `key`; its path goes to the JSONL. Passing the
        path, not frames, means wandb uploads it as-is (frames would need moviepy, which is
        not installed). A failure is a warning, like every other W&B call here."""
        path = Path(path)
        self._fh.write(json.dumps({"timesteps": int(step), "wall_time": time.time(), key: str(path)}) + "\n")

        if self.wandb is None:
            return
        try:
            import wandb  # noqa: PLC0415
            self.wandb.log({key: wandb.Video(str(path), caption=caption,
                                             format=path.suffix.lstrip("."))}, step=int(step))
        except Exception as e:  # noqa: BLE001
            # Unlike _wandb_call, W&B stays on: a video it cannot take (format, size) says
            # nothing about the service, and close() must still finish() the run.
            print(f"  WARNING: W&B video upload failed ({type(e).__name__}: {e}); the file is "
                  f"{path}")

    def summary(self, values: Dict[str, object]) -> None:
        def write(run):
            for k, v in values.items():
                run.summary[k] = _scalar(v)
        self._wandb_call("summary", write)

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass
        self._wandb_call("finish", lambda run: run.finish())
        self.wandb = None


def start_wandb(experiment, run_dir, task_cfg: dict, mode: Optional[str], extra: dict):
    """A W&B run whose config makes runs comparable at a glance in the runs table:
    `changes.*` is what this run changed relative to its base, `cfg.*` is everything.

    The run id is kept in runs/<name>/wandb_id, so a run resumed with --resume-run keeps
    drawing on the same W&B charts instead of starting a new run."""
    mode = mode or task_cfg.get("wandb_mode") or ("online" if task_cfg.get("use_wandb") else "disabled")
    if mode == "disabled":
        return None
    try:
        import wandb  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit("W&B logging was requested but wandb is not installed: pip install wandb "
                         "(or pass --wandb disabled). Refusing to train without the logs you "
                         "asked for.") from e
    from .experiment import flatten  # noqa: PLC0415
    id_file = run_dir.root / "wandb_id"
    run_id = id_file.read_text().strip() if id_file.exists() else None
    if run_id is None:
        # Runs started before wandb_id was written: W&B names its folder run-<date>-<id>.
        old = sorted((run_dir.root / "wandb").glob("*run-*-*"))
        run_id = old[-1].name.rsplit("-", 1)[-1] if old else None
    root = (experiment.lineage or [experiment.path])[0]
    run = wandb.init(
        project=task_cfg.get("wandb_project", "humanoid-standing"),
        entity=task_cfg.get("wandb_entity"),
        name=run_dir.name, group=Path(root).stem, job_type="train",
        tags=list(experiment.tags), notes=experiment.hypothesis or experiment.description,
        dir=str(run_dir.root), mode=mode, id=run_id, resume="allow" if run_id else None,
    )
    # After init, with allow_val_change: a resumed run legitimately has a new start step.
    run.config.update({"experiment": {"name": experiment.name, "source": experiment.path,
                                      "base": experiment.base, "lineage": experiment.lineage},
                       "changes": experiment.changes_flat() or {"none": True},
                       "cfg": flatten(task_cfg), **extra}, allow_val_change=True)
    id_file.write_text(run.id)
    for key in ("eval/score", "eval/survival_pct"):
        run.define_metric(key, summary="max")
    run.define_metric("train/fall_rate", summary="last")
    return run


class TrainingLogger(BaseCallback):
    """Every `every` steps: what episodes look like, what the reward is made of, PPO health."""

    def __init__(self, sink: MetricsSink, every: int = 50_000, control_dt: float = 0.025,
                 log_std_bounds=(-2.0, 0.5), window: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.sink = sink
        self.every = int(every)
        self.dt = float(control_dt)
        self.log_std_bounds = log_std_bounds
        self._next = None
        self._episodes = deque(maxlen=window)   # (seconds, fell, scenario, cause, reward/step)
        self._terms: Dict[str, list] = defaultdict(lambda: [0.0, 0])
        self._t_last, self._n_last = None, None
        self._seen_eps = 0

    def _on_training_start(self) -> None:
        self._next = (self.num_timesteps // self.every + 1) * self.every
        self._t_last, self._n_last = time.time(), self.num_timesteps

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            for k, v in info.items():
                if k.startswith("reward/"):
                    acc = self._terms[k[7:]]
                    acc[0] += float(v)
                    acc[1] += 1
            ep = info.get("episode")
            if ep is not None:
                self._seen_eps += 1
                fell = not info.get("TimeLimit.truncated", False)
                self._episodes.append((float(ep["l"]) * self.dt, fell, info.get("scenario", "all"),
                                       info.get("termination_cause") if fell else None,
                                       float(ep["r"]) / max(1.0, float(ep["l"]))))
        if self.num_timesteps >= self._next:
            self._next += self.every
            self._flush()
        return True

    def _flush(self) -> None:
        m: Dict[str, float] = {}
        eps = list(self._episodes)
        if eps:
            secs = np.array([e[0] for e in eps])
            fell = np.array([e[1] for e in eps], dtype=float)
            m["train/ep_len_s"] = float(secs.mean())
            m["train/fall_rate"] = float(fell.mean())
            m["train/reward_per_step"] = float(np.mean([e[4] for e in eps]))
            causes = [e[3] for e in eps if e[1]]
            for c in set(causes):
                m[f"train/fall_cause/{c}"] = causes.count(c) / len(eps)
            total_s = float(secs.sum()) or 1.0
            for sc in {e[2] for e in eps}:
                sub = [e for e in eps if e[2] == sc]
                m[f"train/scenario/{sc}/fall_rate"] = float(np.mean([e[1] for e in sub]))
                m[f"train/scenario/{sc}/ep_len_s"] = float(np.mean([e[0] for e in sub]))
                # The share of training this scenario actually got. Compare it with the mix:
                # on 10/8 a 40% weight came to 6.8% of steps and nothing said so.
                m[f"train/scenario/{sc}/step_share"] = sum(e[0] for e in sub) / total_s
        m["train/episodes_total"] = self._seen_eps

        means = {k: s / n for k, (s, n) in self._terms.items() if n}
        total_abs = sum(abs(v) for v in means.values()) or 1.0
        for k, v in means.items():
            m[f"reward/{k}"] = v
            m[f"reward/share/{k}"] = abs(v) / total_abs
        self._terms.clear()

        ppo = {}
        for k, v in getattr(self.model.logger, "name_to_value", {}).items():
            if k.startswith("train/") and isinstance(v, (int, float, np.floating, np.integer)):
                ppo[k[6:]] = float(v)
        ppo["ent_coef"] = float(self.model.ent_coef) if not callable(self.model.ent_coef) else float("nan")
        if hasattr(self.model.policy, "log_std"):
            ls = self.model.policy.log_std.detach().cpu().numpy()
            ppo["std_mean"] = float(np.exp(ls).mean())
            ppo["std_at_floor"] = float(np.mean(ls <= self.log_std_bounds[0] + 1e-3))
        for k, v in ppo.items():
            m[f"ppo/{k}"] = v

        warnings = []
        for key, test, msg in HEALTH_RULES:
            tripped = bool(ppo) and "approx_kl" in ppo and test(ppo)
            m[f"health/{key}"] = float(tripped)
            if tripped:
                warnings.append(msg.format(**ppo))
        now = time.time()
        m["perf/fps"] = (self.num_timesteps - self._n_last) / max(1e-6, now - self._t_last)
        self._t_last, self._n_last = now, self.num_timesteps
        self.sink.log(m, self.num_timesteps)

        if self.verbose:
            line = (f"[{self.num_timesteps / 1e6:7.2f}M] "
                    + (f"episodes last {m['train/ep_len_s']:5.1f} s, {100 * m['train/fall_rate']:3.0f}% "
                       f"end in a fall | " if eps else "no finished episodes yet | ")
                    + (f"kl {ppo.get('approx_kl', float('nan')):.4f} clip {ppo.get('clip_fraction', float('nan')):.2f} "
                       f"ev {ppo.get('explained_variance', float('nan')):.2f} std {ppo.get('std_mean', float('nan')):.3f} | "
                       if "approx_kl" in ppo else "")
                    + f"{m['perf/fps']:.0f} steps/s")
            top = sorted(means.items(), key=lambda kv: -abs(kv[1]))[:3]
            if top:
                line += " | reward mostly " + ", ".join(f"{k} {v:+.1f}" for k, v in top)
            shares = {k.split("/")[2]: v for k, v in m.items() if k.endswith("/step_share")}
            if len(shares) > 1:
                line += " | steps: " + " ".join(f"{k} {100 * v:.0f}%" for k, v in
                                               sorted(shares.items(), key=lambda kv: -kv[1]))
            print(line + ("" if warnings else "  [ok]"))
            for w in warnings:
                print(f"    WARNING {w}")
