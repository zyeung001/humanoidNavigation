"""Experiments: a base config plus the few keys that change, and where a run's outputs go.

WHY. config/ holds fifteen real_humanoid_*.yaml files, each a full copy of the last with a
few values changed and a paragraph of history on top. To learn what a run changed you diff
two 300-line files; to learn whether it helped you dig through scoreboard rows that all name
the same model path. An experiment file says only what is different:

    experiment:
      name: stand_history8
      base: config/real_humanoid_stand.yaml
      hypothesis: 4 frames (75 ms) is shorter than the servo dead time; 8 frames lets the
                  policy see its own commands land.
    overrides:
      standing:
        obs_history: 8

and that dict of overrides is logged with the run as `changes.standing.obs_history = 8`, so
the W&B runs table shows the variable each run changed beside the score it got.

A plain config (no `experiment:` block) still loads exactly as before.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PROJ = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, over: dict) -> dict:
    """`over` wins. Dicts merge key by key; lists and scalars are replaced whole."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def flatten(d: dict, prefix: str = "") -> Dict[str, object]:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _resolve(path: str, relative_to: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for root in (PROJ, relative_to):
        if (root / p).exists():
            return root / p
    raise FileNotFoundError(f"cannot find {path} (looked under {PROJ} and {relative_to})")


@dataclass
class Experiment:
    name: str
    path: str
    config: dict                       # fully resolved: the dict training actually reads
    base: Optional[str] = None
    description: str = ""
    hypothesis: str = ""
    tags: List[str] = field(default_factory=list)
    changes: dict = field(default_factory=dict)   # this file's overrides, as written
    lineage: List[str] = field(default_factory=list)   # base chain, root first
    resume: dict = field(default_factory=dict)    # {model, vecnorm}: what this run continues

    def task_cfg(self, task: str = "standing") -> dict:
        """The task block (standing / turning), as train_standing.py reads it."""
        block = self.config.get(task) or (self.config.get("turning") if task == "standing" else None)
        if block is None:
            raise KeyError(f"{self.path} has no '{task}' block")
        return copy.deepcopy(block)

    def changes_flat(self) -> Dict[str, object]:
        return flatten(self.changes)

    def run_name(self, stamp: Optional[str] = None) -> str:
        return f"{self.name}-{stamp or time.strftime('%m%d-%H%M')}"

    def resolved_yaml(self, extra_meta: Optional[dict] = None, task: Optional[str] = None,
                      task_block: Optional[dict] = None) -> str:
        """The config a run used, whole, with what it was built from on top. Pass the task
        block as training finally used it (defaults filled, paths redirected) so the file is
        what ran, not what was asked for."""
        # `resolved: true` marks this as a finished config, not an experiment to build: it
        # still names its base for the reader, but loading it must not merge again.
        meta = {"resolved": True, "name": self.name, "source": self.path, "base": self.base,
                "lineage": self.lineage, "description": self.description,
                "hypothesis": self.hypothesis, "tags": self.tags, "changes": self.changes,
                "resume": self.resume, **(extra_meta or {})}
        body = copy.deepcopy(self.config)
        if task is not None and task_block is not None:
            body[task] = task_block
        return yaml.safe_dump({"experiment": meta, **body}, sort_keys=False, width=100)


def load_experiment(path, _seen: Optional[set] = None) -> Experiment:
    """Load a plain config or an experiment file (which may itself build on an experiment)."""
    path = _resolve(str(path), Path.cwd())
    _seen = set(_seen or ())
    if path in _seen:
        raise ValueError(f"experiment base chain loops back to {path}")
    _seen.add(path)
    raw = yaml.safe_load(open(path, encoding="utf-8")) or {}
    meta = raw.get("experiment")
    rel = _rel(path)
    if not isinstance(meta, dict) or "base" not in meta or meta.get("resolved"):
        # A plain config. A resolved config written by a run also lands here: its experiment
        # block is metadata (no base to build on), and the rest is the config itself.
        body = {k: v for k, v in raw.items() if k != "experiment"}
        info = meta if isinstance(meta, dict) else {}
        # No `resume` here: a run's saved config records what that run resumed FROM, and
        # quietly resuming from it again would restart from the old model, not the run's own.
        return Experiment(name=info.get("name", path.stem), path=rel, config=body,
                          description=info.get("description", ""),
                          hypothesis=info.get("hypothesis", ""), tags=list(info.get("tags", [])),
                          changes=info.get("changes", {}) or {}, lineage=info.get("lineage", []))
    unknown = set(raw) - {"experiment", "overrides"}
    if unknown:
        raise ValueError(f"{rel}: experiment files hold only 'experiment' and 'overrides', "
                         f"not {sorted(unknown)} -- put config keys under overrides")
    parent = load_experiment(_resolve(meta["base"], path.parent), _seen)
    overrides = raw.get("overrides") or {}
    return Experiment(
        name=meta.get("name", path.stem), path=rel, base=parent.path,
        config=deep_merge(parent.config, overrides),
        description=meta.get("description", ""), hypothesis=meta.get("hypothesis", ""),
        tags=list(meta.get("tags", [])),
        changes=deep_merge(parent.changes, overrides) if parent.base else overrides,
        lineage=[*parent.lineage, parent.path] if parent.base else [parent.path],
        resume=meta.get("resume") or parent.resume)    # a chained experiment continues
                                                       # what its base continues, unless told


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJ).as_posix()
    except ValueError:
        return str(path)


@dataclass
class RunDir:
    """Where one training run puts everything, so no run overwrites another's model.

    runs/<name>/config.yaml        the resolved config + what it was built from
                checkpoints/<step>/{model.zip, vecnorm.pkl, eval.json}
                best/{model.zip, vecnorm.pkl, eval.json}   highest robustness score so far
                final/{model.zip, vecnorm.pkl}
                metrics.jsonl  evals.jsonl
    """
    root: Path

    @classmethod
    def create(cls, name: str, base: Path = PROJ / "runs") -> "RunDir":
        root = Path(base) / name
        n = 2
        while root.exists():
            root = Path(base) / f"{name}-{n}"
            n += 1
        root.mkdir(parents=True)
        return cls(root)

    @classmethod
    def open(cls, path) -> "RunDir":
        root = Path(path)
        if not (root / "config.yaml").exists():
            raise FileNotFoundError(f"{root} is not a run directory (no config.yaml)")
        return cls(root)

    def latest_weights(self) -> Optional[Path]:
        """The directory holding the furthest-trained model of this run: a checkpoint, or what
        Ctrl+C (interrupted/) or a crash (crashed/) left behind. Judged by the step count
        stored inside each model.zip, not by folder names or file times."""
        import json  # noqa: PLC0415
        import zipfile  # noqa: PLC0415
        best, best_steps = None, -1
        for d in [*self.root.glob("checkpoints/*"), self.root / "interrupted", self.root / "crashed"]:
            z = d / "model.zip"
            if not (z.exists() and (d / "vecnorm.pkl").exists()):
                continue
            try:
                steps = int(json.loads(zipfile.ZipFile(z).read("data"))["num_timesteps"])
            except (KeyError, ValueError, zipfile.BadZipFile):
                continue
            if steps > best_steps:
                best, best_steps = d, steps
        return best

    @property
    def name(self) -> str:
        return self.root.name

    def checkpoint(self, step: int) -> Path:
        return self.root / "checkpoints" / f"{step / 1e6:08.2f}M"

    @property
    def best(self) -> Path:
        return self.root / "best"

    @property
    def final(self) -> Path:
        return self.root / "final"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def evals(self) -> Path:
        return self.root / "evals.jsonl"
