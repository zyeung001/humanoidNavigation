"""Run a controller through the scenario suite and score it.

THE HEADLINE NUMBER is `score`: for each scenario, the average share of the test the robot
spent upright (100 = never fell), then the plain mean over scenarios. It is computed on fixed
seeds, so the same model always gets the same score and two models meet identical conditions;
a difference between two scores is the models, not luck. Training reward cannot do this --
it changes whenever a reward weight does, and it ran to ~200,000 per episode while saying
nothing about falling.

Why share-of-time rather than survived/failed: a binary pass on 8 episodes moves in steps of
12.5% and cannot see a model that now falls at 12 s instead of 3 s. The survival count is
still reported beside it.

How big a difference is real: `score_se` is the standard error from the episode spread. Two
scores closer than ~2 combined SEs could be noise, and compare_models.py also counts, per
paired episode, which model stood longer -- the more sensitive test when seeds are shared.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .scenarios import SUITE, SUITE_VERSION, Scenario, suite_scenarios
from .wrapper import ScenarioWrapper

PROJ = Path(__file__).resolve().parents[2]

CALM_RATE = 0.30     # rad/s: |ang_vel| below this counts as settled ...
CALM_STEPS = 10      # ... held for a quarter second (40 Hz)


def episode_seed(seed: int, index: int) -> int:
    """Depends only on (seed, index), not on the model and not on the scenario. So every model
    meets the same episode i, and within one model every scenario's episode i shares its
    starting draw, pushes and sensor noise -- push_5N_gyro differs from push_5N ONLY by the
    gyro error. Comparisons across models and across scenarios are both paired."""
    h = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJ, capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJ,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "?"


# ------------------------------------------------------------------------------ results

@dataclass
class EpisodeResult:
    seed: int
    steps: int              # control steps upright
    horizon: int
    fell: bool
    lean_deg: float         # median lean from vertical over the episode
    sway_deg: float         # spread of that lean
    jitter: float           # rms per-step change of the RAW action (rad): how twitchy
    excursion2: float       # mean squared distance of the raw action outside the clamp box
    recover_s: List[float] = field(default_factory=list)   # per disturbance; nan = never settled
    dt: float = 0.025       # control period, s

    @property
    def upright_frac(self) -> float:
        return self.steps / self.horizon


@dataclass
class ScenarioResult:
    name: str
    family: str
    about: str
    episodes: List[EpisodeResult]

    @property
    def standing_pct(self) -> float:
        return 100.0 * float(np.mean([e.upright_frac for e in self.episodes]))

    @property
    def standing_se(self) -> float:
        n = len(self.episodes)
        return 100.0 * float(np.std([e.upright_frac for e in self.episodes], ddof=1) / np.sqrt(n)) \
            if n > 1 else 0.0

    @property
    def survived(self) -> int:
        return sum(not e.fell for e in self.episodes)

    @property
    def fall_s(self) -> float:
        """Median time to fall among the episodes that fell."""
        t = [e.steps * e.dt for e in self.episodes if e.fell]
        return float(np.median(t)) if t else float("nan")

    @property
    def recover_s(self) -> float:
        r = [x for e in self.episodes for x in e.recover_s if np.isfinite(x)]
        return float(np.median(r)) if r else float("nan")

    def mean(self, attr: str) -> float:
        v = [getattr(e, attr) for e in self.episodes if np.isfinite(getattr(e, attr))]
        return float(np.mean(v)) if v else float("nan")


@dataclass
class SuiteResult:
    label: str
    fingerprint: str
    suite_version: str
    seed: int
    seconds: float
    episodes_per_scenario: int
    complete: bool               # every SUITE scenario was run -- only then is `score` THE score
    config: str
    git: str
    when: str
    step: Optional[int]          # training step, when evaluated during training
    wall_s: float
    scenarios: List[ScenarioResult]

    # -- headline
    @property
    def score(self) -> float:
        return float(np.mean([s.standing_pct for s in self.scenarios]))

    @property
    def score_se(self) -> float:
        k = len(self.scenarios)
        return float(np.sqrt(sum(s.standing_se ** 2 for s in self.scenarios)) / k)

    @property
    def survival_pct(self) -> float:
        eps = [e for s in self.scenarios for e in s.episodes]
        return 100.0 * sum(not e.fell for e in eps) / len(eps)

    def families(self) -> Dict[str, float]:
        out: Dict[str, List[float]] = {}
        for s in self.scenarios:
            out.setdefault(s.family, []).append(s.standing_pct)
        return {k: float(np.mean(v)) for k, v in out.items()}

    def by_name(self) -> Dict[str, ScenarioResult]:
        return {s.name: s for s in self.scenarios}

    def worst(self, k: int = 3) -> List[ScenarioResult]:
        return sorted(self.scenarios, key=lambda s: s.standing_pct)[:k]

    def metrics(self, prefix: str = "eval/") -> Dict[str, float]:
        """Flat numbers for W&B / JSONL. Names are stable; dashboards key on them."""
        m = {f"{prefix}score": self.score, f"{prefix}score_se": self.score_se,
             f"{prefix}survival_pct": self.survival_pct}
        for fam, v in self.families().items():
            m[f"{prefix}family/{fam}"] = v
        for s in self.scenarios:
            m[f"{prefix}scenario/{s.name}"] = s.standing_pct
        quiet = self.by_name().get("quiet")
        if quiet is not None:
            m[f"{prefix}quality/lean_deg"] = quiet.mean("lean_deg")
            m[f"{prefix}quality/sway_deg"] = quiet.mean("sway_deg")
            m[f"{prefix}quality/jitter"] = quiet.mean("jitter")
            m[f"{prefix}quality/excursion2"] = quiet.mean("excursion2")
        rec = [s.recover_s for s in self.scenarios if s.family in ("push", "kick")]
        rec = [r for r in rec if np.isfinite(r)]
        if rec:
            m[f"{prefix}quality/recover_s"] = float(np.median(rec))
        return m

    # -- persistence
    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"], d["score_se"] = self.score, self.score_se
        d["survival_pct"], d["families"] = self.survival_pct, self.families()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SuiteResult":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d["scenarios"] = [ScenarioResult(**{**s, "episodes": [EpisodeResult(**e) for e in s["episodes"]]})
                          for s in d["scenarios"]]
        return cls(**d)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, default=float))
        return path

    @classmethod
    def load(cls, path) -> "SuiteResult":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ------------------------------------------------------------------------------ running

def make_eval_env(env_cfg: dict, seconds: float):
    """A StandingEnv in a ScenarioWrapper, set up for evaluation."""
    from src.environments import make_standing_env  # noqa: PLC0415
    cfg = dict(env_cfg)
    steps = int(round(seconds / 0.025))
    cfg.update(max_episode_steps=steps + 5, debug_print_every=0)
    return ScenarioWrapper(make_standing_env(config=cfg), scenarios={s.name: s for s in SUITE})


def _lean_deg(qpos) -> float:
    _, x, y, _ = qpos[3:7]
    return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


def _recoveries(rates: np.ndarray, starts: List[int], end: int, dt: float) -> List[float]:
    out = []
    for i, s in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else end
        calm, rec = 0, float("nan")
        for k in range(s + 1, stop):
            calm = calm + 1 if rates[k] < CALM_RATE else 0
            if calm >= CALM_STEPS:
                rec = (k - s - CALM_STEPS + 1) * dt
                break
        out.append(rec)
    return out


def run_episode(policy, wenv: ScenarioWrapper, scenario: Scenario, seed: int, steps: int,
                stratum=None) -> EpisodeResult:
    base = wenv.base
    data = base.env.unwrapped.data
    np.random.seed(seed)      # StandingEnv draws its noise and randomisation from the global RNG
    obs, _ = wenv.reset(seed=seed, options={"scenario": scenario, "stratum": stratum})
    box = None
    clamp = np.asarray(base.residual_clamp, dtype=np.float64)
    if np.any(clamp > 0):
        b, lo_c, hi_c = base.episode_baseline, base.env.action_space.low, base.env.action_space.high
        box = (np.maximum(b - clamp, lo_c), np.minimum(b + clamp, hi_c))
    leans, rates = [], np.zeros(steps + 1)
    exc, prev, d_act = [], None, []
    fell, t = False, 0
    for t in range(steps):
        a = np.asarray(policy.act(obs), dtype=np.float32)
        if prev is not None:
            d_act.append(float(np.mean(np.square(a - prev))))
        prev = a
        if box is not None:
            e = np.maximum(box[0] - a, 0.0) + np.maximum(a - box[1], 0.0)
            exc.append(float(np.sum(e * e)))
        obs, _, terminated, truncated, _ = wenv.step(a)
        leans.append(_lean_deg(data.qpos))
        rates[t + 1] = float(np.linalg.norm(data.qvel[3:6]))
        if terminated:
            fell = True
            break
    n = t + 1
    upright = leans[:-1] if fell else leans
    return EpisodeResult(
        seed=seed, steps=n if not fell else t, horizon=steps, fell=fell,
        lean_deg=float(np.median(upright)) if upright else float("nan"),
        sway_deg=float(np.std(upright)) if upright else float("nan"),
        jitter=float(np.sqrt(np.mean(d_act))) if d_act else float("nan"),
        excursion2=float(np.mean(exc)) if exc else float("nan"),
        recover_s=_recoveries(rates, wenv.disturbance_steps, n, base._control_dt),
        dt=base._control_dt,
    )


_WORKER: dict = {}


def _worker_init(env_cfg, seconds, recipe):
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415
    sys.stdout = open(os.devnull, "w")       # the env prints its whole config on construction
    import torch  # noqa: PLC0415
    torch.set_num_threads(1)
    from .policies import rebuild  # noqa: PLC0415
    _WORKER["wenv"] = make_eval_env(env_cfg, seconds)
    _WORKER["policy"] = rebuild(recipe)
    _WORKER["steps"] = int(round(seconds / _WORKER["wenv"].base._control_dt))


def _worker_run(task):
    scenario, seed, stratum = task
    return run_episode(_WORKER["policy"], _WORKER["wenv"], scenario, seed, _WORKER["steps"], stratum)


def default_workers() -> int:
    import os  # noqa: PLC0415
    return max(1, (os.cpu_count() or 2) - 1)


def evaluate(policy, env_cfg: dict, scenarios: Optional[List[Scenario]] = None, episodes: int = 8,
             seconds: float = 15.0, seed: int = 12345, config_path: str = "", step: Optional[int] = None,
             workers: int = 1, progress=print) -> SuiteResult:
    """Score `policy` on `scenarios` (default: the whole suite).

    workers > 1 runs episodes in parallel processes; the policy must then carry a `.recipe`
    (anything loaded from a file does). Every episode seeds itself, so the result is identical
    for any number of workers.

    Restores numpy's global RNG afterwards, so evaluating in the middle of training does not
    change what training draws next.
    """
    scenarios = list(scenarios) if scenarios is not None else suite_scenarios()
    complete = {s.name for s in scenarios} >= {s.name for s in SUITE}
    tasks = [(s, episode_seed(seed, i), (i, episodes)) for s in scenarios for i in range(episodes)]
    rng_state = np.random.get_state()
    t0 = time.time()
    try:
        if workers > 1 and getattr(policy, "recipe", None) is not None:
            import multiprocessing as mp  # noqa: PLC0415
            from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                     mp_context=mp.get_context("spawn"), initializer=_worker_init,
                                     initargs=(env_cfg, seconds, policy.recipe)) as pool:
                flat = list(pool.map(_worker_run, tasks, chunksize=1))
        else:
            wenv = make_eval_env(env_cfg, seconds)
            steps = int(round(seconds / wenv.base._control_dt))
            flat = [run_episode(policy, wenv, s, sd, steps, st) for s, sd, st in tasks]
    finally:
        np.random.set_state(rng_state)
    results = []
    for k, s in enumerate(scenarios):
        eps = flat[k * episodes:(k + 1) * episodes]
        r = ScenarioResult(s.name, s.family, s.about, eps)
        results.append(r)
        if progress:
            progress(f"  {s.name:<16s} {r.standing_pct:5.1f}% upright  {r.survived}/{len(eps)} never fell")
    return SuiteResult(
        label=policy.label, fingerprint=policy.fingerprint, suite_version=SUITE_VERSION, seed=seed,
        seconds=seconds, episodes_per_scenario=episodes, complete=complete, config=str(config_path),
        git=git_sha(), when=time.strftime("%Y-%m-%d %H:%M"), step=step, wall_s=time.time() - t0,
        scenarios=results)


# ------------------------------------------------------------------------------ display

def _bar(pct: float, width: int = 20) -> str:
    n = int(round(width * max(0.0, min(100.0, pct)) / 100.0))
    return "#" * n + "." * (width - n)


def format_result(r: SuiteResult, previous: Optional[SuiteResult] = None) -> str:
    """One screen: the score, what got better or worse since last time, and the worst cases."""
    lines = []
    when = f" @ {r.step / 1e6:.1f}M steps" if r.step is not None else ""
    lines.append(f"=== ROBUSTNESS{when}: {r.label}  [{r.fingerprint}]  "
                 f"({r.suite_version}, {len(r.scenarios)} scenarios x {r.episodes_per_scenario} "
                 f"episodes x {r.seconds:.0f} s, {r.wall_s:.0f} s wall)")
    head = f"SCORE {r.score:5.1f} +- {r.score_se:.1f}   (% of test time upright, 100 = never fell)"
    if not r.complete:
        head += "   ** PARTIAL SUITE: not comparable with full-suite scores **"
    if previous is not None and previous.suite_version == r.suite_version:
        d = r.score - previous.score
        head += f"   {'up' if d >= 0 else 'DOWN'} {d:+.1f} since last eval"
    lines.append(head)
    prev = previous.by_name() if previous is not None else {}
    lines.append(f"  {'scenario':<16s} {'upright':>7s}  {'':20s} {'never fell':>10s} "
                 f"{'falls at':>8s} {'settles':>7s} {'change':>7s}")
    for s in r.scenarios:
        p = prev.get(s.name)
        delta = f"{s.standing_pct - p.standing_pct:+6.1f}" if p is not None else ""
        fall = f"{s.fall_s:6.1f} s" if np.isfinite(s.fall_s) else ""
        rec = f"{s.recover_s:5.2f} s" if np.isfinite(s.recover_s) else ""
        lines.append(f"  {s.name:<16s} {s.standing_pct:6.1f}%  {_bar(s.standing_pct)} "
                     f"{s.survived:>4d}/{len(s.episodes):<5d} {fall:>8s} {rec:>7s} {delta:>7s}")
    fams = "  ".join(f"{k} {v:.0f}" for k, v in r.families().items())
    lines.append(f"  by family: {fams}")
    worst = ", ".join(f"{s.name} ({s.standing_pct:.0f}%)" for s in r.worst(3))
    lines.append(f"  weakest: {worst}")
    q = r.by_name().get("quiet")
    if q is not None:
        lines.append(f"  quiet stand: lean {q.mean('lean_deg'):.2f} deg, sway {q.mean('sway_deg'):.2f} deg, "
                     f"jitter {q.mean('jitter'):.4f} rad/step, clamp excursion^2 {q.mean('excursion2'):.4f}")
    return "\n".join(lines)
