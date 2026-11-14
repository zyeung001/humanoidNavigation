#!/usr/bin/env python3
"""Score a real_walking policy against its reference gait alone, on the same held-out episodes.

Each episode is one seed of the run's own training randomisation (plant, sensors, lag, start
offsets), `--seconds` long. Both the policy and "reference alone" (zero action: the gait plus the
feedback prior, no network) run the identical episodes, so the difference is paired.

Reported per controller: share of episodes that stay up for the whole test, mean distance walked
(COM, forward), mean forward speed while up, and the fall times.

  python scripts/eval_walking.py --run runs/<name>                 # final/ (or best/ if present)
  python scripts/eval_walking.py --config config/experiments/walk_step_to.yaml   # reference only
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environments.real_walking_env import make_real_walking_env  # noqa: E402


def run_episodes(cfg, policy, seeds, seconds):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_real_walking_env(config=dict(cfg, debug_print_every=0))
    out = []
    for s in seeds:
        np.random.seed(s)
        obs, _ = env.reset(seed=s)
        d = env.env.unwrapped.data
        x0 = float(d.subtree_com[0][0])
        fell = None
        steps = int(round(seconds / env._control_dt))
        for k in range(steps):
            obs, _, term, trunc, _ = env.step(policy(obs))
            if term:
                fell = (k + 1) * env._control_dt
                break
            if trunc:
                break
        t_up = fell if fell is not None else seconds
        dx = float(d.subtree_com[0][0]) - x0
        out.append(dict(seed=s, fell_s=fell, dx_m=dx, speed_mps=dx / max(t_up, 1e-6)))
    return out


def summary(eps):
    up = np.array([e["fell_s"] is None for e in eps])
    dx = np.array([e["dx_m"] for e in eps])
    return dict(survived=f"{int(up.sum())}/{len(eps)}", survive_rate=float(up.mean()),
                mean_dx_cm=round(100 * float(dx.mean()), 1),
                mean_speed_cm_s=round(100 * float(np.mean([e["speed_mps"] for e in eps])), 2),
                falls_s=[round(e["fell_s"], 1) for e in eps if e["fell_s"] is not None])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="runs/<name>: its config.yaml and final/ (or --which best)")
    g.add_argument("--config", help="experiment yaml: score the reference alone")
    p.add_argument("--which", default="final", choices=["final", "best"])
    p.add_argument("--seed", type=int, default=777, help="first held-out seed (training uses others)")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--out", help="write the per-episode results here (json)")
    a = p.parse_args()

    if a.run:
        run = Path(a.run)
        cfg = yaml.safe_load(open(run / "config.yaml"))["standing"]
    else:
        from src.training.experiment import load_experiment
        cfg = load_experiment(a.config).task_cfg("standing")
    seeds = list(range(a.seed * 1000, a.seed * 1000 + a.episodes))
    zero = np.zeros(17, dtype=np.float32)
    results = {"reference alone": run_episodes(cfg, lambda obs: zero, seeds, a.seconds)}
    if a.run:
        from src.robustness.policies import SB3Policy
        d = run / a.which
        pol = SB3Policy.load(d / "model.zip", d / "vecnorm.pkl", label=f"{run.name}@{a.which}")
        results[pol.label] = run_episodes(cfg, pol.act, seeds, a.seconds)

    for name, eps in results.items():
        print(f"{name:>32}: {json.dumps(summary(eps))}")
    if a.run:
        ref, pol = list(results.values())
        both = [(r["fell_s"] is None, q["fell_s"] is None) for r, q in zip(ref, pol)]
        print(f"paired: policy kept up {sum(q and not r for r, q in both)} episodes the reference "
              f"lost, lost {sum(r and not q for r, q in both)} it kept; distance policy - reference "
              f"{100 * np.mean([q['dx_m'] - r['dx_m'] for r, q in zip(ref, pol)]):+.1f} cm")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
