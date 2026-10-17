#!/usr/bin/env python3
"""Score a standing policy on the robustness suite and record the result.

WHY. Until 9/19 nothing here could tell a good run from a bad one: the kick gate is binary
and every model passed it. The 9/19 scoreboard fixed the seed but tested one quiet episode
and one kick level in an easier world than the robot's (sensor noise and servo variation
off), and its basin column read 2.0 for every model ever scored. On 10/7 the suite below
found that the deployed 195M policy scores only ~4 points above COMMANDING THE POSE WITH NO
FEEDBACK AT ALL -- the robot mostly stands because the pose stands, not because the policy
balances it. That is the number this script exists to move.

WHAT. src/robustness/scenarios.py SUITE: starts (deploy's own, leaning, rotating, off-pose),
torso pushes, velocity kicks, and sensor and plant faults under load, each at fixed seeds so
every model meets identical episodes. See src/robustness/suite.py for how the score is built
and how big a difference is real.

    # a trained model (what training saves)
    python scripts/eval_standing.py --model models/final_real_standing_stand.zip \\
        --vecnorm models/vecnorm_real_standing_stand.pkl --config config/real_humanoid_stand.yaml \\
        --label stand-195M
    # exactly what the robot runs
    python scripts/eval_standing.py --npz models/real_standing_stand_195M.npz \\
        --config config/real_humanoid_stand.yaml
    # a training run's best checkpoint (config comes from the run)
    python scripts/eval_standing.py --run runs/<run-name>
    # the no-learning reference
    python scripts/eval_standing.py --hold-pose --config config/real_humanoid_stand.yaml

Each result is saved to models/standing/evals/ (full detail, per episode) and one summary row
is appended to models/standing/robustness.csv. Rank them with scripts/compare_models.py.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.robustness import policies, suite  # noqa: E402
from src.robustness.scenarios import suite_scenarios  # noqa: E402
from src.training.experiment import load_experiment  # noqa: E402

EVALS = PROJ / "models" / "standing" / "evals"
TABLE = PROJ / "models" / "standing" / "robustness.csv"


def record(result: suite.SuiteResult, evals_dir=EVALS, table=TABLE) -> Path:
    """Save the full result as JSON and append its summary row to the CSV."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in result.label)
    # The suite version is in the name: the same weights scored on a changed suite or plant are
    # a new result, not a replacement for the old one.
    path = result.save(Path(evals_dir) / f"{safe}__{result.fingerprint}__{result.suite_version}.json")
    row = {"when": result.when, "label": result.label, "fingerprint": result.fingerprint,
           "suite": result.suite_version, "complete": int(result.complete), "seed": result.seed,
           "episodes": result.episodes_per_scenario, "git": result.git, "config": result.config,
           "score": round(result.score, 2), "score_se": round(result.score_se, 2),
           "survival_pct": round(result.survival_pct, 1)}
    row.update({f"fam_{k}": round(v, 1) for k, v in result.families().items()})
    row.update({s.name: round(s.standing_pct, 1) for s in result.scenarios})
    table = Path(table)
    rows = list(csv.DictReader(open(table))) if table.exists() else []
    fields = list(dict.fromkeys([*(rows[0].keys() if rows else []), *row.keys()]))
    rows.append(row)
    with open(table, "w", newline="") as f:     # rewritten so a new column never misaligns
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def rank_line(result: suite.SuiteResult, table=TABLE) -> str:
    """Where this result sits among everything recorded on the same suite."""
    if not Path(table).exists() or not result.complete:
        return ""
    rows = [r for r in csv.DictReader(open(table))
            if r.get("suite") == result.suite_version and r.get("complete") == "1"
            and r.get("fingerprint") != result.fingerprint]
    best = {}
    for r in rows:
        if r["label"] not in best or float(r["score"]) > float(best[r["label"]]["score"]):
            best[r["label"]] = r
    others = sorted(best.values(), key=lambda r: -float(r["score"]))
    pos = 1 + sum(float(r["score"]) > round(result.score, 2) for r in others)
    top = others[0] if others else None
    return (f"rank {pos} of {len(others) + 1} recorded models on {result.suite_version}"
            + (f"  (best of the others: {top['label']} {float(top['score']):.1f})" if top else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--model", help="SB3 .zip (needs --vecnorm)")
    what.add_argument("--npz", help="exported policy bundle, run by the robot's numpy_policy.py")
    what.add_argument("--run", help="a runs/<name> directory: scores its best/ checkpoint")
    what.add_argument("--hold-pose", nargs="?", const="baseline", choices=["baseline", "straight"],
                      help="no policy, no feedback: command the residual baseline (default) or "
                           "the straight pose (all zeros, which is also what an UNTRAINED network "
                           "outputs) every step. The bar a policy must clear to be balancing.")
    p.add_argument("--vecnorm")
    p.add_argument("--config", help="training config or experiment file (sets the plant + baseline)")
    p.add_argument("--checkpoint", default="best", help="with --run: best | final | a step dir")
    p.add_argument("--label")
    p.add_argument("--episodes", type=int, default=12,
                   help="per scenario. Keep it the same across models you compare.")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=12345, help="keep identical across models")
    p.add_argument("--workers", type=int, default=suite.default_workers())
    p.add_argument("--scenarios", help="comma list; partial runs are recorded as partial")
    p.add_argument("--families", help="comma list, e.g. push,kick")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    if args.run:
        run = Path(args.run)
        ck = run / args.checkpoint
        args.model, args.vecnorm = str(ck / "model.zip"), str(ck / "vecnorm.pkl")
        args.config = args.config or str(run / "config.yaml")
        args.label = args.label or f"{run.name}/{args.checkpoint}"
    if not args.config:
        p.error("--config is required (it defines the robot, the baseline and the clamp)")
    cfg = load_experiment(args.config).task_cfg("standing")

    if args.hold_pose == "straight":
        pol = policies.HoldPosePolicy(np.zeros(len(cfg["residual_baseline"])), label="hold-straight")
    elif args.hold_pose:
        pol = policies.HoldPosePolicy(cfg["residual_baseline"], label="hold-baseline")
    elif args.npz:
        pol = policies.NpzPolicy(args.npz, label=args.label)
    else:
        if not args.vecnorm:
            p.error("--model needs --vecnorm: the observation statistics are part of the policy")
        pol = policies.SB3Policy.load(args.model, args.vecnorm, label=args.label)

    names = args.scenarios.split(",") if args.scenarios else None
    fams = args.families.split(",") if args.families else None
    scen = suite_scenarios(names=names, families=fams) if (names or fams) else None
    print(f"Scoring {pol.label} [{pol.fingerprint}] with {args.workers} worker(s)...")
    result = suite.evaluate(pol, cfg, scen, episodes=args.episodes, seconds=args.seconds,
                            seed=args.seed, config_path=args.config, workers=args.workers)
    print()
    print(suite.format_result(result))
    if not args.no_save:
        path = record(result)
        print(f"\nsaved {path.relative_to(PROJ)}; row appended to {TABLE.relative_to(PROJ)}")
        line = rank_line(result)
        if line:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
