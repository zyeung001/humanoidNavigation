#!/usr/bin/env python3
"""Rank robustness results side by side: which model is best, and where each one breaks.

Reads saved results only (it never runs the simulator), so it is instant. Results come from
scripts/eval_standing.py (models/standing/evals/) and from training runs
(runs/<name>/best/eval.json and runs/<name>/checkpoints/*/eval.json).

    python scripts/compare_models.py                      # everything recorded, best first
    python scripts/compare_models.py --ref stand-195M     # deltas against a chosen model
    python scripts/compare_models.py runs/stand_*         # the best checkpoint of each run
    python scripts/compare_models.py --run-history runs/stand_pushes-1008-0930   # one run over time

HOW TO READ A DIFFERENCE. Every result scores the same episodes (same seeds, same pushes),
so two models are compared episode by episode: `vs ref` is the mean paired difference in %
of test time upright, +- its standard error, and `won/lost` counts the episodes where this
model stood longer / shorter than the reference. A difference smaller than ~2 SE, or with
won and lost close together, is not a difference. Cells marked + or - in the scenario table
are ones where the paired difference IS beyond 2 SE.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.robustness.suite import SuiteResult  # noqa: E402

EVALS = PROJ / "models" / "standing" / "evals"


def collect(paths):
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir() and (p / "best" / "eval.json").exists():
            out.append(p / "best" / "eval.json")
        elif p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        elif p.suffix == ".json":
            out.append(p)
    return out


def paired(a: SuiteResult, b: SuiteResult):
    """Mean and SE of a-b in % upright over every shared episode, and episodes won/lost."""
    if (a.seed, a.episodes_per_scenario, a.seconds) != (b.seed, b.episodes_per_scenario, b.seconds):
        return None
    bn = b.by_name()
    d = []
    for s in a.scenarios:
        o = bn.get(s.name)
        if o is None:
            continue
        d.extend(100.0 * (x.upright_frac - y.upright_frac) for x, y in zip(s.episodes, o.episodes))
    if not d:
        return None
    d = np.array(d)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), int((d > 0).sum()), int((d < 0).sum())


def scenario_delta(sa, sb):
    d = np.array([100.0 * (x.upright_frac - y.upright_frac) for x, y in zip(sa.episodes, sb.episodes)])
    if len(d) < 2:
        return 0.0, 0.0
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", nargs="*", help="result .json files, run dirs, or eval dirs")
    p.add_argument("--ref", help="label (or substring) to compare everything against "
                                 "(default: the hold-pose result if recorded, else the top one)")
    p.add_argument("--run-history", help="a run dir: every checkpoint's score, in order")
    p.add_argument("--top", type=int, default=6, help="models shown in the scenario table")
    args = p.parse_args()

    if args.run_history:
        files = sorted(Path(args.run_history).glob("checkpoints/*/eval.json"))
        res = [SuiteResult.load(f) for f in files]
        if (Path(args.run_history) / "best" / "eval.json").exists():
            best = SuiteResult.load(Path(args.run_history) / "best" / "eval.json")
        else:
            best = None
        print(f"{Path(args.run_history).name}: {len(res)} kept checkpoints "
              f"(older ones are pruned; evals.jsonl has every one)")
        for r in sorted(res, key=lambda r: r.step or 0):
            mark = "  <- best" if best is not None and best.fingerprint == r.fingerprint else ""
            print(f"  {(r.step or 0) / 1e6:8.2f}M  score {r.score:5.1f} +- {r.score_se:.1f}  "
                  f"weakest {r.worst(1)[0].name}{mark}")
        return 0

    files = collect(args.paths or [EVALS])
    if not files:
        print("No results found. Score a model first: scripts/eval_standing.py --help")
        return 1
    results = [SuiteResult.load(f) for f in files]
    # Latest result per (label, fingerprint): re-scoring the same weights replaces, not adds.
    latest = {}
    for r in results:
        k = (r.label, r.fingerprint)
        if k not in latest or r.when > latest[k].when:
            latest[k] = r
    results = list(latest.values())
    versions = {r.suite_version for r in results}
    if len(versions) > 1:
        newest = max(versions)
        dropped = [r.label for r in results if r.suite_version != newest]
        print(f"NOTE: scores from different suite versions are not comparable; showing "
              f"{newest} only (dropped: {', '.join(dropped)})")
        results = [r for r in results if r.suite_version == newest]
    full = [r for r in results if r.complete]
    partial = [r for r in results if not r.complete]
    results = sorted(full, key=lambda r: -r.score) + partial

    ref = None
    if args.ref:
        ref = next((r for r in results if args.ref in r.label), None)
        if ref is None:
            print(f"No result labelled like {args.ref!r}")
            return 1
    else:
        # Default: the best NO-FEEDBACK pose on record. A policy that does not beat holding a
        # fixed pose is not balancing the robot, whatever its training reward says.
        holds = [r for r in results if r.fingerprint.startswith("hold-") and r.complete]
        ref = max(holds, key=lambda r: r.score) if holds else results[0]

    print(f"\nRANKING ({results[0].suite_version}; reference = {ref.label})")
    print(f"  {'#':>2s}  {'model':<34s} {'score':>11s}  {'vs ref (paired)':>17s} {'won/lost':>9s}  "
          f"{'never fell':>10s}  weakest")
    for i, r in enumerate(results, 1):
        tag = "" if r.complete else "  (partial suite)"
        pr = paired(r, ref) if r is not ref else None
        vs = (f"{pr[0]:+6.1f} +- {pr[1]:4.1f}" if pr else ("reference" if r is ref else "not paired"))
        wl = f"{pr[2]:>4d}/{pr[3]:<4d}" if pr else ""
        print(f"  {i:>2d}  {r.label[:34]:<34s} {r.score:5.1f} +-{r.score_se:4.1f}  {vs:>17s} {wl:>9s}  "
              f"{r.survival_pct:9.0f}%  {r.worst(1)[0].name}{tag}")

    shown = results[:args.top]
    if ref not in shown:
        shown = shown[:-1] + [ref] if len(shown) == args.top else shown + [ref]
    print("\nSCENARIOS (% of test time upright; + / - = beyond 2 SE of the reference, paired)")
    legend = {f"M{i}": r for i, r in enumerate(shown, 1)}
    print(f"  {'scenario':<18s}" + "".join(f"{k:>9s}" for k in legend))
    names = [s.name for s in shown[0].scenarios]
    for name in names:
        cells = []
        rs = ref.by_name().get(name)
        for r in legend.values():
            s = r.by_name().get(name)
            if s is None:
                cells.append(f"{'':>9s}")
                continue
            mark = " "
            if rs is not None and r is not ref:
                m, se = scenario_delta(s, rs)
                mark = "+" if m > 2 * se and m > 0 else "-" if m < -2 * se and m < 0 else " "
            cells.append(f"{s.standing_pct:7.1f}{mark} ")
        print(f"  {name:<18s}" + "".join(cells))
    fams = list(shown[0].families())
    for fam in fams:
        print(f"  {'[' + fam + ']':<18s}" + "".join(f"{r.families().get(fam, float('nan')):7.1f}  "
                                                   for r in legend.values()))
    print("\n  " + "\n  ".join(f"{k} = {r.label}  [{r.fingerprint}]  {r.when}" for k, r in legend.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
