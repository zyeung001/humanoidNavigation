"""How much robustness does the PLANT allow? Oracle LQR / MPC ceilings and the ankle-servo what-if.

Every result is paired (same seeds, same pushes) against the knee law alone -- the config's
`feedback_prior` under a held baseline pose -- so "+x +- se" is a held-out paired difference in
% of the test upright. Oracles read sim truth; none of this is deployable (src/robustness/oracle.py).

    python scripts/ceiling.py lqr                          # Q/R grid on 12345x12, best on 777x24 + 778x48
    python scripts/ceiling.py mpc --scenarios families     # start/push/kick/plant families
    python scripts/ceiling.py mpc --scenarios band --N 256 --H 40 --knots 6 --sigma 0.15
    python scripts/ceiling.py ankle --k 4.4 --ka 0.3 --kd 0.18 --seed 778 --episodes 48
    python scripts/ceiling.py ankle --band                 # gain grid on the single-push band

Results (11/11/2026, flex plant, standing-v2): scripts/ceiling.md.
"""
import argparse
import copy
import functools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402

from src.robustness import oracle, policies, suite  # noqa: E402
from src.robustness.scenarios import suite_scenarios  # noqa: E402
from src.training.experiment import load_experiment  # noqa: E402

FAMS = ("start", "push", "kick", "plant")   # sensor faults do not touch an oracle: left out

# The paired comparison moves into the suite with the early-stop PR; until then it is compare_models'.
if hasattr(suite, "paired"):
    paired = suite.paired
else:
    sys.path.insert(0, str(ROOT / "scripts"))
    from compare_models import paired  # noqa: E402


def law_result(cfg, scen, seed, eps, seconds, workers):
    pol = policies.HoldPosePolicy(cfg["residual_baseline"], label="law")
    return suite.evaluate(pol, cfg, scen, episodes=eps, seconds=seconds, seed=seed, workers=workers,
                          progress=lambda *a, **k: None)


def report(tag, r, law, out):
    d = paired(r, law)
    print(f"{tag}: {r.score:.1f} vs law {law.score:.1f}: {d[0]:+.1f} +- {d[1]:.1f} (won/lost {d[2]}/{d[3]})\n"
          f"   {oracle.families(r)}\n   law {oracle.families(law)}", flush=True)
    falls = {s.name: sum(e.fell for e in s.episodes) for s in r.scenarios}
    law_falls = {s.name: sum(e.fell for e in s.episodes) for s in law.scenarios}
    if len(falls) <= 6:
        print(f"   falls {falls}   law {law_falls}", flush=True)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        r.save(out / f"{tag.replace(' ', '_')}_{r.seed}x{r.episodes_per_scenario}.json")
    return d


def no_prior(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["feedback_prior"] = None          # the oracle is the whole controller
    return cfg


def cmd_lqr(a, cfg, out):
    wenv = oracle.make_env(no_prior(cfg))
    u0 = oracle.settle(wenv)
    io_ = oracle.EnvStateIO(wenv.base)
    A, B, x0, snap = oracle.linearize(io_, u0)
    nv = io_.nv
    print(f"linearised: nx {io_.nx}, nu {io_.nu}", flush=True)
    base_w = dict(pos_z=10.0, tilt=100.0, joint=1.0, linvel=10.0, angvel=10.0, jointvel=0.01, internal=1e-3, r=1.0)
    grid = [{**base_w, "tilt": t, "angvel": t / 10, "linvel": t / 10, "pos_z": t / 10, "r": r}
            for t in (10.0, 100.0, 1000.0) for r in (0.1, 1.0, 10.0)]
    scen = suite_scenarios()
    law = law_result(cfg, scen, 12345, 12, 15.0, a.workers)
    rows = []
    for w in grid:
        K, rho = oracle.lqr_gain(A, B, nv, w)
        f = functools.partial(oracle.lqr_factory, K, x0, u0, snap["qpos"], "lqr")
        r = oracle.run_bound(f, no_prior(cfg), scen, 12345, 12, workers=a.workers, label="lqr")
        d = report(f"lqr tilt{w['tilt']:g} r{w['r']:g} (|eig| {rho:.3f})", r, law, None)
        rows.append((d[0], K, w))
    best = max(rows, key=lambda x: x[0])
    print(f"BEST on training seeds: tilt {best[2]['tilt']:g} r {best[2]['r']:g}", flush=True)
    for seed, eps in ((777, 24), (778, 48)):
        f = functools.partial(oracle.lqr_factory, best[1], x0, u0, snap["qpos"], "lqr-best")
        r = oracle.run_bound(f, no_prior(cfg), scen, seed, eps, workers=a.workers, label="lqr-best")
        report("lqr-best", r, law_result(cfg, scen, seed, eps, 15.0, a.workers), out)


def scenarios_for(name):
    if name == "band":
        return oracle.push_band(), 6.0
    if name == "families":
        return [s for s in suite_scenarios() if s.family in FAMS], 15.0
    names = name.split(",")
    return [s for s in suite_scenarios() if s.name in names], 15.0


def cmd_mpc(a, cfg, out):
    params = {k: getattr(a, k) for k in ("N", "H", "knots", "sigma") if getattr(a, k) is not None}
    mcfg = no_prior(cfg)
    if a.clamp is not None:
        mcfg["residual_clamp"] = a.clamp
    scen, seconds = scenarios_for(a.scenarios)
    f = functools.partial(oracle.mpc_factory, params, "mpc")
    r = oracle.run_bound(f, mcfg, scen, a.seed, a.episodes, seconds, a.workers, label="mpc")
    report(f"mpc {json.dumps(params)} clamp {a.clamp}", r, law_result(cfg, scen, a.seed, a.episodes, seconds,
                                                                       a.workers), out)


def cmd_ankle(a, cfg, out):
    pose = cfg["residual_baseline"]
    if a.band:
        scen, seconds = oracle.push_band(), 6.0
        law = law_result(cfg, scen, a.seed, a.episodes, seconds, a.workers)
        for ka in (0.0, -0.3, 0.3, -0.6, 0.6, -1.2, 1.2):
            for kd in ((0.0,) if ka == 0 else (0.0, 0.3 * ka, 0.6 * ka)):
                f = functools.partial(oracle.ankle_factory, pose, a.k, ka, kd, delay_steps=a.ankle_delay)
                r = oracle.run_bound(f, cfg, scen, a.seed, a.episodes, seconds, a.workers, label="ankle")
                report(f"ankle k{a.k:g} ka{ka:+g} kd{kd:+.2f}", r, law, None)
        return
    scen = suite_scenarios()
    f = functools.partial(oracle.ankle_factory, pose, a.k, a.ka, a.kd, delay_steps=a.ankle_delay)
    r = oracle.run_bound(f, cfg, scen, a.seed, a.episodes, 15.0, a.workers, label=f"ankle k{a.k:g} ka{a.ka:g} kd{a.kd:g}")
    report(f"ankle k{a.k:g} ka{a.ka:g} kd{a.kd:g}", r, law_result(cfg, scen, a.seed, a.episodes, 15.0, a.workers), out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("lqr", "mpc", "ankle"))
    ap.add_argument("--config", default="config/experiments/flex_plant_v2.yaml")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--episodes", type=int, default=None, help="default: 24 (suite), 12 (band)")
    ap.add_argument("--workers", type=int, default=suite.default_workers())
    ap.add_argument("--out", default="runs/ceilings", help="where result JSONs go ('' = none)")
    ap.add_argument("--scenarios", default="families", help="mpc: families | band | name,name,...")
    ap.add_argument("--N", type=int)
    ap.add_argument("--H", type=int)
    ap.add_argument("--knots", type=int)
    ap.add_argument("--sigma", type=float)
    ap.add_argument("--clamp", type=float, help="mpc: residual clamp override (the robot's is 0.2)")
    ap.add_argument("--k", type=float, default=4.4, help="ankle: series spring, N m/rad (7 = play only)")
    ap.add_argument("--ka", type=float, default=0.3)
    ap.add_argument("--kd", type=float, default=0.18)
    ap.add_argument("--ankle-delay", type=int, default=oracle.ANKLE_DELAY_STEPS,
                    help="ankle: servo dead time in 25 ms control steps")
    ap.add_argument("--band", action="store_true", help="ankle: gain grid on the single-push band")
    a = ap.parse_args()
    if a.episodes is None:
        a.episodes = 12 if (a.band or a.scenarios == "band") else 24
    cfg = load_experiment(a.config).task_cfg("standing")
    out = Path(a.out) if a.out else None
    np.set_printoptions(precision=3, suppress=True)
    {"lqr": cmd_lqr, "mpc": cmd_mpc, "ankle": cmd_ankle}[a.mode](a, cfg, out)


if __name__ == "__main__":
    main()
