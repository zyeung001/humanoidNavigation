#!/usr/bin/env python3
"""Build a walking reference (one gait cycle) for RealWalkingEnv from step command sequences.

Each input .npy is a (T, 17) sequence of raw joint commands at 40 Hz (the policy-action space:
before the residual box, EMA, dead time and servo lag), as found by a step trajectory search.
The steps are trimmed to their motion plus `--hold-s` of their final command, joined in order,
and the end is blended back to the start over `--blend-s` so the cycle closes. The cycle is then
replayed open-loop in the experiment's env (nominal plant, no scenario) for `--cycles` cycles to
record what the reference does: the sole heights it produces (foot_z, R/L) and the distance it
covers per cycle (advance_m), which the env rewards.

  python scripts/build_walk_reference.py --config config/experiments/flex_plant_v2.yaml \
      --steps step_R.npy step_L_after.npy --motion-steps 140 --out config/gaits/step_to.npz
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environments.real_walking_env import GaitReference, sole_boxes, sole_heights  # noqa: E402
from src.training.experiment import load_experiment  # noqa: E402

DT = 0.025


def join(steps, motion_steps, hold_s, blend_s):
    hold = int(round(hold_s / DT))
    parts = [np.asarray(s, dtype=np.float32)[:motion_steps + hold] for s in steps]
    q = np.concatenate(parts, axis=0)
    nb = int(round(blend_s / DT))
    if nb > 0:
        w = np.linspace(0.0, 1.0, nb, dtype=np.float32)[:, None]
        q[-nb:] = (1 - w) * q[-nb:] + w * q[0]
    return q


def replay(cfg, q, cycles, seed=0, settle_s=3.0):
    """Open-loop replay of the cycle in the robustness suite's `quiet` eval env (nominal plant,
    clean sensors: where the step search ran), after settling on the STANDING pose: the step
    search started from there, as every RealWalkingEnv episode does.
    Returns the sole heights per step, the forward COM travel per cycle, and the step it fell at."""
    from src.robustness import suite
    from src.robustness.scenarios import SUITE
    quiet = next(s for s in SUITE if s.name == "quiet")
    seconds = settle_s + cycles * len(q) * DT + 1.0
    with contextlib.redirect_stdout(io.StringIO()):
        w = suite.make_eval_env(copy.deepcopy(cfg), seconds)
        np.random.seed(seed)
        w.reset(seed=seed, options={"scenario": quiet, "stratum": None})
    b = w.base
    stand = b.episode_baseline.astype(np.float32)
    for _ in range(int(round(settle_s / DT))):
        w.step(stand)
    d = b.env.unwrapped.data
    boxes = sole_boxes(b.env.unwrapped.model, b.model_spec.foot_body_ids)
    soles, xs = [], [float(d.subtree_com[0][0])]
    for c in range(cycles):
        for i, u in enumerate(q):
            _, _, term, _, _ = w.step(u)
            soles.append(sole_heights(d, boxes))
            if term:
                return np.array(soles), np.diff(xs), c * len(q) + i
        xs.append(float(d.subtree_com[0][0]))
    return np.array(soles), np.diff(xs), None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment yaml (its standing block builds the env)")
    p.add_argument("--steps", nargs="+", required=True, help="step command sequences (.npy, (T,17))")
    p.add_argument("--motion-steps", type=int, default=140, help="moving part of each sequence")
    p.add_argument("--hold-s", type=float, default=0.5, help="hold kept after each step's motion")
    p.add_argument("--blend-s", type=float, default=0.5, help="blend of the cycle's end into its start")
    p.add_argument("--cycles", type=int, default=3, help="cycles replayed to measure the reference")
    p.add_argument("--no-prior", action="store_true",
                   help="replay without the config's feedback_prior (as the step search ran)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = load_experiment(a.config).task_cfg("standing")
    cfg["residual_clamp"] = 3.0       # the replay may go anywhere the servos can
    if a.no_prior:
        cfg["feedback_prior"] = None
    q = join([np.load(s) for s in a.steps], a.motion_steps, a.hold_s, a.blend_s)
    soles, advance, fell_at = replay(cfg, q, a.cycles)
    if fell_at is not None:
        raise SystemExit(f"the reference falls open-loop at step {fell_at} ({fell_at * DT:.2f} s); not saved")
    n = len(q)
    foot_z = soles[-n:]                       # the last cycle: past the start-up transient
    t = np.arange(n) * DT
    ref = GaitReference(t, q, foot_z, n * DT, float(np.mean(advance)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, t=ref.t, q=ref.q, foot_z=ref.foot_z, period_s=ref.period, advance_m=ref.advance,
             sources=np.array([str(s) for s in a.steps]))
    air = (foot_z > 0.008).mean(0)
    print(f"saved {a.out}: period {ref.period:.2f} s, advance per cycle {100 * advance.round(4)} cm "
          f"-> {100 * ref.speed:.2f} cm/s; sole up (>8 mm) R {100 * air[0]:.0f}% / L {100 * air[1]:.0f}% "
          f"of the cycle, max {1000 * foot_z.max(0).round(4)} mm")


if __name__ == "__main__":
    main()
