#!/usr/bin/env python3
"""Scoreboard for a standing policy: one comparable row per model, appended to a CSV.

WHY THIS EXISTS. Until 9/19 nothing here could tell a good run from a bad one.
gate_standing_kick.py is binary and EVERY model passes it -- v1, v2, v3, measured and stand
all went 6/6, and four of them fell on hardware within seconds. ep_len_mean was never
logged. So every change to the reward, the plant or the randomisation was judged by a test
it could not fail, which means none of them were judged at all.

That makes domain randomisation in particular unfalsifiable. It is not free -- it trades
peak performance for robustness and can stop a policy learning outright -- so "did this help
or hurt" has to come back as a number. This produces that number.

FOUR THINGS IT DOES THAT THE GATE DOES NOT

  1. FIXED SEEDS. Every model meets the same initial states and the same kick schedule, so a
     difference between two rows is the model and not the draw.
  2. A MAGNITUDE SWEEP rather than pass/fail. The gate kicks at exactly the trained
     magnitude, so all it can report is that the policy handles what it trained on. What
     matters is the BASIN -- how far past that it survives -- because that margin is what
     has to absorb the sim-to-real gap.
  3. RECOVERY TIME in control steps, for |ang_vel| to fall back under the disturbance that
     caused it. A policy that survives by flailing for four seconds and one that settles in
     twenty steps both "pass"; only this separates them, and only this compares against a
     finger push on the real robot.
  4. HELD-OUT PLANTS (--held-out): mass and friction drawn WIDER than training. A policy
     that only survives its own training distribution has memorised it.

It also carries mean excursion^2 (see check_saturation.py) into the same row, because a
policy whose output is being clipped away is not the controller the other columns measure.

  python scripts/eval_standing.py --model models/final_real_standing_stand.zip \
      --vecnorm models/vecnorm_real_standing_stand.pkl \
      --config config/real_humanoid_stand.yaml --label stand-155M
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
from src.environments import make_standing_env  # noqa: E402

SCOREBOARD = PROJ / "models" / "standing" / "scoreboard.csv"

SETTLE_STEPS = 100        # let the policy reach its stance before anything is asked of it
RECOVER_STEPS = 300       # 7.5 s at 40 Hz -- long enough for a slow recovery to show
QUIET_STEPS = 1200        # 30 s of undisturbed standing
LEVELS = (0.5, 1.0, 2.0, 3.0, 4.0)   # multiples of the TRAINED kick magnitude
N_DIRS = 8                # kick directions per level
CALM_RATE = 0.30          # rad/s: |ang_vel| under this counts as settled
CALM_STEPS = 10           # and it has to stay there for a quarter second


def tilt_cos(data, bid):
    """Yaw-invariant cosine of tilt from vertical (R22). Never quat_w -- it fires on yaw."""
    _w, x, y, _z = data.xquat[bid]
    return float(1.0 - 2.0 * (x * x + y * y))


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJ,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "?"


def build(cfg, vecnorm, model_path, seed):
    from stable_baselines3 import PPO  # noqa: PLC0415
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: PLC0415
    env = make_standing_env(config=cfg)
    venv = VecNormalize.load(str(vecnorm), DummyVecEnv([lambda: env]))
    venv.training = False
    venv.norm_reward = False
    venv.seed(seed)
    return env, venv, PPO.load(str(model_path), device="cpu")


def run_trial(model, venv, env, mj, bid, kick=None, steps=None, settle=SETTLE_STEPS):
    """One episode. `kick` is (vx, vy, yaw_rate) applied at step `settle`."""
    import mujoco  # noqa: PLC0415
    steps = steps if steps is not None else SETTLE_STEPS + RECOVER_STEPS
    d = env.unwrapped.data
    obs = venv.reset()
    min_tilt, recov, kicked_at, calm = 1.0, None, None, 0
    leans = []
    for step in range(steps):
        if kick is not None and step == settle:
            d.qvel[0] += kick[0]
            d.qvel[1] += kick[1]
            d.qvel[5] += kick[2]
            mujoco.mj_forward(mj, d)
            kicked_at = step
        a, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = venv.step(a)
        tc = tilt_cos(d, bid)
        if kicked_at is not None:
            min_tilt = min(min_tilt, tc)
            # Recovered = angular rate back under a FIXED threshold, held for CALM_STEPS.
            # The threshold must not scale with the kick: tying it to the disturbance gives
            # a bigger kick an easier bar and returns 1 step at every level, which is what
            # the first version of this did. Requiring it to stay down rejects the moment
            # the rate passes through zero mid-swing on the way to falling over.
            if recov is None and step > kicked_at:
                if float(np.linalg.norm(d.qvel[3:6])) < CALM_RATE:
                    calm += 1
                    if calm >= CALM_STEPS:
                        recov = step - kicked_at - CALM_STEPS + 1
                else:
                    calm = 0
        else:
            leans.append(float(np.degrees(np.arccos(np.clip(tc, -1.0, 1.0)))))
        if bool(done[0]):
            return {"survived": False, "steps": step, "min_tilt": min_tilt,
                    "recovery": recov, "lean": leans}
    return {"survived": True, "steps": steps, "min_tilt": min_tilt,
            "recovery": recov, "lean": leans}


def saturation(model, venv, raw_cfg, episodes=3, steps=400):
    """Mean excursion^2: how much of the raw policy the residual clamp is discarding."""
    clamp = np.asarray(raw_cfg.get("residual_clamp", 0.0), float)
    if not np.any(clamp > 0.0):
        return float("nan"), 0
    base = np.asarray(raw_cfg["residual_baseline"], float)
    acts = []
    for _ in range(episodes):
        obs = venv.reset()
        for _ in range(steps):
            a, _ = model.predict(obs, deterministic=True)
            acts.append(np.asarray(a, float).ravel())
            obs, _, done, _ = venv.step(a)
            if done[0]:
                break
    A = np.array(acts)
    exc = np.maximum(np.abs(A - base) - clamp, 0.0)
    frac = ((A < base - clamp - 1e-9) | (A > base + clamp + 1e-9)).mean(axis=0)
    return float(np.square(exc).sum(axis=1).mean()), int((frac > 0.5).sum())


def main():
    p = argparse.ArgumentParser(description="Comparable scoreboard row for a standing policy")
    p.add_argument("--model", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--label", default=None, help="short name for the scoreboard row")
    p.add_argument("--seed", type=int, default=12345,
                   help="fixes the initial states AND the kick schedule. Keep it identical "
                        "across models or the rows are not comparable.")
    p.add_argument("--held-out", action="store_true",
                   help="evaluate on mass/friction draws WIDER than training, to see whether "
                        "the policy generalised or memorised its own distribution")
    p.add_argument("--scoreboard", default=str(SCOREBOARD))
    p.add_argument("--no-append", action="store_true")
    args = p.parse_args()

    import mujoco  # noqa: PLC0415
    raw = yaml.safe_load(open(PROJ / args.config)).get("standing") or {}
    cfg = raw.copy()
    # The BENCHMARK has to be deterministic, so the env's own random pushes and observation
    # noise are off; kicks are injected here on a fixed schedule instead.
    cfg.update(obs_noise=False, push_enabled=False, actuator_rand=False,
               max_episode_steps=max(SETTLE_STEPS + RECOVER_STEPS, QUIET_STEPS) + 50)
    if args.held_out:
        # Deliberately outside what it trained on. Surviving only the training distribution
        # is memorisation, and this is the column that tells the two apart.
        lo_m, hi_m = raw.get("rand_mass_range", [0.8, 1.2])
        lo_f, hi_f = raw.get("rand_friction_range", [0.6, 1.4])
        cfg.update(domain_rand=True,
                   rand_mass_range=[lo_m - 0.15, hi_m + 0.15],
                   rand_friction_range=[max(0.05, lo_f - 0.2), hi_f + 0.2])
    else:
        cfg.update(domain_rand=False)

    lin0 = float(raw.get("push_lin_vel", 0.15))
    yaw0 = float(raw.get("push_ang_vel", 0.5))
    env, venv, model = build(cfg, PROJ / args.vecnorm, PROJ / args.model, args.seed)
    mj = env.unwrapped.model
    bid = max(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "base_link"), 1)

    label = args.label or Path(args.model).stem
    print(f"\n=== {label} ===")
    print(f"model  {args.model}")
    print(f"config {args.config}   seed {args.seed}   "
          f"plant {'HELD-OUT (wider than training)' if args.held_out else 'nominal'}")
    print(f"trained kick magnitude: lin {lin0} m/s, yaw {yaw0} rad/s\n")

    # ---- quiet standing ---------------------------------------------------------------
    q = run_trial(model, venv, env, mj, bid, kick=None, steps=QUIET_STEPS, settle=QUIET_STEPS)
    lean_med = float(np.median(q["lean"])) if q["lean"] else float("nan")
    lean_sd = float(np.std(q["lean"])) if q["lean"] else float("nan")
    print(f"quiet stand: {'held all' if q['survived'] else 'FELL at'} {q['steps']} steps "
          f"({q['steps'] / 40:.1f} s)   lean {lean_med:.2f} deg (spread {lean_sd:.2f})")

    # ---- kick sweep -------------------------------------------------------------------
    print(f"\nkick sweep, {N_DIRS} directions per level (x = multiple of trained magnitude)")
    print(f"  {'level':>6s} {'lin m/s':>8s} {'survived':>12s} {'tilt dip':>9s} {'recov':>7s}")
    print("  " + "-" * 48)
    rng = np.random.default_rng(args.seed)
    dirs = np.linspace(0.0, 2.0 * np.pi, N_DIRS, endpoint=False) + rng.uniform(0.0, 0.3)
    basin, per_level = None, {}
    for lv in LEVELS:
        surv, dips, recs = 0, [], []
        for k, ang in enumerate(dirs):
            kick = (lv * lin0 * np.cos(ang), lv * lin0 * np.sin(ang),
                    lv * yaw0 * (+1.0 if k % 2 == 0 else -1.0))
            r = run_trial(model, venv, env, mj, bid, kick=kick)
            surv += int(r["survived"])
            dips.append(r["min_tilt"])
            if r["survived"] and r["recovery"] is not None:
                recs.append(r["recovery"])
        rate = surv / len(dirs)
        med_rec = float(np.median(recs)) if recs else float("nan")
        per_level[lv] = (rate, float(np.min(dips)), med_rec)
        print(f"  {lv:5.1f}x {lv * lin0:8.2f} {surv:>5d}/{len(dirs)} {100 * rate:4.0f}% "
              f"{np.min(dips):9.3f} {med_rec:7.0f}")
        if basin is None and rate < 1.0:
            basin = lv
    if basin is None:
        basin = float(LEVELS[-1])
        print(f"\n  BASIN: survived every level to {basin:.1f}x -- widen LEVELS to find the edge")
    else:
        print(f"\n  BASIN: first failure at {basin:.1f}x the trained magnitude")

    exc2, pinned = saturation(model, venv, raw)
    print(f"  saturation: mean excursion^2 {exc2:.3f} rad^2, {pinned} joint(s) pinned")

    row = {
        "when": time.strftime("%Y-%m-%d %H:%M"), "label": label, "git": git_sha(),
        "model": args.model, "config": args.config, "seed": args.seed,
        "held_out": int(args.held_out),
        "quiet_steps": q["steps"], "quiet_lean_deg": round(lean_med, 3),
        "quiet_lean_sd": round(lean_sd, 3),
        "basin_x": basin, "excursion2": round(exc2, 4), "pinned": pinned,
    }
    for lv in LEVELS:
        rate, dip, rec = per_level[lv]
        row[f"surv_{lv:g}x"] = round(rate, 3)
        row[f"dip_{lv:g}x"] = round(dip, 3)
        row[f"recov_{lv:g}x"] = None if np.isnan(rec) else int(rec)

    if not args.no_append:
        path = Path(args.scoreboard)
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if is_new:
                w.writeheader()
            w.writerow(row)
        print(f"\nappended to {path}")
        print("Compare ROWS, not absolutes: the seed is fixed, so a difference between two "
              "rows is the model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
