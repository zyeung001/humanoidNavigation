"""
Evaluation harness for navigation rebuild models.

Runs N rollouts on NavRebuildEnv and reports:
  - goal-reach rate (Phase 1 pass criterion: >80%)
  - breakdown by termination cause (goal / collision / fall / truncated)
  - mean steps-to-goal on successful episodes
  - mean final distance to goal
  - mean cumulative reward

Usage (Phase 1 open arena):
    python scripts/eval_nav.py \\
        --model runs/nav_phase1/model_final.zip \\
        --vecnorm runs/nav_phase1/vecnorm_final.pkl \\
        --episodes 50

Per-episode JSONL log can be written with --log <path> for the reward-hack
watchlist.
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import configure_mujoco_gl  # noqa: E402
configure_mujoco_gl()

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from src.environments.nav_rebuild_env import NavRebuildEnv  # noqa: E402


def _safe_getstate(self):
    state = self.__dict__.copy()
    state.pop("venv", None)
    state.pop("class_attributes", None)
    state.pop("returns", None)
    return state
VecNormalize.__getstate__ = _safe_getstate


def _make_env(env_kwargs: dict, seed: int):
    def _init():
        configure_mujoco_gl()
        return NavRebuildEnv(seed=seed, **env_kwargs)
    return _init


def run_episode(model, vn, base_env, deterministic: bool):
    """Run one episode; return dict of per-episode metrics."""
    obs = vn.reset()
    ep_reward = 0.0
    steps = 0
    cause = "truncated"
    last_info = {}

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, dones, infos = vn.step(action)
        ep_reward += float(reward[0])
        steps += 1
        last_info = infos[0]
        if dones[0]:
            cause = last_info.get("termination_cause") or (
                "truncated" if last_info.get("TimeLimit.truncated") else "term"
            )
            break

    final_dist = float(last_info.get("nav_dist_to_goal", float("nan")))
    progress_arc = float(last_info.get("nav_progress_arc", 0.0))
    path_len = float(getattr(base_env, "path_total_length", 0.0))
    return {
        "cause": cause,
        "steps": steps,
        "ep_reward": ep_reward,
        "final_dist": final_dist,
        "progress_arc": progress_arc,
        "path_length": path_len,
    }


def main():
    p = argparse.ArgumentParser(description="Eval harness for navigation rebuild")
    p.add_argument("--model", required=True, help="PPO model .zip")
    p.add_argument("--vecnorm", required=True, help="VecNormalize .pkl")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=1000,
                   help="Base seed; episode k uses seed+k.")
    p.add_argument("--max-episode-steps", type=int, default=1500)
    p.add_argument("--goal-dist", type=float, default=5.0,
                   help="Open-arena random goal distance (m).")
    p.add_argument("--no-deterministic", action="store_true",
                   help="Sample actions stochastically (default: deterministic).")
    p.add_argument("--log", default=None,
                   help="Optional JSONL path for per-episode metrics.")
    args = p.parse_args()

    env_kwargs = dict(
        open_arena=True,
        open_arena_goal_dist=args.goal_dist,
        max_episode_steps=args.max_episode_steps,
    )

    print("=" * 64)
    print("Phase 1 navigation eval")
    print(f"  model:       {args.model}")
    print(f"  vecnorm:     {args.vecnorm}")
    print(f"  episodes:    {args.episodes}")
    print(f"  goal_dist:   {args.goal_dist} m")
    print(f"  max_steps:   {args.max_episode_steps}")
    print(f"  determ:      {not args.no_deterministic}")
    print("=" * 64)

    # Single-env eval. Re-seed each episode for reproducibility.
    base_env_holder = {}

    def _factory():
        configure_mujoco_gl()
        env = NavRebuildEnv(seed=args.seed, **env_kwargs)
        base_env_holder["env"] = env
        return env

    venv = DummyVecEnv([_factory])
    vn = VecNormalize.load(args.vecnorm, venv)
    vn.training = False
    vn.norm_reward = False

    print("\nLoading model...")
    model = PPO.load(args.model, env=vn, device="cpu")
    print(f"  obs space: {model.observation_space.shape}")

    base_env = base_env_holder["env"]

    log_f = open(args.log, "w") if args.log else None
    results = []
    for k in range(args.episodes):
        # Seed the underlying env per-episode for reproducibility. NavRebuildEnv.reset()
        # only re-seeds when a seed kwarg is passed; here we pre-set _rng instead.
        base_env._rng = np.random.RandomState(args.seed + k)
        r = run_episode(model, vn, base_env,
                        deterministic=not args.no_deterministic)
        r["episode"] = k
        r["seed"] = args.seed + k
        results.append(r)
        if log_f:
            log_f.write(json.dumps(r) + "\n")
            log_f.flush()
        print(f"  ep {k:3d} seed={args.seed + k:4d} "
              f"cause={r['cause']:>10s} steps={r['steps']:4d} "
              f"final_dist={r['final_dist']:5.2f}m "
              f"reward={r['ep_reward']:7.2f}")

    if log_f:
        log_f.close()

    # ---------- summary ----------
    n = len(results)
    causes = {}
    for r in results:
        causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    n_goal = causes.get("goal", 0)
    goal_rate = n_goal / n if n else 0.0

    successes = [r for r in results if r["cause"] == "goal"]
    mean_steps_success = (np.mean([r["steps"] for r in successes])
                          if successes else float("nan"))
    mean_final_dist = float(np.mean([r["final_dist"] for r in results
                                     if np.isfinite(r["final_dist"])]))
    mean_reward = float(np.mean([r["ep_reward"] for r in results]))

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Episodes:           {n}")
    print(f"  Goal-reach rate:    {goal_rate:.1%}  ({n_goal}/{n})")
    print(f"  Termination breakdown:")
    for cause in sorted(causes.keys(), key=lambda c: -causes[c]):
        print(f"    {cause:>12s}: {causes[cause]:3d}  ({causes[cause]/n:.1%})")
    print(f"  Mean steps (goal-reaching only): {mean_steps_success:.0f}")
    print(f"  Mean final distance to goal:     {mean_final_dist:.2f} m")
    print(f"  Mean cumulative reward:          {mean_reward:.2f}")
    print()
    pass_threshold = 0.80
    verdict = "PASS" if goal_rate >= pass_threshold else "FAIL"
    print(f"  Phase 1 pass criterion (>{pass_threshold:.0%}): {verdict}")
    print("=" * 64)

    venv.close()
    sys.exit(0 if goal_rate >= pass_threshold else 1)


if __name__ == "__main__":
    main()
