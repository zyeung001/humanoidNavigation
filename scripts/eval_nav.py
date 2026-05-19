"""
Evaluation harness for navigation rebuild models.

Runs N rollouts on NavRebuildEnv and reports:
  - goal-reach rate (pass criterion: >=80%)
  - breakdown by termination cause (goal / collision / fall / truncated)
  - mean steps-to-goal on successful episodes
  - mean final distance to goal
  - mean cumulative reward

Supports four maze modes via --maze-type:
  open         — Phase 1 open arena (default)
  corridor / L / U / medium — fixed maze grids from src.maze.maze_maps
  procedural   — fresh random maze each episode

Usage:
    python scripts/eval_nav.py \\
        --model runs/nav_phase3_v3/model_final.zip \\
        --vecnorm runs/nav_phase3_v3/vecnorm_final.pkl \\
        --maze-type corridor --episodes 50

Per-episode JSONL log: --log <path>.
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
from src.maze.maze_maps import CORRIDOR, L_MAZE, U_MAZE, MEDIUM_MAZE  # noqa: E402
# Composite (third-person + top-down) recording helpers.
from scripts.run_maze_nav import (  # noqa: E402
    CachedRenderers,
    render_topdown_map,
    composite_frame,
)

MAZE_GRIDS = {
    "corridor": CORRIDOR,
    "L": L_MAZE,
    "U": U_MAZE,
    "medium": MEDIUM_MAZE,
}


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


def run_episode(model, vn, base_env, deterministic: bool, frames=None,
                record_every=3, renderers=None, rebuild_renderers_after_reset=False):
    """Run one episode; return dict of per-episode metrics. If `frames` is a list,
    append composite RGB frames (third-person + top-down map) every `record_every`
    steps when `renderers` is provided; otherwise falls back to base_env.render().

    If `rebuild_renderers_after_reset` is True, builds a fresh CachedRenderers
    against the post-reset mj_model (required when the underlying maze geometry —
    and thus the model — changes per reset, as in procedural mode)."""
    obs = vn.reset()
    if rebuild_renderers_after_reset and frames is not None:
        if renderers is not None:
            try:
                renderers.close()
            except Exception:
                pass
        mj_model = base_env.env.unwrapped.model
        renderers = CachedRenderers(mj_model, tp_width=640, tp_height=480, td_size=480)
    ep_reward = 0.0
    steps = 0
    cause = "truncated"
    last_info = {}
    height_sum = 0.0
    height_min = float("inf")
    height_n = 0

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, dones, infos = vn.step(action)
        ep_reward += float(reward[0])
        steps += 1
        last_info = infos[0]
        _h = last_info.get("nav_height")
        if _h is not None:
            _h = float(_h)
            height_sum += _h
            height_min = min(height_min, _h)
            height_n += 1
        if frames is not None and (steps % record_every == 0):
            try:
                if renderers is not None:
                    mj_data = base_env.env.unwrapped.data
                    tp = renderers.render_third_person(mj_data)
                    td = render_topdown_map(
                        renderers, mj_data, base_env.grid, base_env.cell_size,
                        goal_xy=base_env.goal_pos, waypoints=list(base_env.path),
                    )
                    frames.append(composite_frame(tp, td))
                else:
                    img = base_env.render()
                    if img is not None:
                        frames.append(img)
            except Exception as e:
                if steps == record_every:
                    print(f"    [record] warning: render failed: {e}")
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
        "mean_height": (height_sum / height_n) if height_n else float("nan"),
        "min_height": height_min if height_n else float("nan"),
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
    p.add_argument("--maze-type", default="open",
                   choices=["open", "procedural"] + list(MAZE_GRIDS.keys()),
                   help="open = Phase 1 open arena (default). "
                        "procedural = fresh random maze each episode. "
                        "Others use the named fixed maze grid.")
    p.add_argument("--proc-rows-min", type=int, default=2)
    p.add_argument("--proc-rows-max", type=int, default=3)
    p.add_argument("--proc-cols-min", type=int, default=2)
    p.add_argument("--proc-cols-max", type=int, default=3)
    p.add_argument("--proc-algorithm", default="dfs", choices=["dfs", "prims"])
    p.add_argument("--no-deterministic", action="store_true",
                   help="Sample actions stochastically (default: deterministic).")
    p.add_argument("--log", default=None,
                   help="Optional JSONL path for per-episode metrics.")
    p.add_argument("--record", action="store_true",
                   help="Record video(s) using MuJoCo rgb_array rendering.")
    p.add_argument("--record-episodes", type=int, default=3,
                   help="Number of episodes to record (default 3, starting from ep 0).")
    p.add_argument("--record-interval", type=int, default=3,
                   help="Render every Nth env step (higher = faster, smaller video).")
    p.add_argument("--video-dir", default=None,
                   help="Output dir for recorded mp4s (default: runs/<derived>/videos).")
    args = p.parse_args()

    render_mode = "rgb_array" if args.record else None

    if args.maze_type == "open":
        env_kwargs = dict(
            open_arena=True,
            open_arena_goal_dist=args.goal_dist,
            max_episode_steps=args.max_episode_steps,
            render_mode=render_mode,
        )
    elif args.maze_type == "procedural":
        env_kwargs = dict(
            open_arena=False,
            procedural=True,
            proc_rows_range=(args.proc_rows_min, args.proc_rows_max),
            proc_cols_range=(args.proc_cols_min, args.proc_cols_max),
            proc_algorithm=args.proc_algorithm,
            max_episode_steps=args.max_episode_steps,
            render_mode=render_mode,
        )
    else:
        env_kwargs = dict(
            open_arena=False,
            grid=MAZE_GRIDS[args.maze_type],
            max_episode_steps=args.max_episode_steps,
            render_mode=render_mode,
        )

    print("=" * 64)
    print(f"Navigation eval ({args.maze_type})")
    print(f"  model:       {args.model}")
    print(f"  vecnorm:     {args.vecnorm}")
    print(f"  episodes:    {args.episodes}")
    print(f"  maze_type:   {args.maze_type}")
    if args.maze_type == "open":
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

    video_dir = None
    renderers = None
    if args.record:
        from pathlib import Path as _Path
        if args.video_dir:
            video_dir = _Path(args.video_dir)
        else:
            video_dir = _Path(args.model).parent / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        # Composite renderer requires the maze_topdown camera, which only exists
        # in generated maze MJCFs (not the open-arena default Humanoid-v5 XML).
        if args.maze_type != "open":
            mj_model = base_env.env.unwrapped.model
            renderers = CachedRenderers(mj_model, tp_width=640, tp_height=480, td_size=480)
        print(f"  recording {args.record_episodes} episode(s) -> {video_dir}"
              f"  composite={'yes' if renderers else 'no'}")

    results = []
    for k in range(args.episodes):
        # Seed the underlying env per-episode for reproducibility. NavRebuildEnv.reset()
        # only re-seeds when a seed kwarg is passed; here we pre-set _rng instead.
        base_env._rng = np.random.RandomState(args.seed + k)

        frames = [] if (args.record and k < args.record_episodes) else None
        r = run_episode(model, vn, base_env,
                        deterministic=not args.no_deterministic,
                        frames=frames,
                        record_every=args.record_interval,
                        renderers=renderers,
                        rebuild_renderers_after_reset=(
                            args.record and args.maze_type == "procedural"
                            and k < args.record_episodes))
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

        if frames:
            import imageio.v2 as imageio
            fps = max(1, 30 // args.record_interval)
            out_path = video_dir / f"{args.maze_type}_ep{k:02d}_{r['cause']}.mp4"
            imageio.mimsave(str(out_path), frames, fps=fps, macro_block_size=1)
            print(f"       saved video -> {out_path} ({len(frames)} frames @ {fps}fps)")

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
    print("  Termination breakdown:")
    for cause in sorted(causes.keys(), key=lambda c: -causes[c]):
        print(f"    {cause:>12s}: {causes[cause]:3d}  ({causes[cause]/n:.1%})")
    print(f"  Mean steps (goal-reaching only): {mean_steps_success:.0f}")
    print(f"  Mean final distance to goal:     {mean_final_dist:.2f} m")
    print(f"  Mean cumulative reward:          {mean_reward:.2f}")

    heights = [r["mean_height"] for r in results
               if np.isfinite(r.get("mean_height", float("nan")))]
    mins = [r["min_height"] for r in results
            if np.isfinite(r.get("min_height", float("nan")))]
    gait = "n/a"
    mean_h = float("nan")
    if heights:
        mean_h = float(np.mean(heights))
        worst_min = float(np.min(mins)) if mins else float("nan")
        # Walking gait ~1.40m. v7b crawl was ~0.83m mean. Flag the gap.
        if mean_h >= 1.20:
            gait = "WALKING"
        elif mean_h >= 1.00:
            gait = "DEGRADED (low gait)"
        else:
            gait = "CRAWLING (hollow pass risk)"
        print(f"  Mean torso height:               {mean_h:.2f} m "
              f"(min {worst_min:.2f}) -> gait: {gait}")
    print()
    pass_threshold = 0.80
    goal_ok = goal_rate >= pass_threshold
    # A goal-reach pass at crawl height is the v7b hollow-pass failure mode.
    gait_ok = (not heights) or mean_h >= 1.20
    if goal_ok and gait_ok:
        verdict = "PASS"
    elif goal_ok and not gait_ok:
        verdict = "HOLLOW (goal-reach OK but gait collapsed — see v7b)"
    else:
        verdict = "FAIL"
    print(f"  Pass criterion (goal>={pass_threshold:.0%} AND gait>=1.20m): "
          f"{verdict}")
    print("=" * 64)

    venv.close()
    sys.exit(0 if goal_rate >= pass_threshold else 1)


if __name__ == "__main__":
    main()
