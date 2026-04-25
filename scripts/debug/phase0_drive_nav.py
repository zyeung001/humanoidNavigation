"""
Phase 0 verification for NavRebuildEnv.

Exercises each termination cause (goal / collision / fall) with deterministic
scripted scenarios, plus a "drive with action stream" smoke test to inspect
the reward curve.

Pass criteria from NAVIGATION_REBUILD_PLAN.md Phase 0:
  - Reward curve is sane.
  - No obvious exploit (standing-still, going backwards, sidestep through wall
    must all be net-negative compared to reaching the goal).
  - Episode terminates correctly on goal / collision / fall.

Run:
    python -m scripts.debug.phase0_drive_nav
"""

from __future__ import annotations

import argparse
import sys
import textwrap

import numpy as np

from src.environments.nav_rebuild_env import NavRebuildEnv
from src.maze.maze_maps import CORRIDOR, L_MAZE


def _make_env(grid=None, max_steps=2000, seed=0):
    grid = grid if grid is not None else CORRIDOR.copy()
    return NavRebuildEnv(grid=grid, max_episode_steps=max_steps, seed=seed)


def _teleport(env, x, y, height=1.4):
    base = env.env.unwrapped
    base.data.qpos[0] = x
    base.data.qpos[1] = y
    base.data.qpos[2] = height
    # Zero velocities so the position holds for one step.
    base.data.qvel[:] = 0.0


def test_construct_and_reset() -> bool:
    print("\n=== TEST 1: construct + reset ===")
    env = _make_env(seed=0)
    obs, info = env.reset()
    expected = env._frozen_obs_dim
    print(f"  obs shape: {obs.shape}  (expected {expected})")
    print(f"  path waypoints: {len(info['nav_path'])}")
    print(f"  path arc length: {info['nav_path_length']:.2f} m")
    print(f"  start: {info['nav_start']}")
    print(f"  goal:  {info['nav_goal']}")
    ok = (
        obs.shape == (expected,)
        and len(info["nav_path"]) >= 2
        and info["nav_path_length"] > 0
    )
    print(f"  pass: {ok}")
    env.close()
    return ok


def test_goal_termination() -> bool:
    print("\n=== TEST 2: goal termination ===")
    env = _make_env(seed=0)
    _, info = env.reset()
    gx, gy = info["nav_goal"]
    _teleport(env, gx, gy)
    _, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape))
    print(f"  reward={reward:.2f}  terminated={terminated}  "
          f"cause={info.get('termination_cause')}")
    ok = (
        terminated
        and info.get("termination_cause") == "goal"
        and reward > env.goal_bonus * 0.5
    )
    print(f"  pass: {ok}")
    env.close()
    return ok


def test_collision_termination() -> bool:
    print("\n=== TEST 3: collision termination ===")
    env = _make_env(seed=0)
    _, info = env.reset()
    # Find a wall geom and teleport agent on top of it.
    base = env.env.unwrapped
    wall_id = next(iter(env._wall_geom_ids))
    wx, wy, _wz = base.model.geom_pos[wall_id]
    _teleport(env, float(wx), float(wy))
    # One zero-action step to settle contact detection.
    _, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape))
    cause = info.get("termination_cause")
    print(f"  reward={reward:.2f}  terminated={terminated}  cause={cause}")
    # Note: dropping the humanoid into a tall box can also produce a fall;
    # accept either collision or fall here, as long as termination fires.
    ok = terminated and cause in ("collision", "fall")
    print(f"  pass: {ok}  (collision specifically: {cause == 'collision'})")
    env.close()
    return ok


def test_fall_termination() -> bool:
    print("\n=== TEST 4: fall termination ===")
    env = _make_env(seed=0)
    _, info = env.reset()
    sx, sy = info["nav_start"]
    _teleport(env, sx, sy, height=0.3)  # well below fall threshold
    _, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape))
    cause = info.get("termination_cause")
    print(f"  reward={reward:.2f}  terminated={terminated}  cause={cause}")
    ok = terminated and cause == "fall" and reward < 0
    print(f"  pass: {ok}")
    env.close()
    return ok


def test_standing_still_is_negative() -> bool:
    """Standing still (zero action) should NEVER be net positive.

    Confirms there is no survival exploit: time penalty + eventual fall must
    dominate any incidental progress reward from initial settling.
    """
    print("\n=== TEST 5: standing still is net negative ===")
    env = _make_env(max_steps=2000, seed=0)
    _, _ = env.reset()
    total = 0.0
    steps = 0
    cause = None
    while True:
        _, r, term, trunc, info = env.step(np.zeros(env.action_space.shape))
        total += r
        steps += 1
        if term or trunc:
            cause = info.get("termination_cause") or ("truncated" if trunc else "term")
            break
    print(f"  total reward={total:.2f}  steps={steps}  cause={cause}")
    ok = total < 0
    print(f"  pass: {ok}  (must be < 0 — no survival exploit)")
    env.close()
    return ok


def test_progress_reward_fires() -> bool:
    """Manually advance agent along path; verify progress reward is positive."""
    print("\n=== TEST 6: progress reward fires on forward motion ===")
    env = _make_env(seed=0)
    _, info = env.reset()
    path = info["nav_path"]
    if len(path) < 2:
        print("  skip — degenerate path")
        env.close()
        return True

    # Teleport agent halfway from start to first non-start waypoint.
    sx, sy = path[0]
    nx, ny = path[1]
    mid_x = sx + 0.4 * (nx - sx)
    mid_y = sy + 0.4 * (ny - sy)
    _teleport(env, mid_x, mid_y)
    _, r, term, _, info = env.step(np.zeros(env.action_space.shape))
    progress_r = info.get("reward/progress", 0.0)
    print(f"  progress_reward={progress_r:.3f}  total_step_reward={r:.3f}  "
          f"terminated={term}  cause={info.get('termination_cause')}")
    ok = progress_r > 0.1
    print(f"  pass: {ok}")
    env.close()
    return ok


def test_no_euclidean_exploit() -> bool:
    """Ensure progress is arc-length, not Euclidean.

    Teleport to the L-maze far corner using a path that's close to the goal
    in Euclidean distance but FAR in path arc-length. Progress should not
    reward this jump.
    """
    print("\n=== TEST 7: no Euclidean ghost-waypoint exploit (L-maze) ===")
    env = _make_env(grid=L_MAZE.copy(), seed=0)
    _, info = env.reset()
    gx, gy = info["nav_goal"]
    sx, sy = info["nav_start"]
    # Place agent near the goal in straight-line Euclidean distance, but
    # NOT on the path. We pick a point offset perpendicular to the path
    # near the goal.
    off_x = gx + 0.05  # tiny offset to avoid being exactly on the goal
    off_y = sy         # same y as start — far from goal along path
    _teleport(env, off_x, off_y)
    _, r, term, _, info = env.step(np.zeros(env.action_space.shape))
    progress_r = info.get("reward/progress", 0.0)
    print(f"  start=({sx:.2f},{sy:.2f})  goal=({gx:.2f},{gy:.2f})  "
          f"agent=({off_x:.2f},{off_y:.2f})")
    print(f"  progress_reward={progress_r:.3f}  total_step_reward={r:.3f}  "
          f"cause={info.get('termination_cause')}")
    # Progress should be small or zero — the agent did not advance along
    # the actual L-shaped path. A naive Euclidean reward would have seen
    # this as huge progress.
    ok = progress_r < 5.0
    print(f"  pass: {ok}  (progress should not be huge for a Euclidean shortcut)")
    env.close()
    return ok


def test_random_action_drive() -> bool:
    """Smoke test with random actions: log reward distribution and termination."""
    print("\n=== TEST 8: random-action drive (smoke) ===")
    env = _make_env(seed=42)
    _, _ = env.reset()
    rng = np.random.RandomState(42)
    rewards = []
    progress = []
    causes = {}
    n_eps = 5
    for _ in range(n_eps):
        env.reset()
        ep_r = 0.0
        ep_progress = []
        for _ in range(env.max_episode_steps):
            a = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
            _, r, term, trunc, info = env.step(a)
            ep_r += r
            ep_progress.append(info.get("reward/progress", 0.0))
            if term or trunc:
                cause = info.get("termination_cause") or "truncated"
                causes[cause] = causes.get(cause, 0) + 1
                break
        rewards.append(ep_r)
        progress.append(np.sum(ep_progress))
    print(f"  episodes: {n_eps}")
    print(f"  mean total reward: {np.mean(rewards):.2f}  "
          f"(min={np.min(rewards):.2f}, max={np.max(rewards):.2f})")
    print(f"  mean total progress reward: {np.mean(progress):.2f}")
    print(f"  termination causes: {causes}")
    env.close()
    # Smoke test: just confirm episodes terminate and reward is finite.
    ok = all(np.isfinite(r) for r in rewards) and len(causes) >= 1
    print(f"  pass: {ok}")
    return ok


TESTS = [
    ("construct+reset", test_construct_and_reset),
    ("goal termination", test_goal_termination),
    ("collision termination", test_collision_termination),
    ("fall termination", test_fall_termination),
    ("standing-still net negative", test_standing_still_is_negative),
    ("progress reward fires", test_progress_reward_fires),
    ("no Euclidean exploit", test_no_euclidean_exploit),
    ("random-action smoke", test_random_action_drive),
]


def main():
    p = argparse.ArgumentParser(description=textwrap.dedent(__doc__))
    p.add_argument("--only", default=None,
                   help="Run only the test whose name contains this substring.")
    args = p.parse_args()

    results = []
    for name, fn in TESTS:
        if args.only and args.only not in name:
            continue
        try:
            ok = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok = False
            print(f"  EXCEPTION in {name}: {e}")
        results.append((name, ok))

    print("\n=== SUMMARY ===")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    n_pass = sum(1 for _, ok in results if ok)
    print(f"  {n_pass}/{len(results)} passed")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
