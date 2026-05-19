"""Quick smoke test for procedural maze mode in NavRebuildEnv.

Times reset() cost across several episodes to verify the rebuild is
viable for training. Also runs a few steps to confirm step() still works
after the underlying env is swapped.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402

from src.utils import configure_mujoco_gl  # noqa: E402
configure_mujoco_gl()

from src.environments.nav_rebuild_env import NavRebuildEnv  # noqa: E402


def main():
    env = NavRebuildEnv(
        procedural=True,
        proc_rows_range=(2, 3),
        proc_cols_range=(2, 3),
        proc_algorithm="dfs",
        max_episode_steps=200,
        velocity_projection_weight=1.0,
        progress_weight=5.0,
        goal_bonus=50.0,
        seed=42,
    )

    n_episodes = 6
    reset_times = []
    step_times = []
    grids = []

    for ep in range(n_episodes):
        t0 = time.perf_counter()
        obs, info = env.reset()
        reset_dt = time.perf_counter() - t0
        reset_times.append(reset_dt)
        grids.append(env.grid.shape)

        # Take a few steps and time them.
        n_steps = 50
        t0 = time.perf_counter()
        for _ in range(n_steps):
            action = env.action_space.sample() * 0.1  # gentle random action
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break
        step_dt = (time.perf_counter() - t0) / n_steps
        step_times.append(step_dt)

        print(f"ep {ep}: grid={env.grid.shape} "
              f"reset={reset_dt*1000:.1f}ms "
              f"step_avg={step_dt*1000:.2f}ms "
              f"obs_dim={obs.shape[0]} "
              f"path_len={env.path_total_length:.2f}m "
              f"start={env.start_pos.tolist()} "
              f"goal={env.goal_pos.tolist()}")

    print()
    print("Summary:")
    print(f"  reset times (ms): mean={np.mean(reset_times)*1000:.1f} "
          f"min={np.min(reset_times)*1000:.1f} "
          f"max={np.max(reset_times)*1000:.1f}")
    print(f"  step times (ms):  mean={np.mean(step_times)*1000:.3f}")
    print(f"  unique grid shapes: {set(grids)}")

    env.close()


if __name__ == "__main__":
    main()
