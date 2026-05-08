"""Diagnose warmstart vs trained nav models — probe per-step what the policy
actually does on the open-arena task.

Logs per episode:
  - goal world angle, body-frame waypoint magnitudes
  - position trajectory, height trajectory
  - mean |action|, action saturation
  - termination cause + step

Compares (warmstart, v3 final) on a fixed set of seeds so we can see whether
the policy degraded during training or was broken from step 0.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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


def rollout(model_path: str, vn_path: str, seeds, goal_dist=5.0, max_steps=300):
    env_kwargs = dict(
        open_arena=True,
        open_arena_goal_dist=goal_dist,
        max_episode_steps=max_steps,
    )
    base = {"env": None}

    def _factory():
        configure_mujoco_gl()
        e = NavRebuildEnv(seed=0, **env_kwargs)
        base["env"] = e
        return e

    venv = DummyVecEnv([_factory])
    vn = VecNormalize.load(vn_path, venv)
    vn.training = False
    vn.norm_reward = False

    model = PPO.load(model_path, env=vn, device="cpu")
    env = base["env"]

    results = []
    for seed in seeds:
        env._rng = np.random.RandomState(seed)
        obs = vn.reset()

        gx, gy = env.goal_pos
        goal_world_angle = float(np.degrees(np.arctan2(gy, gx)))

        traj_xy, heights, actions = [], [], []
        wp_mags = []
        cause = "truncated"
        step = 0
        while step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, dones, infos = vn.step(action)

            d = env.env.unwrapped.data
            xy = (float(d.qpos[0]), float(d.qpos[1]))
            traj_xy.append(xy)
            heights.append(float(d.qpos[2]))
            actions.append(np.asarray(action[0], dtype=np.float32))

            wp = env._waypoint_block()
            wp_mags.append(float(np.hypot(wp[0], wp[1])))

            step += 1
            if dones[0]:
                cause = infos[0].get("termination_cause", "term")
                break

        actions_arr = np.stack(actions) if actions else np.zeros((0, 17))
        traj = np.array(traj_xy) if traj_xy else np.zeros((0, 2))
        results.append({
            "seed": seed,
            "goal_angle_deg": goal_world_angle,
            "cause": cause,
            "steps": step,
            "final_xy": traj[-1].tolist() if len(traj) else [0, 0],
            "max_disp": float(np.linalg.norm(traj[-1])) if len(traj) else 0.0,
            "min_height": float(min(heights)) if heights else 0.0,
            "max_action_abs": float(np.max(np.abs(actions_arr))) if actions_arr.size else 0.0,
            "mean_action_abs": float(np.mean(np.abs(actions_arr))) if actions_arr.size else 0.0,
            "action_saturation": float(np.mean(np.abs(actions_arr) > 0.95)) if actions_arr.size else 0.0,
            "wp_mag_first": wp_mags[0] if wp_mags else 0.0,
            "wp_mag_last": wp_mags[-1] if wp_mags else 0.0,
        })

    venv.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmstart-model", default="models/nav_rebuild/warmstart_model.zip")
    p.add_argument("--warmstart-vn", default="models/nav_rebuild/warmstart_vecnorm.pkl")
    p.add_argument("--final-model", default="runs/nav_phase1_v3/model_final.zip")
    p.add_argument("--final-vn", default="runs/nav_phase1_v3/vecnorm_final.pkl")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--goal-dist", type=float, default=5.0)
    args = p.parse_args()

    seeds = list(range(1000, 1000 + args.n))

    print("=" * 70)
    print(f"WARMSTART rollout — {args.warmstart_model}")
    print("=" * 70)
    rs = rollout(args.warmstart_model, args.warmstart_vn, seeds,
                 args.goal_dist, args.max_steps)
    print(f"  {'seed':>4} {'goal_deg':>9} {'cause':>10} {'steps':>5} "
          f"{'disp':>5} {'h_min':>5} {'|a|max':>7} {'|a|mean':>7} {'sat':>5}")
    for r in rs:
        print(f"  {r['seed']:>4} {r['goal_angle_deg']:>9.1f} "
              f"{r['cause']:>10s} {r['steps']:>5d} {r['max_disp']:>5.2f} "
              f"{r['min_height']:>5.2f} {r['max_action_abs']:>7.3f} "
              f"{r['mean_action_abs']:>7.3f} {r['action_saturation']:>5.2f}")

    print("\n" + "=" * 70)
    print(f"V3 FINAL rollout — {args.final_model}")
    print("=" * 70)
    rs = rollout(args.final_model, args.final_vn, seeds,
                 args.goal_dist, args.max_steps)
    print(f"  {'seed':>4} {'goal_deg':>9} {'cause':>10} {'steps':>5} "
          f"{'disp':>5} {'h_min':>5} {'|a|max':>7} {'|a|mean':>7} {'sat':>5}")
    for r in rs:
        print(f"  {r['seed']:>4} {r['goal_angle_deg']:>9.1f} "
              f"{r['cause']:>10s} {r['steps']:>5d} {r['max_disp']:>5.2f} "
              f"{r['min_height']:>5.2f} {r['max_action_abs']:>7.3f} "
              f"{r['mean_action_abs']:>7.3f} {r['action_saturation']:>5.2f}")


if __name__ == "__main__":
    main()
