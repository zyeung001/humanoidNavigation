"""Test warmstart with goal placed exactly in front (body-frame +x).

If the agent walks forward when dx0 is small and dy0=0, then the failure on
random angles is the OOD magnitude / dropped command block. If it still
spasms, the slicing itself broke the policy.
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


def run_with_fixed_goal(model_path, vn_path, gx, gy, max_steps=300):
    env_kwargs = dict(open_arena=True, open_arena_goal_dist=5.0,
                      max_episode_steps=max_steps)
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

    # Reset with deterministic seed first; then OVERRIDE the path so goal is
    # at (gx, gy) regardless of what env.reset() picked.
    obs = vn.reset()
    env.path = [(0.0, 0.0), (float(gx), float(gy))]
    env.start_pos = np.array([0.0, 0.0])
    env.goal_pos = np.array([float(gx), float(gy)])
    seg_len = float(np.hypot(gx, gy))
    env.cumulative_arc = np.array([0.0, seg_len])
    env.path_total_length = seg_len
    env.current_segment_idx = 0
    env.next_waypoint_idx = 1
    env.prev_progress = 0.0

    # Re-augment current obs with new waypoint block so first action sees
    # the correct goal direction.
    base_obs = env.env.unwrapped._get_obs()
    augmented = env._augment_obs(base_obs).astype(np.float32)
    obs = vn.normalize_obs(augmented[None, :])

    traj = []
    actions = []
    cause = "truncated"
    step = 0
    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, dones, infos = vn.step(action)
        d = env.env.unwrapped.data
        traj.append((float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2])))
        actions.append(np.asarray(action[0], dtype=np.float32))
        step += 1
        if dones[0]:
            cause = infos[0].get("termination_cause", "term")
            break
    venv.close()

    traj = np.array(traj)
    actions = np.stack(actions)
    return {
        "goal": (gx, gy),
        "cause": cause,
        "steps": step,
        "final_xy": traj[-1, :2].tolist() if len(traj) else [0, 0],
        "max_disp": float(max(np.hypot(traj[:, 0], traj[:, 1]))) if len(traj) else 0.0,
        "min_height": float(traj[:, 2].min()) if len(traj) else 0.0,
        "mean_action_abs": float(np.mean(np.abs(actions))) if actions.size else 0.0,
        "max_action_abs": float(np.max(np.abs(actions))) if actions.size else 0.0,
        "traj_first_5_xy": traj[:5, :2].tolist() if len(traj) >= 5 else [],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/nav_rebuild/warmstart_model.zip")
    p.add_argument("--vn", default="models/nav_rebuild/warmstart_vecnorm.pkl")
    args = p.parse_args()

    cases = [
        ("forward 1m", 1.0, 0.0),
        ("forward 2m", 2.0, 0.0),
        ("forward 5m", 5.0, 0.0),
        ("right 1m", 0.0, -1.0),
        ("right 2m", 0.0, -2.0),
        ("backward 5m", -5.0, 0.0),
    ]
    print(f"{'case':>15s} {'cause':>10s} {'steps':>5s} {'maxdisp':>7s} "
          f"{'finalx':>7s} {'finaly':>7s} {'minh':>5s} {'|a|mean':>7s}")
    for name, gx, gy in cases:
        r = run_with_fixed_goal(args.model, args.vn, gx, gy, max_steps=200)
        fx, fy = r["final_xy"]
        print(f"{name:>15s} {r['cause']:>10s} {r['steps']:>5d} "
              f"{r['max_disp']:>7.2f} {fx:>7.2f} {fy:>7.2f} "
              f"{r['min_height']:>5.2f} {r['mean_action_abs']:>7.3f}")
        if r["traj_first_5_xy"]:
            print(f"     first5: {r['traj_first_5_xy']}")


if __name__ == "__main__":
    main()
