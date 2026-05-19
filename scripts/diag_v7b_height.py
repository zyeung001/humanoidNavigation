"""Quick diagnostic: mean/min/max torso height per episode for a nav model."""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.utils import configure_mujoco_gl

configure_mujoco_gl()
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environments.nav_rebuild_env import NavRebuildEnv
from src.maze.maze_maps import CORRIDOR, L_MAZE, U_MAZE


def _safe_getstate(self):
    s = self.__dict__.copy()
    for k in ("venv", "class_attributes", "returns"):
        s.pop(k, None)
    return s


VecNormalize.__getstate__ = _safe_getstate

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--vecnorm", required=True)
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--maze-type", default="corridor")
args = p.parse_args()

grids = {"corridor": CORRIDOR, "L": L_MAZE, "U": U_MAZE}
if args.maze_type == "procedural":
    kw = dict(procedural=True, proc_rows_range=(2, 3), proc_cols_range=(2, 3), max_episode_steps=1500)
else:
    kw = dict(grid=grids[args.maze_type], max_episode_steps=1500)


def _make():
    configure_mujoco_gl()
    return NavRebuildEnv(seed=1000, **kw)


venv = DummyVecEnv([_make])
vn = VecNormalize.load(args.vecnorm, venv)
vn.training = False
vn.norm_reward = False
model = PPO.load(args.model, env=vn, device="cpu")
base = venv.envs[0]

print(f"{'ep':>3} {'cause':>10} {'steps':>5} {'mean_h':>7} {'min_h':>6} {'p10_h':>6} {'p50_h':>6}")
all_heights = []
for k in range(args.episodes):
    base._rng = np.random.RandomState(1000 + k)
    obs = vn.reset()
    heights = []
    cause = "truncated"
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, d, infos = vn.step(a)
        h = float(base.env.unwrapped.data.qpos[2])
        heights.append(h)
        if d[0]:
            cause = infos[0].get("termination_cause", "truncated")
            break
    h = np.array(heights)
    all_heights.extend(heights)
    print(f"{k:>3} {cause:>10} {len(heights):>5} {h.mean():>7.3f} {h.min():>6.3f} {np.percentile(h,10):>6.3f} {np.percentile(h,50):>6.3f}")

H = np.array(all_heights)
print("\noverall  mean={:.3f}  min={:.3f}  p10={:.3f}  p50={:.3f}  p90={:.3f}".format(
    H.mean(), H.min(), np.percentile(H, 10), np.percentile(H, 50), np.percentile(H, 90)))
