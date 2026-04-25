"""
Adapt a walking model (1495 obs) to a NavRebuildEnv warm-start (1430 obs).

Walking obs layout:
    [0:1424]    stacked body (4 frames × (350 humanoid + 6 COM))
    [1424:1435] command block (11 dims)
    [1435:1495] zero padding to "frozen_obs_dim"

NavRebuildEnv obs layout:
    [0:1424]    stacked body — IDENTICAL columns to walking
    [1424:1430] waypoint block (6 dims, body-frame dx,dy to next 3 waypoints)

Surgery:
  - Slice walking weights/biases at columns [0:1424] (body part) and append
    6 fresh zero-init columns for the waypoint block.
  - Drop walking weights at [1424:1495] (command block + dead zero-padding).
  - VecNormalize: keep obs_rms[0:1424], reset 6 new dims to mean=0/var=1.

Result: a model with 1430-dim input where body weights are unchanged from
the walking model. The waypoint->action mapping starts at zero and is learned
from the navigation reward.

Usage:
    python scripts/adapt_walking_to_nav.py \
        --model models/walking/best/model.zip \
        --vecnorm models/walking/best/vecnorm.pkl \
        --output-model models/nav_rebuild/warmstart_model.zip \
        --output-vecnorm models/nav_rebuild/warmstart_vecnorm.pkl
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pickle

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Patch SB3 VecNormalize.__getstate__ to use pop() instead of del — the
# `del state[key]` calls fail when loading older pickles where some keys
# are missing. Same fix as scripts/train_walking.py.
def _safe_getstate(self):
    state = self.__dict__.copy()
    state.pop("venv", None)
    state.pop("class_attributes", None)
    state.pop("returns", None)
    return state
VecNormalize.__getstate__ = _safe_getstate


WALKING_OBS_DIM = 1495
NAV_OBS_DIM = 1430
BODY_DIM = 1424   # stacked body columns (identical between walking and nav)
WP_DIM = 6        # waypoint block in nav


def adapt_policy(model_path: str, output_path: str):
    """Slice walking input layer to 1430 dims (body kept, command dropped)."""
    print(f"  Loading model from {model_path}...", flush=True)
    model = PPO.load(model_path, device="cpu", custom_objects={
        "learning_rate": 1e-4,
        "lr_schedule": None,
    })
    print(f"  Model obs space: {model.observation_space.shape}", flush=True)
    if model.observation_space.shape[0] != WALKING_OBS_DIM:
        raise ValueError(
            f"Expected walking model with obs dim {WALKING_OBS_DIM}, "
            f"got {model.observation_space.shape[0]}"
        )

    state_dict = model.policy.state_dict()
    adapted = 0
    for key, tensor in state_dict.items():
        # 2D weight matrices with input dim == 1495 — slice columns.
        if tensor.dim() == 2 and tensor.shape[1] == WALKING_OBS_DIM:
            new_t = torch.zeros(tensor.shape[0], NAV_OBS_DIM, dtype=tensor.dtype)
            new_t[:, :BODY_DIM] = tensor[:, :BODY_DIM]
            # [BODY_DIM:NAV_OBS_DIM] left as zero — waypoint weights, learned.
            state_dict[key] = new_t
            adapted += 1
            print(f"  Adapted {key}: {tuple(tensor.shape)} -> "
                  f"{tuple(new_t.shape)}")
        # 1D tensors with size 1495 — same column slicing.
        elif tensor.dim() == 1 and tensor.shape[0] == WALKING_OBS_DIM:
            new_t = torch.zeros(NAV_OBS_DIM, dtype=tensor.dtype)
            new_t[:BODY_DIM] = tensor[:BODY_DIM]
            state_dict[key] = new_t
            adapted += 1
            print(f"  Adapted {key}: {tuple(tensor.shape)} -> "
                  f"{tuple(new_t.shape)}")

    if adapted == 0:
        raise RuntimeError(
            "No tensors with dim 1495 found — is this really a walking model?"
        )
    print(f"  Total tensors adapted: {adapted}")

    # Rebuild policy with new obs space, then load adapted weights.
    new_obs_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(NAV_OBS_DIM,), dtype=np.float32
    )
    print(f"  Rebuilding policy with obs space ({NAV_OBS_DIM},)...", flush=True)
    model.observation_space = new_obs_space
    model.policy = model.policy_class(
        new_obs_space,
        model.action_space,
        model.lr_schedule,
        **model.policy_kwargs,
    )
    model.policy.load_state_dict(state_dict)
    print("  Weights loaded successfully", flush=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save(output_path)
    print(f"  Saved adapted model to {output_path}")
    return model


def adapt_vecnorm(vecnorm_path: str, output_path: str):
    """Slice VecNormalize obs stats to 1430 dims (body kept, command dropped)."""
    import gymnasium as gym

    with open(vecnorm_path, "rb") as f:
        old_vn = pickle.load(f)

    old_mean = np.asarray(old_vn.obs_rms.mean).copy()
    old_var = np.asarray(old_vn.obs_rms.var).copy()
    old_count = old_vn.obs_rms.count
    if old_mean.shape[0] != WALKING_OBS_DIM:
        raise ValueError(
            f"Expected VecNormalize obs_rms of shape ({WALKING_OBS_DIM},), "
            f"got {old_mean.shape}"
        )

    ret_rms_mean = (old_vn.ret_rms.mean.copy() if hasattr(old_vn.ret_rms, "mean")
                    else 0.0)
    ret_rms_var = (old_vn.ret_rms.var.copy() if hasattr(old_vn.ret_rms, "var")
                   else 1.0)
    ret_rms_count = (old_vn.ret_rms.count if hasattr(old_vn.ret_rms, "count")
                     else 1.0)

    new_mean = np.zeros(NAV_OBS_DIM, dtype=old_mean.dtype)
    new_var = np.ones(NAV_OBS_DIM, dtype=old_var.dtype)
    new_mean[:BODY_DIM] = old_mean[:BODY_DIM]
    new_var[:BODY_DIM] = old_var[:BODY_DIM]
    # [BODY_DIM:NAV_OBS_DIM] — fresh identity stats for waypoint block; the
    # navigation env produces meters in body frame, so fresh normalization
    # is appropriate.

    obs_space = spaces.Box(low=-np.inf, high=np.inf,
                           shape=(NAV_OBS_DIM,), dtype=np.float32)
    dummy_env = DummyVecEnv([lambda: gym.make("Humanoid-v5")])
    dummy_env.observation_space = obs_space

    new_vn = VecNormalize(dummy_env, norm_obs=True, norm_reward=True,
                          clip_obs=10.0, clip_reward=10.0)
    new_vn.obs_rms.mean = new_mean
    new_vn.obs_rms.var = new_var
    new_vn.obs_rms.count = old_count
    new_vn.ret_rms.mean = ret_rms_mean
    new_vn.ret_rms.var = ret_rms_var
    new_vn.ret_rms.count = ret_rms_count

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    new_vn.save(output_path)
    dummy_env.close()
    print(f"  Saved adapted VecNormalize to {output_path}")
    print(f"  Body stats kept (first 1424 dims). Waypoint stats: "
          f"mean={new_mean[BODY_DIM:].tolist()}, "
          f"var={new_var[BODY_DIM:].tolist()}")


def main():
    p = argparse.ArgumentParser(
        description="Adapt walking model (1495 obs) -> NavRebuildEnv (1430 obs)."
    )
    p.add_argument("--model", required=True,
                   help="Path to walking model .zip (1495 obs).")
    p.add_argument("--vecnorm", required=True,
                   help="Path to walking VecNormalize .pkl (1495 obs).")
    p.add_argument("--output-model", required=True,
                   help="Output path for adapted model.")
    p.add_argument("--output-vecnorm", required=True,
                   help="Output path for adapted VecNormalize.")
    args = p.parse_args()

    print("=" * 64)
    print(f"Adapting walking model: {WALKING_OBS_DIM} -> {NAV_OBS_DIM} obs dims")
    print(f"  Body columns [0:{BODY_DIM}] kept from walking")
    print(f"  Command/padding [{BODY_DIM}:{WALKING_OBS_DIM}] DROPPED")
    print(f"  Waypoint block [{BODY_DIM}:{NAV_OBS_DIM}] zero-init (will be learned)")
    print("=" * 64)

    print("\n[1/2] Adapting policy weights...")
    adapt_policy(args.model, args.output_model)

    print("\n[2/2] Adapting VecNormalize stats...")
    adapt_vecnorm(args.vecnorm, args.output_vecnorm)

    print("\n" + "=" * 64)
    print("Done. Train with:")
    print(f"  python scripts/train_nav.py \\")
    print(f"      --model {args.output_model} \\")
    print(f"      --vecnorm {args.output_vecnorm} \\")
    print("       --timesteps 5000000")
    print("=" * 64)


if __name__ == "__main__":
    main()
