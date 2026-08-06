"""
Warm-start expansion: 228-dim standing model -> 239-dim turning model.

Adapts the standing->walking 1484->1495 transfer (+11 command block appended at the
END) to the real-robot turning substrate. The turning obs is

    [ 228 stacked proprioceptive dims | 11 command block ]

(see src/environments/real_turning_env.py::_process_observation, which concatenates the
command block LAST). So the 11 new dims are appended at indices 228..238.

Policy: first-layer weights `mlp_extractor.policy_net.0.weight` and
`mlp_extractor.value_net.0.weight` grow [512,228] -> [512,239]; the new 11 columns are
ZERO-init so the expanded policy ignores the command block at start (deterministic action
== original 228-model's action on the shared dims). Bias, all later layers, action_net,
value_net head, and log_std are untouched.

VecNormalize: obs_rms.mean/var (228,) -> (239,); the new 11 entries are IDENTITY
(mean 0, var 1) so the command block passes through unnormalized initially. clip_obs,
epsilon, ret_rms, count unchanged.

num_timesteps is RESET to 0 so `train_standing.py --timesteps T` reads T as an ABSOLUTE
budget (it computes remaining = T - model.num_timesteps; the standing model carries ~60M
steps which would otherwise make T cumulative and corrupt the LR/clip schedules).

Usage:
    python scripts/expand_turning_obs.py \
        --model models/final_real_standing_balanced.zip \
        --vecnorm models/vecnorm_real_standing_balanced.pkl \
        --output-model models/turn_warmstart.zip \
        --output-vecnorm models/turn_warmstart.pkl
"""

import argparse
import pickle

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

STANDING_DIM = 228
COMMAND_BLOCK_DIM = 11
TURNING_DIM = STANDING_DIM + COMMAND_BLOCK_DIM  # 239


def expand_policy(model_path: str, output_path: str, reset_timesteps: bool):
    """Append COMMAND_BLOCK_DIM zero columns to the first-layer weights."""
    print(f"  Loading model from {model_path} ...", flush=True)
    model = PPO.load(model_path, device="cpu", custom_objects={
        "learning_rate": 3e-4,
        "lr_schedule": None,
    })
    in_dim = int(model.observation_space.shape[0])
    print(f"  Model loaded. Obs space: {model.observation_space.shape}, "
          f"num_timesteps={model.num_timesteps:,}", flush=True)
    if in_dim != STANDING_DIM:
        raise ValueError(f"Expected {STANDING_DIM}-dim standing model, got {in_dim}")

    state_dict = model.policy.state_dict()
    expanded = 0
    for key, tensor in list(state_dict.items()):
        if tensor.dim() == 2 and tensor.shape[1] == STANDING_DIM:
            # First-layer weight [out, 228] -> [out, 239]; new 11 cols zero.
            new_tensor = torch.zeros(tensor.shape[0], TURNING_DIM, dtype=tensor.dtype)
            new_tensor[:, :STANDING_DIM] = tensor
            state_dict[key] = new_tensor
            expanded += 1
            print(f"  Expanded {key}: {tuple(tensor.shape)} -> {tuple(new_tensor.shape)} "
                  f"(new cols {STANDING_DIM}..{TURNING_DIM - 1} = 0)")
        elif tensor.dim() == 1 and tensor.shape[0] == STANDING_DIM:
            # Defensive: no such tensor exists in this model, but handle it.
            new_tensor = torch.zeros(TURNING_DIM, dtype=tensor.dtype)
            new_tensor[:STANDING_DIM] = tensor
            state_dict[key] = new_tensor
            expanded += 1
            print(f"  Expanded {key}: {tuple(tensor.shape)} -> {tuple(new_tensor.shape)}")

    if expanded != 2:
        print(f"  WARNING: expanded {expanded} layers (expected 2: policy_net.0 + value_net.0)")

    new_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(TURNING_DIM,), dtype=np.float32)
    print(f"  Rebuilding policy with obs space {new_obs_space.shape} ...", flush=True)
    model.observation_space = new_obs_space
    model.policy = model.policy_class(
        new_obs_space,
        model.action_space,
        model.lr_schedule,
        **model.policy_kwargs,
    )
    model.policy.load_state_dict(state_dict)
    print("  Weights loaded into rebuilt policy.", flush=True)

    if reset_timesteps:
        print(f"  Resetting num_timesteps {model.num_timesteps:,} -> 0 "
              f"(--timesteps becomes an absolute budget)")
        model.num_timesteps = 0
        model._num_timesteps_at_start = 0

    model.save(output_path)
    print(f"  Saved expanded model to {output_path}")
    return model


def expand_vecnorm(vecnorm_path: str, output_path: str):
    """Append COMMAND_BLOCK_DIM identity entries (mean 0, var 1) to obs_rms."""
    with open(vecnorm_path, "rb") as f:
        vn = pickle.load(f)  # SB3 saves with venv stripped (__getstate__), so venv is None

    old_mean = np.asarray(vn.obs_rms.mean)
    old_var = np.asarray(vn.obs_rms.var)
    print(f"  Original obs_rms shape: {old_mean.shape}, count={vn.obs_rms.count:,.0f}")
    if old_mean.shape[0] != STANDING_DIM:
        raise ValueError(f"Expected {STANDING_DIM}-dim vecnorm, got {old_mean.shape[0]}")

    new_mean = np.zeros(TURNING_DIM, dtype=old_mean.dtype)
    new_var = np.ones(TURNING_DIM, dtype=old_var.dtype)
    new_mean[:STANDING_DIM] = old_mean   # body stats copied verbatim
    new_var[:STANDING_DIM] = old_var
    # new_mean[228:] stays 0, new_var[228:] stays 1 -> command block passes through.

    vn.obs_rms.mean = new_mean
    vn.obs_rms.var = new_var
    # count, ret_rms, clip_obs, epsilon, gamma all unchanged.

    # Keep the wrapper's own observation_space consistent so VecNormalize.load's
    # shape check against the 239-dim turning env passes.
    new_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(TURNING_DIM,), dtype=np.float32)
    vn.observation_space = new_obs_space
    if vn.__dict__.get("old_obs") is not None:
        vn.old_obs = None

    # VecNormalize.__getstate__ deletes "venv" and "class_attributes"; a pickle loaded
    # from a saved file no longer carries them, so re-add placeholders before dumping
    # (VecNormalize.save / pickle.dump both go through __getstate__). Use __dict__ to
    # bypass VecEnvWrapper's attribute-forwarding __getattr__ (which recurses into the
    # None venv and raises).
    vn.__dict__.setdefault("venv", None)
    vn.__dict__.setdefault("class_attributes", {})
    vn.__dict__.setdefault("returns", np.zeros(1))

    vn.save(output_path)

    print(f"  Saved expanded VecNormalize to {output_path}")
    print(f"  Command block mean: {new_mean[STANDING_DIM:]}")
    print(f"  Command block var:  {new_var[STANDING_DIM:]}")


def main():
    p = argparse.ArgumentParser(description="Expand standing model 228 -> 239 (turning warm-start)")
    p.add_argument("--model", required=True, help="Path to 228-dim standing model .zip")
    p.add_argument("--vecnorm", required=True, help="Path to 228-dim VecNormalize .pkl")
    p.add_argument("--output-model", required=True, help="Output path for expanded model .zip")
    p.add_argument("--output-vecnorm", required=True, help="Output path for expanded VecNormalize .pkl")
    p.add_argument("--keep-timesteps", action="store_true",
                   help="Keep the standing model's num_timesteps (then --timesteps is cumulative)")
    args = p.parse_args()

    print("=" * 64)
    print(f"Turning warm-start expansion: {STANDING_DIM} -> {TURNING_DIM} obs dims")
    print(f"  Appending {COMMAND_BLOCK_DIM}-dim command block at indices "
          f"{STANDING_DIM}..{TURNING_DIM - 1}")
    print("=" * 64)

    print("\n[1/2] Expanding policy weights ...")
    expand_policy(args.model, args.output_model, reset_timesteps=not args.keep_timesteps)

    print("\n[2/2] Expanding VecNormalize stats ...")
    expand_vecnorm(args.vecnorm, args.output_vecnorm)

    print("\n" + "=" * 64)
    print("Done. Warm-start the turning training (USER runs) with e.g.:")
    print("  python scripts/train_standing.py --config config/turning/heading_torso.yaml \\")
    print(f"      --model {args.output_model} --vecnorm {args.output_vecnorm} --timesteps 20000000")
    print("  python scripts/train_standing.py --config config/turning/heading_pelvis.yaml \\")
    print(f"      --model {args.output_model} --vecnorm {args.output_vecnorm} --timesteps 20000000")
    print("=" * 64)


if __name__ == "__main__":
    main()
