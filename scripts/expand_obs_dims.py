"""
Expand a walking model from 1493 obs dims (9-dim command block) to 1495 (11-dim).

Inserts 2 new dims at positions 5 and 10 in the command block:
  Old: [vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual, err_vx, err_vy, err_speed, err_angle]
  New: [vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual, yaw_actual, err_vx, err_vy, err_speed, err_angle, err_yaw]

The 2 new dims (yaw_actual at index 5, err_yaw at index 10) are initialized to zero
in the policy weights and identity (mean=0, var=1) in VecNormalize.

Usage:
    python scripts/expand_obs_dims.py --model models/walking/final/final_walking_model.zip \
        --vecnorm models/walking/final/vecnorm_walking.pkl \
        --output-model models/walking/final/expanded_walking_model.zip \
        --output-vecnorm models/walking/final/expanded_vecnorm.pkl
"""

import argparse
import numpy as np
import torch
import pickle
import copy
from stable_baselines3 import PPO


def expand_policy(model_path: str, output_path: str):
    """Expand policy weights from 1493 to 1495 input dims."""
    print(f"  Loading model from {model_path}...", flush=True)
    model = PPO.load(model_path, device='cpu', custom_objects={
        "learning_rate": 3e-4,
        "lr_schedule": None,
    })
    print(f"  Model loaded. Obs space: {model.observation_space.shape}", flush=True)

    state_dict = model.policy.state_dict()

    # The command block starts at dim 1484 in the observation
    # Old command block (9 dims): indices 1484-1492
    # New command block (11 dims): indices 1484-1494
    #
    # We need to insert:
    #   yaw_actual at position 1489 (after vy_actual at 1488)
    #   err_yaw at position 1494 (at the end)

    cmd_start = 1484
    # In the old layout: [0:vx_cmd, 1:vy_cmd, 2:yaw_cmd, 3:vx_actual, 4:vy_actual,
    #                     5:err_vx, 6:err_vy, 7:err_speed, 8:err_angle]
    # Insert yaw_actual after index 4 (position 5), err_yaw at end (position 10)
    insert_positions = [cmd_start + 5, cmd_start + 10]  # 1489, 1494 (after first insert shifts indices)

    expanded = 0
    for key, tensor in state_dict.items():
        if tensor.dim() == 2 and tensor.shape[1] == 1493:
            # Input layer — expand columns
            new_tensor = torch.zeros(tensor.shape[0], 1495, dtype=tensor.dtype)

            # Copy dims 0:1489 (body + first 5 command dims)
            new_tensor[:, :cmd_start + 5] = tensor[:, :cmd_start + 5]
            # Skip new dim at 1489 (yaw_actual) — leave as zero
            # Copy dims 1489:1493 (err_vx, err_vy, err_speed, err_angle) -> 1490:1494
            new_tensor[:, cmd_start + 6:cmd_start + 10] = tensor[:, cmd_start + 5:cmd_start + 9]
            # Skip new dim at 1494 (err_yaw) — leave as zero

            state_dict[key] = new_tensor
            expanded += 1
            print(f"  Expanded {key}: {tensor.shape} -> {new_tensor.shape}")

        elif tensor.dim() == 1 and tensor.shape[0] == 1493:
            # Bias or similar 1D tensor matching obs dim
            new_tensor = torch.zeros(1495, dtype=tensor.dtype)
            new_tensor[:cmd_start + 5] = tensor[:cmd_start + 5]
            new_tensor[cmd_start + 6:cmd_start + 10] = tensor[cmd_start + 5:cmd_start + 9]
            state_dict[key] = new_tensor
            expanded += 1
            print(f"  Expanded {key}: {tensor.shape} -> {new_tensor.shape}")

    print(f"  Total layers expanded: {expanded}")

    # Rebuild policy with new obs space, then load expanded weights
    from gymnasium import spaces
    new_obs_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(1495,), dtype=np.float32
    )

    print(f"  Rebuilding policy with obs space (1495,)...", flush=True)
    model.observation_space = new_obs_space
    # Recreate policy with correct dimensions
    model.policy = model.policy_class(
        new_obs_space,
        model.action_space,
        model.lr_schedule,
        **model.policy_kwargs
    )
    # Load expanded weights into the new policy
    model.policy.load_state_dict(state_dict)
    print(f"  Weights loaded successfully", flush=True)

    model.save(output_path)
    print(f"  Saved expanded model to {output_path}")
    return model


def expand_vecnorm(vecnorm_path: str, output_path: str):
    """Expand VecNormalize stats from 1493 to 1495 dims."""
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium as gym

    # Load original stats from the raw pickle
    with open(vecnorm_path, 'rb') as f:
        old_vn = pickle.load(f)

    old_mean = old_vn.obs_rms.mean.copy()
    old_var = old_vn.obs_rms.var.copy()
    old_count = old_vn.obs_rms.count
    ret_rms_mean = old_vn.ret_rms.mean.copy() if hasattr(old_vn.ret_rms, 'mean') else 0.0
    ret_rms_var = old_vn.ret_rms.var.copy() if hasattr(old_vn.ret_rms, 'var') else 1.0
    ret_rms_count = old_vn.ret_rms.count if hasattr(old_vn.ret_rms, 'count') else 1.0

    print(f"  Original obs_rms shape: {old_mean.shape}")

    # Expand stats
    cmd_start = 1484
    new_mean = np.zeros(1495, dtype=old_mean.dtype)
    new_mean[:cmd_start + 5] = old_mean[:cmd_start + 5]
    new_mean[cmd_start + 6:cmd_start + 10] = old_mean[cmd_start + 5:cmd_start + 9]

    new_var = np.ones(1495, dtype=old_var.dtype)
    new_var[:cmd_start + 5] = old_var[:cmd_start + 5]
    new_var[cmd_start + 6:cmd_start + 10] = old_var[cmd_start + 5:cmd_start + 9]

    # Create a fresh VecNormalize with correct obs space
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1495,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)

    def make_dummy():
        env = gym.make("Humanoid-v5")
        # Override obs/act spaces for the wrapper
        return env

    dummy_env = DummyVecEnv([make_dummy])
    # Override the observation space to match our target
    dummy_env.observation_space = obs_space

    new_vn = VecNormalize(dummy_env, norm_obs=True, norm_reward=True,
                          clip_obs=10.0, clip_reward=10.0)

    # Copy expanded stats
    new_vn.obs_rms.mean = new_mean
    new_vn.obs_rms.var = new_var
    new_vn.obs_rms.count = old_count
    new_vn.ret_rms.mean = ret_rms_mean
    new_vn.ret_rms.var = ret_rms_var
    new_vn.ret_rms.count = ret_rms_count

    # Save using VecNormalize's own method
    new_vn.save(output_path)

    # Clean up
    dummy_env.close()

    print(f"  Saved expanded VecNormalize to {output_path}")
    print(f"  Command block mean: {new_mean[cmd_start:]}")
    print(f"  Command block var:  {new_var[cmd_start:]}")


def main():
    parser = argparse.ArgumentParser(description="Expand walking model obs dims 1493→1495")
    parser.add_argument('--model', required=True, help='Path to 1493-dim model .zip')
    parser.add_argument('--vecnorm', required=True, help='Path to 1493-dim VecNormalize .pkl')
    parser.add_argument('--output-model', required=True, help='Output path for expanded model')
    parser.add_argument('--output-vecnorm', required=True, help='Output path for expanded VecNormalize')
    args = parser.parse_args()

    print("=" * 60)
    print("Expanding walking model: 1493 → 1495 obs dims")
    print("  Adding: yaw_actual (dim 1489), err_yaw (dim 1494)")
    print("=" * 60)

    print("\n[1/2] Expanding policy weights...")
    expand_policy(args.model, args.output_model)

    print("\n[2/2] Expanding VecNormalize stats...")
    expand_vecnorm(args.vecnorm, args.output_vecnorm)

    print("\n" + "=" * 60)
    print("Done! Resume training with:")
    print(f"  python scripts/train_walking.py --model {args.output_model} \\")
    print(f"      --override config/variants/fresh_start.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
