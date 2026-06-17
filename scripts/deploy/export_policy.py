#!/usr/bin/env python3
# export_policy.py
"""
Export the SB3 PPO standing policy to a torch-free NumPy bundle (.npz) so the Pi can
run inference with numpy only (no torch, no stable-baselines3, no pickle).

Bundles BOTH the policy MLP weights AND the VecNormalize obs stats, so the single .npz
is everything the deploy loop needs. The policy is a plain MLP with no output squashing
(SB3 squash_output=False), so the deterministic action is just the final linear layer.

Run on the DEV machine (needs torch+SB3), then scp the .npz to the Pi.

  python scripts/deploy/export_policy.py        # -> models/real_standing_policy.npz
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "final_real_standing_model.zip"))
    p.add_argument("--vecnorm", default=str(ROOT / "models" / "vecnorm_real_standing.pkl"))
    p.add_argument("--out", default=str(ROOT / "models" / "real_standing_policy.npz"))
    args = p.parse_args()

    import torch  # noqa: F401  (only needed here on the dev machine)
    from stable_baselines3 import PPO

    print(f"Loading {args.model}")
    model = PPO.load(args.model, device="cpu")
    pol = model.policy
    assert pol.squash_output is False, "exporter assumes no output squashing"

    # Extract the policy MLP (hidden Linear+SiLU stack) + the action mean head.
    import torch.nn as nn
    hidden = [m for m in pol.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    out = pol.action_net
    arrays = {"n_hidden": np.array(len(hidden), dtype=np.int64)}
    for i, lin in enumerate(hidden):
        arrays[f"h{i}_w"] = lin.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"h{i}_b"] = lin.bias.detach().cpu().numpy().astype(np.float32)
    arrays["out_w"] = out.weight.detach().cpu().numpy().astype(np.float32)
    arrays["out_b"] = out.bias.detach().cpu().numpy().astype(np.float32)

    # VecNormalize obs stats (so the Pi doesn't need the pkl / SB3).
    with open(args.vecnorm, "rb") as f:
        vn = pickle.load(f)
    arrays["obs_mean"] = np.asarray(vn.obs_rms.mean, dtype=np.float32)
    arrays["obs_var"] = np.asarray(vn.obs_rms.var, dtype=np.float32)
    arrays["clip_obs"] = np.array(float(getattr(vn, "clip_obs", 10.0)), dtype=np.float32)
    arrays["epsilon"] = np.array(float(getattr(vn, "epsilon", 1e-8)), dtype=np.float32)
    obs_dim = arrays["obs_mean"].shape[0]
    act_dim = arrays["out_b"].shape[0]

    # ---- parity check: numpy forward vs SB3 deterministic, on normalized inputs ----
    rng = np.random.default_rng(0)
    z = rng.standard_normal((16, obs_dim)).astype(np.float32)

    def np_forward(x):
        for i in range(len(hidden)):
            x = silu(x @ arrays[f"h{i}_w"].T + arrays[f"h{i}_b"])
        return x @ arrays["out_w"].T + arrays["out_b"]

    # Compare to SB3's UNCLIPPED Gaussian mean (the true weight-extraction check).
    # model.predict() additionally clips to the action space; deploy_standing applies
    # that same clip downstream, so validating the pre-clip mean is the right test.
    with torch.no_grad():
        dist = model.policy.get_distribution(torch.as_tensor(z))
        sb3_mean = dist.distribution.mean.cpu().numpy()
    np_act = np_forward(z)
    max_err = float(np.max(np.abs(np_act - sb3_mean)))
    print(f"parity max|numpy - sb3 mean| = {max_err:.2e}  (obs_dim={obs_dim}, act_dim={act_dim})")
    assert max_err < 1e-4, f"parity check FAILED ({max_err:.2e}) — do not deploy this npz"

    np.savez(args.out, **arrays)
    sz = Path(args.out).stat().st_size / 1e6
    print(f"OK: wrote {args.out} ({sz:.2f} MB). Parity verified. torch/SB3 not needed to run it.")


if __name__ == "__main__":
    main()
