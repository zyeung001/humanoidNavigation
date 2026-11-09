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

A model trained over a feedback prior (src/environments/feedback_prior.py) needs its gains on
the robot too: they are read from --config (default: the run's config.yaml, for a model inside
runs/<name>/) and written into the bundle as `prior_gain`, which deploy_standing.py applies.

  # the prior ALONE (a network that outputs exactly zero): the controller to test on the robot
  python scripts/deploy/export_policy.py --prior-only \\
      --config config/experiments/scratch_prior.yaml --out models/real_standing_prior_only.npz

Every bundle is written through numpy_policy.stamp(): it gets `format_version` and a `requires` list
naming each runtime feature its optional keys need (feedback_prior, prior_deadband), and is loaded
back with NumpyPolicy.load before this script reports OK. A Pi whose numpy_policy.py predates a
feature then refuses the bundle by name instead of running without it (on 10/25 the gyro deadband
would have been silently ignored by an old copy).
"""

from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy_policy  # noqa: E402  (the Pi's own loader checks every bundle written here)


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "final_real_standing_model.zip"))
    p.add_argument("--vecnorm", default=str(ROOT / "models" / "vecnorm_real_standing.pkl"))
    p.add_argument("--out", default=str(ROOT / "models" / "real_standing_policy.npz"))
    p.add_argument("--config", help="training config or experiment file; its feedback_prior goes "
                                    "into the bundle (default: the run's config.yaml if the model "
                                    "is inside runs/<name>/)")
    p.add_argument("--prior-only", action="store_true",
                   help="no model: a bundle whose network outputs zero, so the robot runs the "
                        "straight pose + the --config feedback prior and nothing else")
    p.add_argument("--pose-from-config", action="store_true",
                   help="with --prior-only: output the --config's residual_baseline (e.g. the knee2 "
                        "stance) instead of the straight pose, so the law acts around the trained stance")
    args = p.parse_args()

    if args.config is None and not args.prior_only:
        run_cfg = Path(args.model).resolve().parent.parent / "config.yaml"
        args.config = str(run_cfg) if run_cfg.exists() else None
    prior = prior_gain(args.config)
    deadband = prior_deadband(args.config) if prior is not None else None
    if args.prior_only:
        if prior is None:
            p.error("--prior-only needs a --config with a feedback_prior")
        write_prior_only(args.out, prior, args.config,
                         pose=config_pose(args.config) if args.pose_from_config else None,
                         deadband=deadband)
        return

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

    if prior is not None:
        arrays["prior_gain"] = prior
        print(f"feedback prior from {args.config}: {np.count_nonzero(prior)} gains")
        if deadband is not None:
            arrays["prior_deadband"] = deadband
            print(f"  with signal deadband {deadband.tolist()}")
    elif args.config:
        print(f"no feedback prior in {args.config}")
    else:
        print("no --config and no run config next to the model: exported WITHOUT a feedback prior")
    save_bundle(args.out, arrays)
    sz = Path(args.out).stat().st_size / 1e6
    print(f"OK: wrote {args.out} ({sz:.2f} MB). Parity verified. torch/SB3 not needed to run it.")


def save_bundle(out, arrays):
    """Write `arrays` with the format/requires header (numpy_policy.stamp), then load it back with
    the Pi's loader, so a bundle this repo's own numpy_policy.py would refuse is never written."""
    np.savez(out, **numpy_policy.stamp(arrays))
    pol = numpy_policy.NumpyPolicy.load(out)
    print(f"bundle format {numpy_policy.BUNDLE_FORMAT}, requires {pol.requires} -- a Pi whose "
          f"numpy_policy.py lacks any of these refuses to load it")
    return pol


def prior_gain(config):
    """The (17, 6) feedback-prior gains a config trains with, or None."""
    if not config:
        return None
    import sys  # noqa: PLC0415
    sys.path.insert(0, str(ROOT))
    from src.environments import feedback_prior  # noqa: PLC0415
    from src.training.experiment import load_experiment  # noqa: PLC0415
    cfg = load_experiment(config).task_cfg("standing")
    names = feedback_prior.joint_names(str(ROOT / cfg.get("joint_map", "config/joint_servo_map.yaml")))
    return feedback_prior.gain_matrix(cfg.get("feedback_prior"), names)


def prior_deadband(config):
    """The (6,) signal deadband a config's prior uses (feedback_prior_deadband), or None."""
    import sys  # noqa: PLC0415
    sys.path.insert(0, str(ROOT))
    from src.environments import feedback_prior  # noqa: PLC0415
    from src.training.experiment import load_experiment  # noqa: PLC0415
    cfg = load_experiment(config).task_cfg("standing")
    return feedback_prior.deadband_vector(cfg.get("feedback_prior_deadband"))


def config_pose(config):
    """The residual_baseline (sim rad, actuator order) a config trains around."""
    import sys  # noqa: PLC0415
    sys.path.insert(0, str(ROOT))
    from src.training.experiment import load_experiment  # noqa: PLC0415
    return np.asarray(load_experiment(config).task_cfg("standing")["residual_baseline"], np.float32)


def write_prior_only(out, prior, config, obs_dim=228, pose=None, deadband=None):
    """A bundle the deploy loop runs like any policy: a constant output (the straight pose, or
    `pose`) + the prior (and its signal deadband, if the config has one)."""
    act_dim = prior.shape[0]
    out_b = np.zeros(act_dim, np.float32) if pose is None else np.asarray(pose, np.float32)
    extra = {} if deadband is None else {"prior_deadband": np.asarray(deadband, np.float32)}
    save_bundle(out, dict(n_hidden=np.array(0, dtype=np.int64),
                          out_w=np.zeros((act_dim, obs_dim), np.float32), out_b=out_b,
                          obs_mean=np.zeros(obs_dim, np.float32), obs_var=np.ones(obs_dim, np.float32),
                          clip_obs=np.array(50.0, np.float32), epsilon=np.array(1e-8, np.float32),
                          prior_gain=prior.astype(np.float32), **extra))
    what = "the straight pose" if pose is None else "the config's stance"
    print(f"OK: wrote {out}: {what} + the feedback prior from {config}, no network.")


if __name__ == "__main__":
    main()
