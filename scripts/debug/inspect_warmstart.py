"""Inspect the warmstart policy: dump architecture, log_std, and raw mean
output before action-space clipping. Compare to walking model on walking obs.
"""
import os
import sys
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import configure_mujoco_gl  # noqa: E402
configure_mujoco_gl()

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


def main():
    print("=" * 70)
    print("WALKING MODEL")
    print("=" * 70)
    walk = PPO.load("models/walking/best/model.zip", device="cpu")
    print(f"  obs space:       {walk.observation_space.shape}")
    print(f"  action space:    {walk.action_space}")
    print(f"  action low:      {walk.action_space.low[:3]}")
    print(f"  action high:     {walk.action_space.high[:3]}")
    print(f"  log_std:         min={walk.policy.log_std.min().item():.3f}, "
          f"max={walk.policy.log_std.max().item():.3f}, "
          f"mean={walk.policy.log_std.mean().item():.3f}")
    for n, p in walk.policy.named_parameters():
        print(f"    {n}: {tuple(p.shape)}")

    print("\n" + "=" * 70)
    print("WARMSTART (ADAPTED)")
    print("=" * 70)
    ws = PPO.load("models/nav_rebuild/warmstart_model.zip", device="cpu")
    print(f"  obs space:       {ws.observation_space.shape}")
    print(f"  action space:    {ws.action_space}")
    print(f"  log_std:         min={ws.policy.log_std.min().item():.3f}, "
          f"max={ws.policy.log_std.max().item():.3f}, "
          f"mean={ws.policy.log_std.mean().item():.3f}")
    for n, p in ws.policy.named_parameters():
        print(f"    {n}: {tuple(p.shape)} "
              f"|w|mean={p.abs().mean().item():.4f} "
              f"|w|max={p.abs().max().item():.4f}")

    # ---------- compare raw policy outputs ----------
    # Build a nav-style obs: zeros for body, dx0=2 (forward 2m), rest zero.
    # Pass through warmstart's vecnorm + policy. Capture pre-clip mean.
    print("\n" + "=" * 70)
    print("RAW POLICY OUTPUT (PRE-CLIP)")
    print("=" * 70)

    # Warmstart on nav obs
    venv = DummyVecEnv([lambda: NavRebuildEnv(open_arena=True, seed=0)])
    vn = VecNormalize.load("models/nav_rebuild/warmstart_vecnorm.pkl", venv)
    vn.training = False
    obs = vn.reset()
    print(f"  nav obs shape:           {obs.shape}")
    print(f"  nav obs[0:8]:            {obs[0, :8]}")
    print(f"  nav obs[1424:1430] (wp): {obs[0, 1424:1430]}")
    print(f"  nav obs nonzero count:   {(np.abs(obs[0]) > 1e-6).sum()}")
    print(f"  nav obs |max|:           {np.abs(obs[0]).max():.3f}")

    obs_t = torch.as_tensor(obs).float()
    with torch.no_grad():
        # SB3 ActorCriticPolicy: features = extract_features → latent_pi/latent_vf
        # → action_net(latent_pi) gives mean.
        feats = ws.policy.extract_features(obs_t)
        if isinstance(feats, tuple):
            pi_feats, vf_feats = feats
        else:
            pi_feats = vf_feats = feats
        latent_pi = ws.policy.mlp_extractor.policy_net(pi_feats)
        action_mean = ws.policy.action_net(latent_pi)
        print(f"  warmstart latent_pi: shape={tuple(latent_pi.shape)} "
              f"|max|={latent_pi.abs().max().item():.3f} "
              f"|mean|={latent_pi.abs().mean().item():.3f}")
        print(f"  warmstart action_mean (pre-clip):")
        print(f"    values: {action_mean[0].numpy()}")
        print(f"    |max|={action_mean.abs().max().item():.3f}")
    venv.close()

    # Walking on walking obs (sanity check)
    print("\n  --- walking model on walking obs ---")
    from src.environments.walking_env import WalkingEnv  # noqa: E402
    venv2 = DummyVecEnv([lambda: WalkingEnv()])
    vn2 = VecNormalize.load("models/walking/best/vecnorm.pkl", venv2)
    vn2.training = False
    obs2 = vn2.reset()
    print(f"  walking obs shape:       {obs2.shape}")
    obs2_t = torch.as_tensor(obs2).float()
    with torch.no_grad():
        feats = walk.policy.extract_features(obs2_t)
        if isinstance(feats, tuple):
            pi_feats, vf_feats = feats
        else:
            pi_feats = vf_feats = feats
        latent_pi = walk.policy.mlp_extractor.policy_net(pi_feats)
        action_mean = walk.policy.action_net(latent_pi)
        print(f"  walking latent_pi: shape={tuple(latent_pi.shape)} "
              f"|max|={latent_pi.abs().max().item():.3f} "
              f"|mean|={latent_pi.abs().mean().item():.3f}")
        print(f"  walking action_mean (pre-clip):")
        print(f"    values: {action_mean[0].numpy()}")
        print(f"    |max|={action_mean.abs().max().item():.3f}")
    venv2.close()


if __name__ == "__main__":
    main()
