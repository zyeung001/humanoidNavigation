"""Validate the 228->239 turning warm-start expansion (handoff steps 1-3)."""
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environments.real_turning_env import make_real_turning_env

ORIG_MODEL = "models/final_real_standing_balanced.zip"
EXP_MODEL = "models/turn_warmstart.zip"
EXP_VECNORM = "models/turn_warmstart.pkl"
STANDING_DIM, TURNING_DIM = 228, 239


def main():
    orig = PPO.load(ORIG_MODEL, device="cpu")
    exp = PPO.load(EXP_MODEL, device="cpu")

    # [1] shape + load
    assert exp.observation_space.shape == (TURNING_DIM,), exp.observation_space.shape
    assert orig.observation_space.shape == (STANDING_DIM,)
    print(f"[1] OK  expanded obs_space={exp.observation_space.shape}, "
          f"num_timesteps={exp.num_timesteps:,}")

    # new 11 columns of both first layers must be exactly zero
    for key in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
        w = exp.policy.state_dict()[key]
        new_cols = w[:, STANDING_DIM:]
        assert w.shape == (512, TURNING_DIM), (key, w.shape)
        assert torch.count_nonzero(new_cols) == 0, f"{key} new cols not zero"
        # shared 228 cols must be byte-identical to the original
        assert torch.equal(w[:, :STANDING_DIM],
                           orig.policy.state_dict()[key]), f"{key} shared cols changed"
    print("[1] OK  first-layer new cols zero, shared cols identical to original")

    # [2] parity: arbitrary command block -> action unchanged vs original on first 228 dims
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(200):
        body = rng.standard_normal(STANDING_DIM).astype(np.float32)
        cmd = rng.standard_normal(11).astype(np.float32) * 10.0  # arbitrary, large
        full = np.concatenate([body, cmd]).astype(np.float32)
        a_orig, _ = orig.predict(body, deterministic=True)
        a_exp, _ = exp.predict(full, deterministic=True)
        max_diff = max(max_diff, float(np.abs(a_orig - a_exp).max()))
    assert max_diff < 1e-5, f"parity failed, max action diff={max_diff}"
    print(f"[2] OK  parity over 200 random obs: max |action diff| = {max_diff:.2e}")

    # [3] smoke: real turning env + expanded vecnorm, a few steps, obs stays 239
    cfg = {
        "env_kind": "real_turning", "xml_file": "models/humanoid_real_v2.xml",
        "heading_source": "pelvis", "proprioceptive_obs": True,
        "obs_include_com": False, "obs_feature_norm": False, "obs_history": 4,
        "action_smoothing": True, "action_smoothing_tau": 0.3,
    }
    venv = DummyVecEnv([lambda: make_real_turning_env(render_mode=None, config=cfg)])
    vn = VecNormalize.load(EXP_VECNORM, venv)
    vn.training = False
    assert vn.observation_space.shape == (TURNING_DIM,), vn.observation_space.shape
    obs = vn.reset()
    assert obs.shape[1] == TURNING_DIM, obs.shape
    for _ in range(20):
        act, _ = exp.predict(obs, deterministic=True)
        obs, rew, done, info = vn.step(act)
        assert obs.shape[1] == TURNING_DIM, obs.shape
    venv.close()
    print(f"[3] OK  20 env steps via expanded model+vecnorm, obs stays {TURNING_DIM}-dim")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
