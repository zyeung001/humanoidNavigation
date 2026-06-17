#!/usr/bin/env python3
# numpy_policy.py
"""
Pure-NumPy inference for the standing policy. No torch, no stable-baselines3, no pickle.

Loads the .npz produced by export_policy.py (policy MLP weights + VecNormalize stats) and
runs the deterministic forward pass:  normalize(obs) -> [Linear+SiLU]*N -> Linear -> action.
Owns its own obs normalization, so callers pass RAW (unnormalized) observations.
"""

from __future__ import annotations
import numpy as np


def silu(x):
    return x / (1.0 + np.exp(-x))


class NumpyPolicy:
    def __init__(self, hidden, out_w, out_b, obs_mean, obs_var, clip_obs, epsilon):
        self.hidden = hidden            # list of (W, b), W is (out,in)
        self.out_w, self.out_b = out_w, out_b
        self.obs_mean, self.obs_var = obs_mean, obs_var
        self.clip_obs, self.epsilon = float(clip_obs), float(epsilon)
        self.obs_dim = obs_mean.shape[0]
        self.act_dim = out_b.shape[0]

    @classmethod
    def load(cls, npz_path):
        d = np.load(npz_path)
        n = int(d["n_hidden"])
        hidden = [(d[f"h{i}_w"].astype(np.float32), d[f"h{i}_b"].astype(np.float32)) for i in range(n)]
        return cls(
            hidden,
            d["out_w"].astype(np.float32), d["out_b"].astype(np.float32),
            d["obs_mean"].astype(np.float32), d["obs_var"].astype(np.float32),
            float(d["clip_obs"]), float(d["epsilon"]),
        )

    def normalize(self, obs):
        z = (np.asarray(obs, np.float32) - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(z, -self.clip_obs, self.clip_obs)

    def predict(self, obs_raw):
        """RAW obs (obs_dim,) -> deterministic action (act_dim,)."""
        x = self.normalize(obs_raw)
        for w, b in self.hidden:
            x = silu(x @ w.T + b)
        return (x @ self.out_w.T + self.out_b).astype(np.float32)
