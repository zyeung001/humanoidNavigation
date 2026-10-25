#!/usr/bin/env python3
# numpy_policy.py
"""
Pure-NumPy inference for the standing policy. No torch, no stable-baselines3, no pickle.

Loads the .npz produced by export_policy.py (policy MLP weights + VecNormalize stats) and
runs the deterministic forward pass:  normalize(obs) -> [Linear+SiLU]*N -> Linear -> action.
Owns its own obs normalization, so callers pass RAW (unnormalized) observations.

A policy trained over a feedback prior (src/environments/feedback_prior.py) carries its gains
as `prior_gain` (17 x 6). predict() is the network ALONE; the caller adds prior_term(frame) to
the command, as StandingEnv does in sim. None = no prior (every bundle exported before 10/17).
`prior_deadband` (6,), if present, is the soft dead zone applied to each signal first (10/25).
"""

from __future__ import annotations
import numpy as np


def silu(x):
    return x / (1.0 + np.exp(-x))


class NumpyPolicy:
    def __init__(self, hidden, out_w, out_b, obs_mean, obs_var, clip_obs, epsilon, prior_gain=None,
                 prior_deadband=None):
        self.hidden = hidden            # list of (W, b), W is (out,in)
        self.prior_gain = prior_gain
        self.prior_deadband = prior_deadband
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
            d["prior_gain"].astype(np.float32) if "prior_gain" in d.files else None,
            d["prior_deadband"].astype(np.float32) if "prior_deadband" in d.files else None,
        )

    def normalize(self, obs):
        z = (np.asarray(obs, np.float32) - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(z, -self.clip_obs, self.clip_obs)

    def prior_term(self, frame):
        """K @ (proj_grav, gyro) of the newest per-frame feature; zeros without a prior."""
        if self.prior_gain is None:
            return np.zeros(self.act_dim, dtype=np.float32)
        s = np.asarray(frame, np.float32)[:self.prior_gain.shape[1]]
        if self.prior_deadband is not None:     # = feedback_prior.apply_deadband (sim)
            s = np.sign(s) * np.maximum(np.abs(s) - self.prior_deadband, 0.0)
        return (self.prior_gain @ s).astype(np.float32)

    def predict(self, obs_raw):
        """RAW obs (obs_dim,) -> deterministic action (act_dim,)."""
        x = self.normalize(obs_raw)
        for w, b in self.hidden:
            x = silu(x @ w.T + b)
        return (x @ self.out_w.T + self.out_b).astype(np.float32)
