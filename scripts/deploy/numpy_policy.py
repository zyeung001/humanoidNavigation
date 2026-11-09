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

FAIL LOUD, NEVER DROP A FEATURE. The Pi gets this file by scp, the bundle by a separate scp, so
the two drift. A loader that reads only the keys it knows would run a newer bundle with an older
copy of this file and silently skip whatever it does not know: on 10/25 the gyro deadband would
have been ignored and the robot would have twitched with nothing in the log to say why. So a
bundle names the runtime features it needs (`requires`, written by export_policy.py via stamp()),
and load() refuses a bundle that
  - requires a feature not in SUPPORTED_FEATURES (this copy of the file is older than the bundle),
  - contains a key this version does not understand (a bundle exported before `requires` existed,
    by a newer exporter -- the same drift without the list to catch it),
  - says it is a newer bundle format than BUNDLE_FORMAT.
A legacy bundle (no `requires`, format 1) whose keys are all understood still loads unchanged.
"""

from __future__ import annotations

import numpy as np

# 1 = every bundle exported before `requires` existed (no format_version key); 2 = `requires` +
# `format_version`. Bump it when the meaning of an existing key changes, not when a feature is
# added (a new feature is a new `requires` entry).
BUNDLE_FORMAT = 2

# The optional keys this version interprets, and the runtime feature each one is. A new optional
# key goes here AND into its behaviour below in the same commit, or the exporter will happily
# require a feature that this file then claims to support.
FEATURE_KEYS = {
    "prior_gain": "feedback_prior",       # K (17, 6): command += K @ (proj_grav, gyro) each frame
    "prior_deadband": "prior_deadband",   # (6,): soft dead zone on those signals before K
}
SUPPORTED_FEATURES = frozenset(FEATURE_KEYS.values())

CORE_KEYS = frozenset({"n_hidden", "out_w", "out_b", "obs_mean", "obs_var", "clip_obs", "epsilon"})
HEADER_KEYS = frozenset({"format_version", "requires"})

SYNC_HINT = ("Sync scripts/deploy/numpy_policy.py to the Pi (python scripts/robot_run.py sync) "
             "so its copy matches the exporter that wrote this bundle.")


class BundleError(ValueError):
    """The bundle needs something this copy of numpy_policy.py cannot do (or is malformed)."""


def silu(x):
    return x / (1.0 + np.exp(-x))


def bundle_features(keys) -> list:
    """The runtime features a set of npz keys needs, in FEATURE_KEYS order."""
    keys = set(keys)
    return [feat for key, feat in FEATURE_KEYS.items() if key in keys]


def stamp(arrays: dict) -> dict:
    """`arrays` + the header export_policy.py writes: format_version and the `requires` list
    naming every feature its optional keys need. Use it for EVERY bundle that is written."""
    out = {k: v for k, v in arrays.items() if k not in HEADER_KEYS}
    out["format_version"] = np.array(BUNDLE_FORMAT, dtype=np.int64)
    out["requires"] = np.array(bundle_features(out), dtype=str)
    return out


def check_bundle(d, path="bundle") -> dict:
    """Validate an opened npz (anything with .files and [key]) against this version.
    Returns {"format": int, "requires": [...], "legacy": bool}; raises BundleError."""
    keys = set(d.files)
    fmt = int(d["format_version"]) if "format_version" in keys else 1
    if fmt > BUNDLE_FORMAT:
        raise BundleError(f"{path} is bundle format {fmt}; this numpy_policy.py reads up to "
                          f"format {BUNDLE_FORMAT}. {SYNC_HINT}")
    requires = [str(x) for x in np.atleast_1d(d["requires"])] if "requires" in keys else None
    if fmt >= 2 and requires is None:
        raise BundleError(f"{path} says format {fmt} but has no `requires` list; re-export it.")

    missing = [f for f in (requires or []) if f not in SUPPORTED_FEATURES]
    if missing:
        raise BundleError(f"{path} requires {missing}, which this numpy_policy.py does not implement "
                          f"(it supports {sorted(SUPPORTED_FEATURES)}). Running it anyway would "
                          f"silently drop {'it' if len(missing) == 1 else 'them'}. {SYNC_HINT}")

    absent = sorted(CORE_KEYS - keys)
    if absent:
        raise BundleError(f"{path} is missing {absent}: not a policy bundle from export_policy.py")
    n = int(d["n_hidden"])
    layers = {f"h{i}_{p}" for i in range(n) for p in "wb"}
    if layers - keys:
        raise BundleError(f"{path} has n_hidden={n} but no {sorted(layers - keys)}")
    unknown = sorted(keys - CORE_KEYS - HEADER_KEYS - set(FEATURE_KEYS) - layers)
    if unknown:
        raise BundleError(f"{path} contains {unknown}, which this numpy_policy.py does not understand: "
                          f"loading it would ignore {'it' if len(unknown) == 1 else 'them'}. {SYNC_HINT}")

    carried = bundle_features(keys)
    if requires is not None and set(requires) != set(carried):
        raise BundleError(f"{path} requires {requires} but its keys carry {carried}; the bundle is "
                          f"inconsistent, re-export it with export_policy.py.")
    return {"format": fmt, "requires": carried if requires is None else requires,
            "legacy": requires is None}


def describe(npz_path) -> dict:
    """check_bundle() of a file on disk (what robot_run.py sync prints before copying it)."""
    with np.load(npz_path) as d:
        return check_bundle(d, npz_path)


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
        self.requires = []              # set by load(): the features the bundle asked for

    @classmethod
    def load(cls, npz_path):
        d = np.load(npz_path)
        info = check_bundle(d, npz_path)    # raises before anything runs without a feature
        n = int(d["n_hidden"])
        hidden = [(d[f"h{i}_w"].astype(np.float32), d[f"h{i}_b"].astype(np.float32)) for i in range(n)]
        pol = cls(
            hidden,
            d["out_w"].astype(np.float32), d["out_b"].astype(np.float32),
            d["obs_mean"].astype(np.float32), d["obs_var"].astype(np.float32),
            float(d["clip_obs"]), float(d["epsilon"]),
            d["prior_gain"].astype(np.float32) if "prior_gain" in d.files else None,
            d["prior_deadband"].astype(np.float32) if "prior_deadband" in d.files else None,
        )
        pol.requires = list(info["requires"])
        return pol

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
