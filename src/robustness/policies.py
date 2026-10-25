"""Controllers the evaluator can run. Each maps a RAW observation to a RAW action.

Raw on both sides, because that is what the robot does: deploy_standing.py normalises the
observation itself and clips the action itself. Going through SB3's predict() instead clips
to the action space stored in the .zip, which is how four joints parked past their stops
went unnoticed until 9/24.

  SB3Policy       a .zip + its VecNormalize .pkl, or the live model during training
  NpzPolicy       the exported .npz, run by scripts/deploy/numpy_policy.py -- the robot's code
  HoldPosePolicy  no learning at all: command the residual baseline every step. The bar any
                  trained policy has to clear; if it does not beat this, it is not helping.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

PROJ = Path(__file__).resolve().parents[2]


def file_fingerprint(path) -> str:
    """First 12 hex of the file's sha256: names the weights, not the path. The same path has
    held a dozen different models over this project's life."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


class _Normalizer:
    def __init__(self, mean, var, clip_obs, epsilon):
        self.mean = np.asarray(mean, dtype=np.float32).copy()
        self.std = np.sqrt(np.asarray(var, dtype=np.float32) + float(epsilon)).astype(np.float32)
        self.clip = float(clip_obs)

    def __call__(self, obs):
        return np.clip((np.asarray(obs, np.float32) - self.mean) / self.std, -self.clip, self.clip)


class SB3Policy:
    def __init__(self, sb3_policy, normalizer: _Normalizer, label: str, fingerprint: str,
                 recipe=None):
        self.policy = sb3_policy
        self.norm = normalizer
        self.label = label
        self.fingerprint = fingerprint
        self.obs_dim = int(normalizer.mean.shape[0])
        # How to rebuild this policy in another process (parallel evaluation). None for a
        # live training snapshot -- save a checkpoint and load that instead.
        self.recipe = recipe

    @classmethod
    def load(cls, model_path, vecnorm_path, label: Optional[str] = None) -> "SB3Policy":
        from stable_baselines3 import PPO  # noqa: PLC0415
        # Schedules are pickled closures; replacing them avoids unpickling problems across
        # SB3 versions, and evaluation never trains.
        model = PPO.load(str(model_path), device="cpu", custom_objects={
            "learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0})
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        norm = _Normalizer(vn.obs_rms.mean, vn.obs_rms.var, getattr(vn, "clip_obs", 10.0),
                           getattr(vn, "epsilon", 1e-8))
        label = label or Path(model_path).stem
        return cls(model.policy, norm, label, file_fingerprint(model_path),
                   recipe=("sb3", str(model_path), str(vecnorm_path), label))

    @classmethod
    def from_training(cls, model, vec_normalize, label: str) -> "SB3Policy":
        """Snapshot of the model being trained: its policy and a COPY of the current stats."""
        rms = vec_normalize.obs_rms
        norm = _Normalizer(rms.mean, rms.var, vec_normalize.clip_obs, vec_normalize.epsilon)
        return cls(model.policy, norm, label, f"step{int(model.num_timesteps)}")

    def act(self, obs: np.ndarray) -> np.ndarray:
        import torch  # noqa: PLC0415
        z = torch.as_tensor(self.norm(obs)[None], device=self.policy.device)
        with torch.no_grad():
            a = self.policy.get_distribution(z).mode()
        return a[0].cpu().numpy().astype(np.float32)


def _numpy_policy_module():
    path = PROJ / "scripts" / "deploy" / "numpy_policy.py"
    spec = importlib.util.spec_from_file_location("numpy_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class NpzPolicy:
    """The exported bundle, run by the exact module the Pi imports."""

    def __init__(self, npz_path, label: Optional[str] = None):
        self._p = _numpy_policy_module().NumpyPolicy.load(str(npz_path))
        self.label = label or Path(npz_path).stem
        self.fingerprint = file_fingerprint(npz_path)
        self.obs_dim = int(self._p.obs_dim)
        self.recipe = ("npz", str(npz_path), self.label)
        # Gains of the feedback prior it was trained over (applied by the env, not by act()),
        # or None. eval_standing.py puts them in the env config so the npz runs as on the robot.
        self.prior_gain = self._p.prior_gain

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._p.predict(obs)


class HoldPosePolicy:
    """Command one fixed pose forever: standing with no feedback at all. The residual baseline
    is what the robot holds open-loop on hardware; all-zeros (straight) is what an untrained
    network outputs, since SB3 initialises the action head near zero."""

    def __init__(self, baseline, label: str = "hold-baseline"):
        self.baseline = np.asarray(baseline, dtype=np.float32).copy()
        self.label = label
        self.fingerprint = label        # no weights to hash; the pose IS the controller
        self.obs_dim = None
        self.recipe = ("hold", self.baseline.tolist(), label)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.baseline.copy()


def _posture_module():
    path = PROJ / "scripts" / "deploy" / "posture_control.py"
    spec = importlib.util.spec_from_file_location("posture_control", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PosturePolicy:
    """A fixed stance plus the slow posture correction the robot runs
    (scripts/deploy/posture_control.py, the same file). Stateful: reset() every episode."""

    FRAME = 57          # newest frame = the last 57 obs values; projected gravity comes first

    def __init__(self, stance, posture_cfg: dict, dt: float = 0.025, label: str = "posture"):
        self.stance = np.asarray(stance, dtype=np.float32).copy()
        self.cfg = dict(posture_cfg)
        self.dt = float(dt)
        self._c = _posture_module().PostureController.from_config(self.cfg, dt=self.dt,
                                                                   n_joints=len(self.stance))
        self.label = label
        self.fingerprint = label
        self.obs_dim = None
        self.recipe = ("posture", self.stance.tolist(), self.cfg, self.dt, label)

    def reset(self) -> None:
        self._c.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        pg = np.asarray(obs, dtype=np.float32)[-self.FRAME:][:3]
        return self.stance + self._c.offsets(pg)


def rebuild(recipe):
    """The inverse of `.recipe`: construct the same policy in this process."""
    kind, *args = recipe
    if kind == "sb3":
        return SB3Policy.load(args[0], args[1], label=args[2])
    if kind == "npz":
        return NpzPolicy(args[0], label=args[1])
    if kind == "hold":
        return HoldPosePolicy(args[0], label=args[1])
    if kind == "posture":
        return PosturePolicy(args[0], args[1], dt=args[2], label=args[3])
    raise ValueError(f"unknown policy recipe {kind!r}")
