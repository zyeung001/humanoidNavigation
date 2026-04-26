"""
Phase 1 navigation training: open arena, single random goal, warm-started.

Loads an adapted model from `scripts/adapt_walking_to_nav.py` (1430 obs)
and continues training in `NavRebuildEnv(open_arena=True)`.

Pass criterion (per NAVIGATION_REBUILD_PLAN.md): >80% goal reach within
timeout. Train for ~5M steps as a starting point and re-evaluate.

Usage:
    python scripts/train_nav.py \\
        --model models/nav_rebuild/warmstart_model.zip \\
        --vecnorm models/nav_rebuild/warmstart_vecnorm.pkl \\
        --timesteps 5000000 \\
        --n-envs 8

For a smoke test (verifies pipeline only):
    python scripts/train_nav.py \\
        --model models/nav_rebuild/warmstart_model.zip \\
        --vecnorm models/nav_rebuild/warmstart_vecnorm.pkl \\
        --timesteps 4096 --n-envs 2 --output-dir runs/nav_smoke
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import configure_mujoco_gl, get_subprocess_start_method  # noqa: E402
configure_mujoco_gl()

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from src.environments.nav_rebuild_env import NavRebuildEnv  # noqa: E402
from src.training.metrics_logger import JsonlMetricsCallback  # noqa: E402
from src.training.transfer_callbacks import (  # noqa: E402
    LogStdClampCallback,
    ValueFunctionWarmupCallback,
)


class _SafeCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that strips VFWarmup's monkey-patched `train` from
    model.__dict__ before save. The patched closure captures the callback
    instance whose `parent.callbacks` chain reaches the JsonlMetricsCallback's
    open append-mode file, which cloudpickle refuses to serialize. Restoring
    after save keeps the extra-VF training intact across checkpoints."""

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            patched_train = self.model.__dict__.pop("train", None)
            patched_flag = self.model.__dict__.pop("_extra_vf_patched", None)
            try:
                ok = super()._on_step()
            finally:
                if patched_train is not None:
                    self.model.__dict__["train"] = patched_train
                if patched_flag is not None:
                    self.model.__dict__["_extra_vf_patched"] = patched_flag
            return ok
        return super()._on_step()


def _safe_getstate(self):
    state = self.__dict__.copy()
    state.pop("venv", None)
    state.pop("class_attributes", None)
    state.pop("returns", None)
    return state
VecNormalize.__getstate__ = _safe_getstate


def _make_env_fn(rank: int, seed: int, env_kwargs: dict):
    def _init():
        configure_mujoco_gl()
        env = NavRebuildEnv(seed=seed + rank, **env_kwargs)
        env = Monitor(env)
        return env
    return _init


def make_vec_env(n_envs: int, seed: int, env_kwargs: dict, use_subproc: bool):
    if n_envs > 1 and use_subproc:
        start_method = get_subprocess_start_method()
        return SubprocVecEnv(
            [_make_env_fn(i, seed, env_kwargs) for i in range(n_envs)],
            start_method=start_method,
        )
    return DummyVecEnv([_make_env_fn(i, seed, env_kwargs) for i in range(n_envs)])


def main():
    p = argparse.ArgumentParser(description="Phase 1 navigation training")
    p.add_argument("--model", required=True, help="Adapted warm-start model (.zip)")
    p.add_argument("--vecnorm", required=True, help="Adapted VecNormalize stats (.pkl)")
    p.add_argument("--timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="runs/nav_phase1")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--n-steps", type=int, default=2048,
                   help="Rollout length per env per update.")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--no-subproc", action="store_true",
                   help="Force DummyVecEnv (sequential). Useful for debugging.")
    p.add_argument("--save-freq", type=int, default=500_000,
                   help="Save checkpoint every N timesteps.")
    p.add_argument("--max-episode-steps", type=int, default=1500)
    p.add_argument("--goal-dist", type=float, default=5.0,
                   help="Open-arena random goal distance from start (m).")
    # ---- transfer-warmup hyperparameters (walking → nav) ----
    p.add_argument("--vf-warmup-steps", type=int, default=250_000,
                   help="Phase 1: policy frozen, VF-only training.")
    p.add_argument("--vf-rampup-steps", type=int, default=500_000,
                   help="Phase 2: policy updates ramped 0 -> max_scale.")
    p.add_argument("--policy-max-scale", type=float, default=0.5,
                   help="Phase 3: permanent scale on policy updates.")
    p.add_argument("--extra-vf-epochs", type=int, default=7,
                   help="Extra VF-only gradient steps per train() cycle.")
    p.add_argument("--log-std-max", type=float, default=0.0,
                   help="Upper clamp on policy log_std (std=1.0 at 0.0).")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    env_kwargs = dict(
        open_arena=True,
        open_arena_goal_dist=args.goal_dist,
        max_episode_steps=args.max_episode_steps,
    )

    print("=" * 64)
    print("Phase 1 navigation training (open arena, single goal)")
    print(f"  warm-start model:    {args.model}")
    print(f"  warm-start vecnorm:  {args.vecnorm}")
    print(f"  n_envs:              {args.n_envs}")
    print(f"  timesteps:           {args.timesteps:,}")
    print(f"  goal distance:       {args.goal_dist} m")
    print(f"  output dir:          {args.output_dir}")
    print("=" * 64)

    venv = make_vec_env(args.n_envs, args.seed, env_kwargs,
                        use_subproc=not args.no_subproc)
    vn = VecNormalize.load(args.vecnorm, venv)
    vn.training = True
    vn.norm_obs = True
    vn.norm_reward = True

    custom_objects = {
        "learning_rate": args.learning_rate,
        "lr_schedule": lambda _progress: args.learning_rate,
        "clip_range": lambda _progress: args.clip_range,
        "ent_coef": args.ent_coef,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
    }

    print("\nLoading adapted PPO model...")
    model = PPO.load(args.model, env=vn, custom_objects=custom_objects)
    # No TensorBoard: SummaryWriter holds a thread lock that crashes
    # CheckpointCallback's pickle path. JsonlMetricsCallback captures the
    # same train/* and rollout/* metrics into JSONL via SB3's logger.
    model.tensorboard_log = None
    print(f"  obs space: {model.observation_space.shape}")
    if model.observation_space.shape[0] != 1430:
        raise RuntimeError(
            f"Adapted model has unexpected obs dim "
            f"{model.observation_space.shape[0]} (expected 1430). "
            "Re-run scripts/adapt_walking_to_nav.py."
        )

    # The walking model may have been saved with a monkey-patched `train`
    # method (extra-VF callback in train_walking.py). Restore the bound
    # PPO.train so navigation training uses the standard update loop.
    if hasattr(model, "_extra_vf_patched"):
        try:
            del model._extra_vf_patched
        except AttributeError:
            pass
    # Always rebind train to the standard PPO method (idempotent).
    import types
    standard_train = types.MethodType(PPO.train, model)
    needs_restore = (
        not hasattr(model.train, "__func__")
        or model.train.__func__ is not PPO.train
    )
    if needs_restore:
        model.train = standard_train
        print("  Restored standard PPO.train (was monkey-patched)")

    # Force PPO hyperparams from CLI; custom_objects sometimes ignores n_steps
    # / batch_size / n_epochs when loading from older checkpoints.
    model.n_steps = args.n_steps
    model.batch_size = args.batch_size
    model.n_epochs = args.n_epochs
    model.gamma = args.gamma
    model.ent_coef = args.ent_coef
    # Rebuild rollout buffer to match n_steps × n_envs.
    model._setup_lr_schedule()
    from stable_baselines3.common.buffers import RolloutBuffer
    model.rollout_buffer = RolloutBuffer(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=vn.num_envs,
    )

    # ---------- callbacks ----------
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics", "training.jsonl")

    # VFWarmupCallback FIRST — it monkey-patches model.train and freezes
    # policy params during _on_training_start. Other callbacks can't depend
    # on those side effects, but order matters for the patch sequence.
    callbacks = [
        ValueFunctionWarmupCallback(
            warmup_steps=args.vf_warmup_steps,
            rampup_steps=args.vf_rampup_steps,
            max_scale=args.policy_max_scale,
            extra_vf_epochs=args.extra_vf_epochs,
            verbose=1,
        ),
        LogStdClampCallback(
            log_std_min=-2.0,
            log_std_max=args.log_std_max,
            clamp_freq=2000,
            verbose=1,
        ),
        _SafeCheckpointCallback(
            save_freq=max(1, args.save_freq // max(1, args.n_envs)),
            save_path=ckpt_dir,
            name_prefix="model",
            save_vecnormalize=True,
            save_replay_buffer=False,
            verbose=1,
        ),
        JsonlMetricsCallback(
            output_path=metrics_path,
            log_freq=5000,
            buffer_size=1000,
            verbose=1,
        ),
    ]

    print(f"\nTransfer warmup:")
    print(f"  VF-only:    0 .. {args.vf_warmup_steps:,}")
    print(f"  Policy ramp: {args.vf_warmup_steps:,} .. "
          f"{args.vf_warmup_steps + args.vf_rampup_steps:,} (0 -> "
          f"{args.policy_max_scale})")
    print(f"  Steady scale {args.policy_max_scale} thereafter")
    print(f"\nLogging:")
    print(f"  JSONL:       {metrics_path}")
    print(f"  Checkpoints: {ckpt_dir}  (every {args.save_freq} steps)")

    print("\nStarting training...")
    model.learn(
        total_timesteps=args.timesteps,
        log_interval=10,
        progress_bar=False,
        reset_num_timesteps=False,
        callback=callbacks,
    )

    final_model = os.path.join(args.output_dir, "model_final.zip")
    final_vn = os.path.join(args.output_dir, "vecnorm_final.pkl")
    model.save(final_model)
    vn.save(final_vn)
    print(f"\nSaved final model: {final_model}")
    print(f"Saved final vecnorm: {final_vn}")
    venv.close()


if __name__ == "__main__":
    main()
