"""
Navigation training entry point.

Loads a model adapted by `scripts/adapt_walking_to_nav.py` (1430 obs) or a
checkpoint from a prior nav run, and continues training in `NavRebuildEnv`.

Three environment modes (mutually exclusive at the top level):

    open arena       --maze-types open  (default; Phase 1)
    fixed maze(s)    --maze-types corridor,L,U   (cyclic across envs)
    procedural       --procedural               (fresh random maze per reset)

Procedural mode also supports `--mix-fixed corridor:1,L:1` to pin some envs
to fixed maps (used in Phase 3 to recover specific maze skills).

Usage:
    python scripts/train_nav.py \\
        --model models/nav_rebuild/warmstart_model.zip \\
        --vecnorm models/nav_rebuild/warmstart_vecnorm.pkl \\
        --timesteps 5000000 --n-envs 8
"""

import argparse
import os
import sys
import types
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import configure_mujoco_gl, get_subprocess_start_method  # noqa: E402
configure_mujoco_gl()

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.buffers import RolloutBuffer  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from src.environments.nav_rebuild_env import NavRebuildEnv  # noqa: E402
from src.maze.maze_maps import CORRIDOR, L_MAZE, U_MAZE, MEDIUM_MAZE  # noqa: E402
from src.training.metrics_logger import JsonlMetricsCallback  # noqa: E402
from src.training.transfer_callbacks import (  # noqa: E402
    LogStdClampCallback,
    ValueFunctionWarmupCallback,
)

MAZE_GRIDS = {
    "corridor": CORRIDOR,
    "L": L_MAZE,
    "U": U_MAZE,
    "medium": MEDIUM_MAZE,
}


class _SafeCheckpointCallback(CheckpointCallback):
    # VFWarmup monkey-patches model.train; that closure captures a callback
    # whose parent chain reaches JsonlMetricsCallback's open file handle,
    # which cloudpickle refuses to serialize. Strip the patch around save
    # and restore it afterwards.
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


def make_vec_env(n_envs: int, seed: int, env_kwargs_per_rank, use_subproc: bool):
    """env_kwargs_per_rank: callable rank->dict, or single dict applied to all."""
    if callable(env_kwargs_per_rank):
        kwargs_fn = env_kwargs_per_rank
    else:
        kwargs_fn = lambda _r: env_kwargs_per_rank  # noqa: E731
    if n_envs > 1 and use_subproc:
        start_method = get_subprocess_start_method()
        return SubprocVecEnv(
            [_make_env_fn(i, seed, kwargs_fn(i)) for i in range(n_envs)],
            start_method=start_method,
        )
    return DummyVecEnv([_make_env_fn(i, seed, kwargs_fn(i)) for i in range(n_envs)])


def _open_arena_kwargs(args, base_kwargs):
    kw = dict(base_kwargs, open_arena=True, open_arena_goal_dist=args.goal_dist)
    if args.goal_angle_range is not None:
        half = float(args.goal_angle_range)
        kw["open_arena_angle_range"] = (-half, half)
    return kw


def _parse_mix_fixed(spec_str: str, n_envs: int) -> dict:
    """Parse '--mix-fixed corridor:1,L:1' into {rank: maze_name}."""
    if not spec_str:
        return {}
    spec = {}
    for tok in spec_str.split(","):
        name, count = tok.split(":")
        name = name.strip()
        if name not in MAZE_GRIDS:
            raise ValueError(
                f"Unknown maze '{name}' in --mix-fixed. "
                f"Valid: {', '.join(MAZE_GRIDS.keys())}"
            )
        spec[name] = int(count)
    if sum(spec.values()) >= n_envs:
        raise ValueError(
            f"--mix-fixed total ({sum(spec.values())}) must be < n_envs "
            f"({n_envs}); leave at least one env for procedural."
        )
    fixed_assignments, rank = {}, 0
    for name, count in spec.items():
        for _ in range(count):
            fixed_assignments[rank] = name
            rank += 1
    return fixed_assignments


def _build_env_kwargs_factory(args, base_kwargs, maze_list):
    """Return (factory, mode_str). Factory is rank->kwargs or a single dict."""
    if args.procedural:
        proc_kwargs = dict(
            base_kwargs,
            open_arena=False,
            procedural=True,
            proc_rows_range=(args.proc_rows_min, args.proc_rows_max),
            proc_cols_range=(args.proc_cols_min, args.proc_cols_max),
            proc_algorithm=args.proc_algorithm,
        )
        fixed_assignments = _parse_mix_fixed(args.mix_fixed, args.n_envs)
        if not fixed_assignments:
            mode_str = (f"procedural ({args.proc_algorithm}), "
                        f"rows {args.proc_rows_min}-{args.proc_rows_max}, "
                        f"cols {args.proc_cols_min}-{args.proc_cols_max}")
            return proc_kwargs, mode_str

        def factory(rank: int):
            if rank in fixed_assignments:
                return dict(base_kwargs, open_arena=False,
                            grid=MAZE_GRIDS[fixed_assignments[rank]])
            return proc_kwargs

        dist = Counter(fixed_assignments.values())
        n_proc = args.n_envs - sum(dist.values())
        mode_str = (
            f"procedural ({args.proc_algorithm}, {n_proc} envs) + fixed: "
            + ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        )
        return factory, mode_str

    if maze_list == ["open"]:
        return _open_arena_kwargs(args, base_kwargs), \
               f"open-arena, goal_dist={args.goal_dist}m"

    rank_to_maze = [maze_list[i % len(maze_list)] for i in range(args.n_envs)]

    def factory(rank: int):
        mt = rank_to_maze[rank]
        if mt == "open":
            return _open_arena_kwargs(args, base_kwargs)
        return dict(base_kwargs, open_arena=False, grid=MAZE_GRIDS[mt])

    dist = Counter(rank_to_maze)
    mode_str = "fixed mazes — " + ", ".join(
        f"{k}:{v}" for k, v in sorted(dist.items())
    )
    return factory, mode_str


def main():
    p = argparse.ArgumentParser(description="Navigation training")
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
    p.add_argument("--maze-types", default="open",
                   help="'open' (Phase 1, default), or comma-separated list "
                        "of fixed mazes from {corridor,L,U,medium}. "
                        "Multiple mazes are distributed cyclically across "
                        "envs (e.g. 'corridor,L,U' with --n-envs 8 -> "
                        "{corridor:3, L:3, U:2}).")
    p.add_argument("--goal-dist", type=float, default=5.0,
                   help="Open-arena random goal distance from start (m).")
    p.add_argument("--goal-angle-range", type=float, default=None,
                   help="Half-width (rad) of goal angle range. Goal angle "
                        "sampled from (-X, +X). Default None = uniform "
                        "over (-pi, pi).")
    p.add_argument("--heading-cosine-weight", type=float, default=0.0,
                   help="Dense heading-cosine reward weight. 0 disables. "
                        "WARNING: survival-shaped — v4 collapsed to standing-"
                        "and-facing exploit. Prefer --vel-proj-weight.")
    p.add_argument("--vel-proj-weight", type=float, default=0.0,
                   help="Velocity-projection reward weight. Pays "
                        "dot(vel_xy, goal_unit) clipped to [0, vmax]. "
                        "Standing -> 0 (closes reverse path). 0 disables.")
    p.add_argument("--vel-proj-vmax", type=float, default=1.0,
                   help="Cap on velocity-projection reward (m/s).")
    p.add_argument("--progress-weight", type=float, default=5.0,
                   help="Path arc-length progress weight.")
    p.add_argument("--goal-bonus", type=float, default=50.0,
                   help="Terminal goal-reached bonus.")
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
    # ---- procedural maze flags ----
    p.add_argument("--procedural", action="store_true",
                   help="Sample a fresh random maze on every reset(). "
                        "Mutually exclusive with --maze-types containing fixed "
                        "mazes (procedural always overrides). Tears down + "
                        "rebuilds Humanoid-v5 env per episode (~50-200ms).")
    p.add_argument("--proc-rows-min", type=int, default=2)
    p.add_argument("--proc-rows-max", type=int, default=3)
    p.add_argument("--proc-cols-min", type=int, default=2)
    p.add_argument("--proc-cols-max", type=int, default=3)
    p.add_argument("--proc-algorithm", default="dfs", choices=["dfs", "prims"])
    p.add_argument("--mix-fixed", default="",
                   help="When --procedural is set, pin some envs to fixed mazes. "
                        "Format: 'corridor:1,L:1'. Sum must be < n_envs. "
                        "Remaining envs run procedural. Used to recover specific "
                        "maze skills lost during pure procedural training.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_kwargs = dict(
        max_episode_steps=args.max_episode_steps,
        heading_cosine_weight=args.heading_cosine_weight,
        velocity_projection_weight=args.vel_proj_weight,
        velocity_projection_vmax=args.vel_proj_vmax,
        progress_weight=args.progress_weight,
        goal_bonus=args.goal_bonus,
    )

    maze_list = [m.strip() for m in args.maze_types.split(",") if m.strip()]
    for m in maze_list:
        if m != "open" and m not in MAZE_GRIDS:
            raise ValueError(
                f"Unknown maze type '{m}'. Valid: open, "
                f"{', '.join(MAZE_GRIDS.keys())}"
            )

    env_kwargs_factory, mode_str = _build_env_kwargs_factory(args, base_kwargs, maze_list)

    print("=" * 64)
    print("Navigation training")
    print(f"  warm-start model:    {args.model}")
    print(f"  warm-start vecnorm:  {args.vecnorm}")
    print(f"  n_envs:              {args.n_envs}")
    print(f"  timesteps:           {args.timesteps:,}")
    print(f"  mode:                {mode_str}")
    print(f"  output dir:          {args.output_dir}")
    print("=" * 64)

    venv = make_vec_env(args.n_envs, args.seed, env_kwargs_factory,
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
    # (extra-VF callback in train_walking.py). Rebind to PPO.train so this
    # script uses the standard update loop; VFWarmupCallback may patch again.
    if hasattr(model, "_extra_vf_patched"):
        del model._extra_vf_patched
    if (not hasattr(model.train, "__func__")
            or model.train.__func__ is not PPO.train):
        model.train = types.MethodType(PPO.train, model)
        print("  Restored standard PPO.train (was monkey-patched)")

    # custom_objects sometimes ignores n_steps/batch_size/n_epochs when loading
    # older checkpoints. Force them from CLI and rebuild the rollout buffer.
    model.n_steps = args.n_steps
    model.batch_size = args.batch_size
    model.n_epochs = args.n_epochs
    model.gamma = args.gamma
    model.ent_coef = args.ent_coef
    model._setup_lr_schedule()
    model.rollout_buffer = RolloutBuffer(
        model.n_steps,
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=vn.num_envs,
    )

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics", "training.jsonl")

    # VFWarmupCallback first: it monkey-patches model.train and freezes the
    # policy in _on_training_start. Order matters for the patch sequence.
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

    print("\nTransfer warmup:")
    print(f"  VF-only:    0 .. {args.vf_warmup_steps:,}")
    print(f"  Policy ramp: {args.vf_warmup_steps:,} .. "
          f"{args.vf_warmup_steps + args.vf_rampup_steps:,} (0 -> "
          f"{args.policy_max_scale})")
    print(f"  Steady scale {args.policy_max_scale} thereafter")
    print("\nLogging:")
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
