"""
Modal training launcher for humanoid navigation.

Usage:
    # Install & authenticate (one-time)
    pip install modal
    modal setup

    # Create WandB secret (one-time)
    modal secret create wandb-secret WANDB_API_KEY=<your-key>

    # Train standing (fresh)
    modal run train_modal.py --task standing --timesteps 5000000

    # Train standing (resume from checkpoint)
    modal run train_modal.py --task standing --timesteps 5000000 --model models/best_standing_model.zip

    # Train walking (from standing model)
    modal run train_modal.py --task walking --from-standing --model models/best_standing_model.zip --timesteps 30000000

    # Train walking (resume walking checkpoint)
    modal run train_modal.py --task walking --model models/walking/best/model.zip --vecnorm models/walking/best/vecnorm.pkl --timesteps 30000000

    # Train maze
    modal run train_modal.py --task maze --walking-model models/walking/best/model.zip --timesteps 50000000

    # Run detached (keeps running after terminal closes)
    modal run --detach train_modal.py --task walking --from-standing --model models/best_standing_model.zip

    # Download results after training
    modal volume ls humanoid-training
    modal volume get humanoid-training /results/ ./modal_results/
"""

import modal
import os
import subprocess

# ---------------------------------------------------------------------------
# Modal app & infrastructure
# ---------------------------------------------------------------------------

app = modal.App("humanoid-navigation")

volume = modal.Volume.from_name("humanoid-training", create_if_missing=True)

VOLUME_PATH = "/mnt/training"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "libegl1",
        "libgles2",
        "libgl1-mesa-glx",
        "libosmesa6-dev",
    ])
    .pip_install([
        "gymnasium[mujoco]",
        "stable-baselines3>=2.0.0",
        "torch>=2.0.0",
        "opencv-python-headless",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "wandb",
    ])
    .env({
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONIOENCODING": "utf-8",
        "LANG": "C.UTF-8",
    })
    # Mount project source code
    .add_local_python_source("src")
    .add_local_dir("./config", remote_path="/root/config")
    .add_local_dir("./scripts", remote_path="/root/scripts")
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Scale n_envs to match available cores. MuJoCo envs are single-threaded,
# so ~3 envs per physical core is a reasonable ratio.
ENVS_PER_CORE = 3

# Default core count (override with --cores)
DEFAULT_CORES = 32


def _upload_local_file(local_path: str, vol_dest: str):
    """Upload a local file to the volume via a helper function."""
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Read locally, write inside the remote function
    with open(local_path, "rb") as f:
        data = f.read()

    upload_file_to_volume.remote(data, vol_dest)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=60,
)
def upload_file_to_volume(data: bytes, dest_path: str):
    """Write bytes to a path on the volume."""
    full = os.path.join(VOLUME_PATH, dest_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)
    volume.commit()
    print(f"  Uploaded -> {dest_path} ({len(data):,} bytes)")


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=60,
)
def list_volume(path: str = ""):
    """List files on the volume."""
    target = os.path.join(VOLUME_PATH, path)
    if not os.path.exists(target):
        print(f"Path does not exist: {path}")
        return
    for root, dirs, files in os.walk(target):
        level = root.replace(target, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            print(f"{sub_indent}{f} ({size:,} bytes)")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=DEFAULT_CORES,
    memory=32768,  # 32 GiB
    timeout=86400,  # 24 hours
    retries=modal.Retries(max_retries=3, initial_delay=1.0, backoff_coefficient=1.0),
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    task: str,
    timesteps: int,
    cores: int,
    n_envs: int,
    from_standing: bool = False,
    model_path: str = None,
    vecnorm_path: str = None,
    walking_model_path: str = None,
    walking_vecnorm_path: str = None,
    extra_args: str = "",
):
    """Run training inside a Modal container."""
    import sys
    import shutil

    # Add project paths
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")

    results_dir = os.path.join(VOLUME_PATH, "results", task)
    os.makedirs(results_dir, exist_ok=True)

    # Symlink output directories so scripts save to the volume
    for subdir in ["models", "data"]:
        vol_sub = os.path.join(results_dir, subdir)
        local_sub = os.path.join("/root", subdir)
        os.makedirs(vol_sub, exist_ok=True)
        if os.path.exists(local_sub):
            shutil.rmtree(local_sub)
        os.symlink(vol_sub, local_sub)

    # If model files were uploaded to the volume, resolve their paths
    def resolve_vol_path(p):
        if p is None:
            return None
        vol_p = os.path.join(VOLUME_PATH, "uploads", p)
        if os.path.exists(vol_p):
            return vol_p
        # Also check results from previous runs
        for check in [
            os.path.join(results_dir, p),
            os.path.join(VOLUME_PATH, "results", p),
            os.path.join(VOLUME_PATH, p),
        ]:
            if os.path.exists(check):
                return check
        return vol_p  # Return expected path (will error later if missing)

    # Build command
    if task == "standing":
        cmd = [sys.executable, "/root/scripts/train_standing.py"]
        cmd += ["--timesteps", str(timesteps)]
        if model_path:
            cmd += ["--model", resolve_vol_path(model_path)]
        if vecnorm_path:
            cmd += ["--vecnorm", resolve_vol_path(vecnorm_path)]

    elif task == "walking":
        cmd = [sys.executable, "/root/scripts/train_walking.py"]
        cmd += ["--timesteps", str(timesteps)]
        cmd += ["--n-envs", str(n_envs)]
        if from_standing and model_path:
            cmd += ["--from-standing", "--model", resolve_vol_path(model_path)]
        elif model_path:
            cmd += ["--model", resolve_vol_path(model_path)]
        if vecnorm_path:
            cmd += ["--vecnorm", resolve_vol_path(vecnorm_path)]

    elif task == "maze":
        cmd = [sys.executable, "/root/scripts/train_maze.py"]
        cmd += ["--timesteps", str(timesteps)]
        wm = walking_model_path or model_path
        if wm:
            cmd += ["--walking-model", resolve_vol_path(wm)]
        if walking_vecnorm_path or vecnorm_path:
            wv = walking_vecnorm_path or vecnorm_path
            cmd += ["--walking-vecnorm", resolve_vol_path(wv)]
        if n_envs:
            cmd += ["--n-envs", str(n_envs)]

    else:
        raise ValueError(f"Unknown task: {task}")

    # Append any extra CLI args
    if extra_args:
        cmd += extra_args.split()

    print("=" * 60)
    print(f"MODAL TRAINING: {task.upper()}")
    print("=" * 60)
    print(f"  Cores: {cores}")
    print(f"  n_envs: {n_envs}")
    print(f"  Timesteps: {timesteps:,}")
    print("  Device: cpu")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Results: {results_dir}")
    print("=" * 60)

    # Force CPU device (no GPU on this container)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Patch config to use correct n_envs and cpu device
    import yaml
    config_path = "/root/config/training_config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if task in cfg:
        cfg[task]["n_envs"] = n_envs
        cfg[task]["device"] = "cpu"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Config patched: n_envs={n_envs}, device=cpu")

    # Run training
    result = subprocess.run(cmd, cwd="/root")

    # Commit results to volume
    volume.commit()
    print(f"\nTraining exited with code {result.returncode}")
    print(f"Results saved to volume at: results/{task}/")

    return result.returncode


# ---------------------------------------------------------------------------
# Local entrypoint — runs on YOUR machine, dispatches to Modal
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    task: str = "walking",
    timesteps: int = None,
    cores: int = DEFAULT_CORES,
    n_envs: int = None,
    from_standing: bool = False,
    model: str = None,
    vecnorm: str = None,
    walking_model: str = None,
    walking_vecnorm: str = None,
    extra_args: str = "",
):
    # Default timesteps per task
    if timesteps is None:
        defaults = {"standing": 5_000_000, "walking": 30_000_000, "maze": 50_000_000}
        timesteps = defaults.get(task, 10_000_000)

    # Scale n_envs to cores if not specified
    if n_envs is None:
        n_envs = cores * ENVS_PER_CORE

    print(f"Task: {task}")
    print(f"Cores: {cores}, n_envs: {n_envs}")
    print(f"Timesteps: {timesteps:,}")

    # Upload model files to volume if they exist locally
    files_to_upload = []
    if model and os.path.exists(model):
        files_to_upload.append((model, f"uploads/{model}"))
    if vecnorm and os.path.exists(vecnorm):
        files_to_upload.append((vecnorm, f"uploads/{vecnorm}"))
    if walking_model and os.path.exists(walking_model):
        files_to_upload.append((walking_model, f"uploads/{walking_model}"))
    if walking_vecnorm and os.path.exists(walking_vecnorm):
        files_to_upload.append((walking_vecnorm, f"uploads/{walking_vecnorm}"))

    # Also upload matching vecnorm if model is provided but vecnorm isn't
    if model and not vecnorm:
        # Try common vecnorm locations
        model_dir = os.path.dirname(model)
        for candidate in [
            os.path.join(model_dir, "vecnorm.pkl"),
            os.path.join(model_dir, "vecnorm_walking.pkl"),
            model.replace("model.zip", "vecnorm.pkl"),
            "models/vecnorm.pkl",
        ]:
            if os.path.exists(candidate):
                files_to_upload.append((candidate, f"uploads/{candidate}"))
                print(f"  Auto-detected vecnorm: {candidate}")
                break

    if files_to_upload:
        print(f"\nUploading {len(files_to_upload)} file(s) to Modal volume...")
        for local_path, vol_dest in files_to_upload:
            with open(local_path, "rb") as f:
                data = f.read()
            upload_file_to_volume.remote(data, vol_dest)

    # Dispatch training
    print(f"\nStarting training on Modal ({cores} cores)...")
    exit_code = train.remote(
        task=task,
        timesteps=timesteps,
        cores=cores,
        n_envs=n_envs,
        from_standing=from_standing,
        model_path=model,
        vecnorm_path=vecnorm,
        walking_model_path=walking_model,
        walking_vecnorm_path=walking_vecnorm,
        extra_args=extra_args,
    )

    if exit_code == 0:
        print("\nTraining complete!")
        print("Download results with:")
        print(f"  modal volume get humanoid-training /results/{task}/ ./modal_results/{task}/")
    else:
        print(f"\nTraining failed with exit code {exit_code}")
        print("Check logs in the Modal dashboard.")
