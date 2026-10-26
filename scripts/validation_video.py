#!/usr/bin/env python3
"""Film a standing policy on the fixed showcase timeline: the validation video.

Training makes this video by itself when a run finishes (runs/<name>/validation.mp4, logged to
W&B as validation/video). This makes the same video -- the same timeline, seed and layout,
through the same function -- for a run that already exists, a checkpoint inside it, or an
exported .npz. What it shows: src/robustness/video.py; the timeline: src/robustness/showcase.py.

    python scripts/validation_video.py --run runs/<name>                   # best/, else final/
    python scripts/validation_video.py --run runs/<name> --checkpoint final
    python scripts/validation_video.py --run runs/<name> --checkpoint checkpoints/00035.00M
    python scripts/validation_video.py --npz models/kneelaw_db.npz --config config/experiments/flex_plant.yaml
    python scripts/validation_video.py --run runs/<name> --out C:/tmp/clip --size 640x360

Writes <out>.mp4 (or .webm when H.264 is unavailable) and <out>.json (event log, per-beat
outcome). Does not upload anything.
"""
import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from src.utils import configure_mujoco_gl  # noqa: E402

configure_mujoco_gl()

import numpy as np  # noqa: E402

from src.environments import feedback_prior  # noqa: E402
from src.robustness import policies  # noqa: E402
from src.robustness.video import checkpoint_label, make_validation_video, pick_checkpoint  # noqa: E402
from src.training.experiment import load_experiment  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--run", help="a runs/<name> directory")
    what.add_argument("--npz", help="an exported policy bundle (needs --config)")
    p.add_argument("--checkpoint", default="best", help="with --run: best | final | checkpoints/<dir>")
    p.add_argument("--config", help="with --npz: the config that defines the plant")
    p.add_argument("--out", help="output path without extension (default <run>/validation)")
    p.add_argument("--fps", type=float, default=20)
    p.add_argument("--size", default="960x540", help="WxH")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    if args.run:
        run = Path(args.run)
        ck = pick_checkpoint(run, args.checkpoint)
        if not (ck / "model.zip").exists():
            p.error(f"{ck} has no model.zip")
        cfg = load_experiment(str(run / "config.yaml")).task_cfg("standing")
        label = f"{run.name} . {checkpoint_label(ck)}"
        pol = policies.SB3Policy.load(ck / "model.zip", ck / "vecnorm.pkl", label=label)
        out = Path(args.out) if args.out else run / "validation"
    else:
        if not args.config:
            p.error("--npz needs --config (it defines the robot and the baseline)")
        cfg = load_experiment(args.config).task_cfg("standing")
        pol = policies.NpzPolicy(args.npz)
        # Run the bundle with the prior the robot applies, as eval_standing.py does.
        names = feedback_prior.joint_names(cfg.get("joint_map", "config/joint_servo_map.yaml"))
        prior = feedback_prior.gain_matrix(cfg.get("feedback_prior"), names)
        if pol.prior_gain is not None and prior is None:
            cfg["feedback_prior"] = pol.prior_gain.tolist()
        elif not np.array_equal(pol.prior_gain if pol.prior_gain is not None else 0,
                                prior if prior is not None else 0):
            p.error(f"{args.config} and {args.npz} disagree on the feedback prior")
        label = f"{pol.label} [{pol.fingerprint}]"
        out = Path(args.out) if args.out else Path(args.npz).with_name(Path(args.npz).stem + "_validation")
    res = make_validation_video(pol, cfg, out, label, width=w, height=h, fps=args.fps, seed=args.seed)
    print(f"video {res.path}\nlog   {res.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
