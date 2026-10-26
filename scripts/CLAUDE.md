# scripts/ — Entry Points

## Training Scripts

### `train_standing.py`

Train the standing (balance) controller. `--config` takes a plain config or an experiment file
(`config/experiments/`). Every run writes `runs/<experiment>-<date>/` (resolved config,
checkpoints, `best/` by robustness score, `final/`, logs); plain configs also still write their
legacy `final_model_path`/`vecnormalize_path`.

```bash
# Experiment (resume model comes from the file), W&B on
python scripts/train_standing.py --config config/experiments/stand_pushes.yaml --wandb online
# Check an experiment builds and see its starting score, without training
python scripts/train_standing.py --config config/experiments/scratch_history8.yaml --dry-run
# Plain config, as before
python scripts/train_standing.py --config config/real_humanoid_stand.yaml \
    --model models/final_real_standing_stand.zip --vecnorm models/vecnorm_real_standing_stand.pkl \
    --timesteps 215000000
```

Key args: `--timesteps` (cumulative), `--model`/`--vecnorm`, `--wandb online|offline|disabled`,
`--seed`, `--eval-every N` (0 = off), `--dry-run`, `--run-name`. Ctrl+C saves to
`runs/<name>/interrupted/` (never over the legacy final path).

### `eval_standing.py` / `compare_models.py` — which model is best

```bash
python scripts/eval_standing.py --npz models/real_standing_stand_195M.npz --config config/real_humanoid_stand.yaml --label stand-195M
python scripts/eval_standing.py --run runs/<name>            # its best/ checkpoint
python scripts/eval_standing.py --hold-pose straight --config config/real_humanoid_stand.yaml   # no-feedback reference
python scripts/compare_models.py                             # ranked, paired vs the best no-feedback pose
python scripts/compare_models.py --run-history runs/<name>   # one run over time
```

`eval_standing.py` runs the `src/robustness` suite (~1 min, 11 workers) and records JSON to
`models/standing/evals/` + a row in `models/standing/robustness.csv`. `compare_models.py` only
reads results. The old `models/standing/scoreboard.csv` (9/19 harness) is history; its rows
are not comparable with the new score. `gate_standing_kick.py` is superseded (binary, every
model passed it).

### `train_walking.py`

Train the walking controller, typically via transfer from a standing model.

```bash
# From standing (recommended)
python scripts/train_walking.py --from-standing \
    --model models/best_standing_model.zip \
    --timesteps 30000000

# Resume walking training
python scripts/train_walking.py \
    --model models/walking/final/final_walking_model.zip \
    --vecnorm models/walking/final/vecnorm_walking.pkl \
    --timesteps 30000000
```

Key args: `--from-standing` (enable transfer), `--model`, `--vecnorm`, `--timesteps`.

Adds `CommandStatsProtectorCallback` (re-pins command stats every 10k steps) and `WalkingMetricsCallback` (curriculum progress every 50 episodes). Uses `WalkingCurriculumEnv` with 3 stages.

## Model Surgery

### `expand_obs_dims.py`

Expands a saved walking model from 1493 obs dims (9-dim command block) to 1495 (11-dim) by inserting `yaw_actual` and `err_yaw` into the command block. Operates on both the policy weights (zero-initialized new columns) and the VecNormalize stats (identity stats for new dims).

```bash
python scripts/expand_obs_dims.py \
    --model models/walking/final/final_walking_model.zip \
    --vecnorm models/walking/final/vecnorm_walking.pkl \
    --output-model models/walking/final/expanded_walking_model.zip \
    --output-vecnorm models/walking/final/expanded_vecnorm.pkl
```

New command block layout after expansion:
`[vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual, yaw_actual, err_vx, err_vy, err_speed, err_angle, err_yaw]`

## Evaluation

### `evaluate.py`

Unified evaluation and video recording for both tasks.

```bash
# Evaluate standing
python scripts/evaluate.py --task standing \
    --model models/best_standing_model.zip

# Evaluate walking + record video
python scripts/evaluate.py --task walking \
    --model models/walking/final/final_walking_model.zip --record

# Walking with specific velocity command
python scripts/evaluate.py --task walking \
    --model models/walking/final/final_walking_model.zip \
    --vx 1.0 --vy 0.0 --record
```

Videos saved to `data/videos/{task}_eval.mp4`.

### `record_video.py`

Legacy video recording script. Prefer `evaluate.py --record` instead.

## Camera ground truth (`scripts/camera/`)

Dev box only (OpenCV); the Pi never needs it. Printed AprilTags (36h11) on each link + one
phone on a tripod = an independent measurement of what the body physically did: where it
bent (10/25: ~75% of pelvis pitch was invisible to the encoders), whether a foot rolled,
lifted or slid, and which way a fall twisted. Tag ids, links, sizes and mounts:
`config/camera/tags.yaml`. **Filming guide for the user: `scripts/camera/README.md`.**

```bash
python scripts/camera/make_targets.py                        # data/camera/camera_targets_{A4,Letter}.pdf
python scripts/camera/calibrate_camera.py board.mp4 --name pixel7_4k60   # -> config/camera/pixel7_4k60.yaml
python scripts/camera/video_pose.py run.mp4 --camera config/camera/pixel7_4k60.yaml
python scripts/camera/sync.py run.tags.csv logs/pi/<run>.csv  # -> run.aligned.csv keyed on log t_wall
```

| File | What it does |
|------|--------------|
| `camlib.py` | Shared: tags.yaml loader, face mounts, Z-Y-X roll/pitch/yaw (pitch > 0 = forward, = sign of pg_x), IPPE_SQUARE pose + continuity choice, edge-fit corner refinement |
| `make_targets.py` | Vector PDF, tags at exact size, 100 mm scale bar per page, ChArUco 5x7 board page |
| `calibrate_camera.py` | Board video/photos -> intrinsics; picks sharp, diverse views (farthest-point); warns RMS > 1 px |
| `video_pose.py` | `<stem>.tags.csv` (per frame per tag: camera-frame tag pose, world-frame link pose, missing/blurred/ambiguous/jump flags), `<stem>.links.csv` (per link world pose; per joint pair relative angles in the chain's own axis order), `<stem>.pose.json` |
| `sync.py` | Tap at start/end: tag motion (or frame difference) vs \|acc\| (v2) or gyro (v1). Coarse xcorr, then onsets; drift from 2 taps. Writes `<stem>.aligned.csv` (cam.\*, enc.\*, imu.\* per log row) + `<stem>.sync.json` (offset, drift, cam-vs-encoder/IMU offsets) |

Read the numbers this way: WORLD = the floor tag (z up), link frames = sim base frame at q = 0.
A pair angle (e.g. `R_knee.pitch`) is the encoder joint angle plus a constant mounting
offset; `sync.json` reports that offset (median) and the scatter around it -- the scatter
is real flex/play the encoders miss. `R_ankle`/`L_ankle` have no servo: anything there is
compliance. Positions are the tag centre, so use their changes (foot lift = foot z rise).
Accuracy on synthetic renders: ~1 mm / ~0.3 deg per 32 mm tag at 0.8 m in 1080p (cv2's own
SUBPIX corners were 2% / 1.8 deg off on tags that small, hence the edge refinement); sync
+-12 ms (half a 40 Hz log sample).

## Debug Scripts (`scripts/debug/`)

| Script | Purpose |
|--------|---------|
| `test_standing.py` | Standalone standing model evaluation |
| `test_walking.py` | Standalone walking model evaluation |
| `analyze_rewards.py` | Reward component analysis |
| `diagnose_transfer.py` | Debug standing → walking transfer |
| `standing_plot.py` | Plot standing training metrics |
| `walking_plot.py` | Plot walking training metrics |
