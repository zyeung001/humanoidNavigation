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

### `robot_run.py` — one command per hardware run (laptop side)

```bash
cp config/robot.yaml.example config/robot.yaml      # once: host/user/repo of the Pi (gitignored)
python scripts/robot_run.py run -- --policy-npz models/real_standing_knee2_kneelaw_db.npz \
    --residual-config config/stances/knee2.yaml --posture config/stances/posture.yaml
python scripts/robot_run.py run --script pose_sweep.py -- --joints waist_pitch
python scripts/robot_run.py sync  -- <same deploy args>   # only sync (prints the md5 table)
python scripts/robot_run.py attach | stop | fetch           # after an SSH drop / to end it / logs only
python scripts/robot_run.py <any of the above> --dry-run    # print the exact ssh/scp, touch nothing
```

- `sync`: one ssh `md5sum` of every `scripts/deploy/*.py` + the files the run's arguments name +
  the default servo map / IMU calibration; scp only what differs; md5 again; mismatch = error.
  Text files (.py .yaml .yml .json .txt .sh .md) are hashed and sent as CRLF->LF, i.e. as the git
  blob: this laptop has `core.autocrlf=true`, so a raw md5 of its CRLF checkout never matches the
  Pi's git (LF) copy -- on 10/25 that made the Pi's `imu_calib.yaml` look different when it was
  the repo's calibration (bf19d884 = the git blob). A Pi file differing only in line endings is
  re-copied as LF, silently. `config/imu_calib.yaml` and `config/joint_servo_map.yaml` are
  written by the Pi's calibration tools, so when their content differs the Pi copy is KEPT
  (warning) unless `--push-calibration`.
  Each `.npz` is checked with the `numpy_policy.py` being synced and its `requires` printed.
  Abbreviated file flags (`--policy` = `--policy-npz` to the deploy scripts' argparse) are
  resolved against the target script's flags and spelled out, so they are synced and checked
  too; an ambiguous one that might be a file flag is refused.
- `run`: sync, then the deploy script runs inside `tmux` on the Pi and you are attached with
  `ssh -t`; `--log` is added to `deploy_standing.py` if absent (not if `--no-log` is given).
  A KEPT calibration file stops `run` (its warning would vanish under tmux): re-run with
  `--push-calibration` or `--keep-pi-calibration`; the latter copies the Pi's servo map to
  `logs/pi/pi_maps/` and diagnoses with it. **An SSH drop no longer releases
  the torque** (it did since 10/18 via `release_on_hangup`): the run continues until `attach` +
  Ctrl-C, `stop`, or the tilt cut. `--no-tmux` restores the old behaviour. If the Pi cannot be
  reached after the attach ends, the run is assumed live and nothing is fetched.
- `fetch` (automatic after the tmux session ends): every file in the Pi's `logs/` that is new or
  LARGER than the copy in `logs/pi/`. A SMALLER one is a different run under a reused name (no
  RTC on the Pi; fake-hwclock steps back after an unclean power-off): its whole `<stem>.*` group
  is saved as `<stem>-pi2.*`, never over the old run. Then `scripts/debug/diagnose_run.py` on
  each new `.csv` (saved as `<stem>.diagnose.txt`) and `scripts/debug/run_report.py` if present,
  run with `PYTHONIOENCODING=utf-8`.

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

## Debug Scripts (`scripts/debug/`)

| Script | Purpose |
|--------|---------|
| `test_standing.py` | Standalone standing model evaluation |
| `test_walking.py` | Standalone walking model evaluation |
| `analyze_rewards.py` | Reward component analysis |
| `diagnose_transfer.py` | Debug standing → walking transfer |
| `standing_plot.py` | Plot standing training metrics |
| `walking_plot.py` | Plot walking training metrics |
