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

## Debug Scripts (`scripts/debug/`)

| Script | Purpose |
|--------|---------|
| `test_standing.py` | Standalone standing model evaluation |
| `test_walking.py` | Standalone walking model evaluation |
| `analyze_rewards.py` | Reward component analysis |
| `diagnose_transfer.py` | Debug standing → walking transfer |
| `standing_plot.py` | Plot standing training metrics |
| `walking_plot.py` | Plot walking training metrics |
| `diagnose_run.py` | Hardware frame log -> timeline of tips, leans, oscillation, pushes, rig health |
| `log_pose.py` | Hardware frame log -> per-row body pose, COM, support margins, hidden flex (`<stem>.pose.csv` + `.pose_summary.json`) |
| `run_report.py` | Hardware frame log -> one-page run report (`<stem>.report.md` numbers-only + `.report.html` with plots); `--replay` = open-loop sim overlay |

### After a hardware run: `log_pose.py` / `run_report.py`

```bash
python scripts/debug/run_report.py logs/pi/<stamp>_standing.csv            # md + html + pose CSV/summary, next to the log
python scripts/debug/run_report.py logs/pi/<stamp>_standing.csv --replay   # + sim driven open loop by the logged commands
python scripts/debug/log_pose.py logs/pi/<stamp>_standing.csv              # pose CSV + summary only
```

Read `<stem>.report.md` first (numbers only, no images). It holds: fell or not and when,
pitch/roll extremes, hidden flex, min COM margin and toward which edge, joint tracking, loop
timing, FSR, servo telemetry (v2), events (`<stem>.events.jsonl`) and console warnings
(`<stem>.console.txt`) when those sidecars exist, and `diagnose_run.py`'s timeline verbatim.

How a row is posed (details in `log_pose.py`'s docstring):
- Joints from `u_meas` via the servo map (`frame_log.pose_from_row`); reads outside 0..1023 or NaN
  hold the previous value and are counted (10/25 01:28 had 13 joints read 14338 at t=42.3 s).
- Pelvis attitude from the logged `pg`, yaw 0. Stance foot = lowest sole centre; it sits on the
  floor; each supporting sole is a flat 160x80 (or 120x80) box -> signed margins to toe/heel/
  inner/outer and to the support polygon (+ = inside).
- Model chosen by the log date: `humanoid_real_v2_compliant_soleplate.xml` from 2026-10-24 (16 cm
  plates), `humanoid_real_v2_compliant.xml` before; `--model` overrides. Recorded in the summary.
- **Hidden flex** = IMU attitude - the attitude the encoders imply with both soles flat (encoder
  side EMA-filtered like the deploy `pg`). `pitch_missed_by_encoders` = share of the IMU pitch
  excursion the joints do not explain (> 1 = the joints moved against the lean, e.g. the knee law).
- `pg` is accelerometer-based, so attitude/flex/margins are inflated during pushes; trust rows with
  `quiet = 1` (|av| < 0.2 rad/s) for static numbers.
- `com_err` = COM had every servo reached `u_cmd` (feet flat, same flex) - measured COM.

