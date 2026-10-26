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

## Robot runs (`scripts/deploy/`, run on the Pi)

`deploy_standing.py` and `pose_sweep.py` RECORD every hardware run by default
(`scripts/deploy/frame_log.py`, schema 2; `--no-log` to skip): `logs/<time>_<tag>.csv` +
`.meta.json` + `.console.txt` + `.events.jsonl`. To diagnose a run, read all of
them: the CSV has per-frame commands, encoder units, raw accel, servo load/voltage/temperature
(RAW registers; decoding in the meta), the feedback law's share (`q_prior.*`), the posture
controller's internals and the FSR volts; the meta says exactly which files (md5) and args ran;
the console has what the operator saw, timestamped to line up with `t_wall`; the events file
has the operator's keypress markers (`scripts/deploy/events.py`: push_forward/back/left/right,
fell, caught, kick, mark, note, each with `t_wall`/`t_rel` and the nearest logged `step`) and a
`run_note` line with their after-run "what did you see?" (also `events.operator_note` in the
meta). Start from the events: they say where in the frames to look. No events file = no key
was pressed or there was no terminal (`events.reason` in the meta says which). No `run_note`
after a kill or SSH hangup is expected: the question is not asked then (nobody may be there).
A tilt cut leaves the cut frame and ~1 s of the fall (torque off, `u_cmd` NaN); a Ctrl-C/hangup
leaves the ramp home. `servo_telemetry.telemetry_fallback: true` in the meta = a servo ignored
the 8-byte read at startup and the run logged position only.
`scripts/debug/diagnose_run.py <csv>` summarises one. Details: `scripts/deploy/README.md`.

## Debug Scripts (`scripts/debug/`)

| Script | Purpose |
|--------|---------|
| `test_standing.py` | Standalone standing model evaluation |
| `test_walking.py` | Standalone walking model evaluation |
| `analyze_rewards.py` | Reward component analysis |
| `diagnose_transfer.py` | Debug standing → walking transfer |
| `standing_plot.py` | Plot standing training metrics |
| `walking_plot.py` | Plot walking training metrics |
