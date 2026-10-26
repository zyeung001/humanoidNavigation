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
| `sync.py` | Tap at start/end: tag motion (or frame difference) vs \|acc\| and gyro (v2: whichever matches better; v1: gyro). Coarse = correlation of event-onset trains (counts coinciding events, so a video-only handling event cannot win), restricted to lags that keep the log inside the video; then onsets; drift from 2 taps. `verified` only when start and end agree, else SYNC NOT VERIFIED + exit 2. Writes `<stem>.aligned.csv` (cam.\*, cam.world_ref, enc.\*, imu.pg.\* = filtered pg, imu.acc.\* = raw accel, per log row; cam.<link>.\* blank where world_ref = 0) + `<stem>.sync.json` (verified, offset, drift, log filter alphas, cam-vs-encoder/IMU offsets) |

Read the numbers this way: WORLD = the floor tag (z up), link frames = sim base frame at q = 0.
A pair angle (e.g. `R_knee.pitch`) is the encoder joint angle plus a constant mounting
offset; `sync.json` reports that offset (median) and the scatter around it -- the scatter
is real flex/play the encoders miss. `R_ankle`/`L_ankle` have no servo: anything there is
compliance. Positions are the tag centre, so use their changes (foot lift = foot z rise).
Filming geometry is set by the vertical field of view: floor tag + feet + flag top fit at
~1.4 m in landscape (4K) or ~1.0 m in portrait (1080p); at 1 m landscape the floor tag drops
out and world_ref = 0. Without a floor tag video_pose uses image-up as the upright prior.
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
- Pelvis attitude from the logged `pg`, yaw 0. `anchor` = the lowest sole centre (kept unless the
  other sole is 2 mm lower, so two flat feet do not swap it every frame); it sits on the floor.
  `stance` = the feet carrying weight (`R`/`L`/`both`, other sole within 10 mm). Each supporting sole
  is a flat 160x80 (or 120x80) box -> signed margins to toe/heel/inner/outer and to the support
  polygon (+ = inside; inner = toward the other foot, + y = the robot's left).
- Model chosen by the log date: `humanoid_real_v2_compliant_soleplate.xml` from 2026-10-24 (16 cm
  plates), `humanoid_real_v2_compliant.xml` before; `--model` overrides. Recorded in the summary.
- **Hidden flex** = IMU attitude - the attitude the encoders imply with both soles flat (encoder
  side EMA-filtered like the deploy `pg`). `pitch_missed_by_encoders` = share of the IMU pitch
  excursion the joints do not explain (> 1 = the joints moved against the lean, e.g. the knee law).
- `pg` is accelerometer-based, so attitude/flex/margins are inflated during pushes; trust rows with
  `quiet = 1` (|av| < 0.2 rad/s) for static numbers.
- `com_err` = COM had every servo reached `u_cmd` (feet flat, same flex) - measured COM.
- The report's joint-tracking table is `q_cmd - q_meas` in SIM degrees (servo sign applied), so a
  sag on both knees reads the same sign on R and L; raw-unit errors would flip sign on mirrored servos.
- `pitch_excursion_ratio_imu_over_enc_1s` = IMU/encoder pitch std after a 1 s moving average (slow
  motion only; 2.7 on 10/25 03:13), next to the raw ratio (2.05) and quiet-row flex stats.

