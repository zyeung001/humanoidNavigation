# Deploy — proprioceptive standing on the real robot

Sim-to-real for the 19M tau=0.3 standing policy (`models/final_real_standing_model.zip`).
All files here run **on the Pi** except the `--dry-run` paths, which run anywhere.

## Files
| File | Role |
|------|------|
| `sim_real_map.py` | Single source of truth for index↔servo↔sign↔limit + rad/units conversion. Loads `config/joint_servo_map.yaml`. |
| `hardware.py` | **The hardware seam.** `ServoBus` (raw-serial SCS: position, plus speed/load/voltage/temperature in the same read when `telemetry=True`) + `IMU` (ICM-20948). Reconcile with your Pi driver HERE only. |
| `verify_signs.py` | Bench tool: confirm which servo each action index drives and its sign. Writes results back into the map. |
| `deploy_standing.py` | The closed-loop inference loop. |
| `export_policy.py` | **Dev box only:** dump the SB3 policy + vecnorm stats to a torch-free `.npz`. |
| `numpy_policy.py` | Pure-NumPy inference for the `.npz` (no torch/SB3/pickle). |
| `frame_log.py` | Run recording (schema 2): per-frame CSV + `.meta.json` + `.console.txt`, FSR sampling over several ADS1115 boards. |
| `pose_sweep.py` | Measure the pose the robot balances in (CoP vs trim); records its live phase like `deploy_standing.py`. |
| `events.py` | Operator event markers: single keys on the SSH terminal -> `<stem>.events.jsonl`, plus the after-run "what did you see?". |

## Recommended: torch-free NumPy policy
Running torch + SB3 on a Pi works but the ARM wheels are heavy/fiddly. The policy is a small
`[512,512,256]` SiLU MLP, so export it to a NumPy bundle and the Pi needs only `numpy` + `pyyaml`:

```bash
# on the DEV box (has torch+SB3):
python scripts/deploy/export_policy.py        # -> models/real_standing_policy.npz (~2 MB)
#   bundles policy weights AND vecnorm stats; self-checks parity vs SB3 (asserts <1e-4).
# scp models/real_standing_policy.npz pi@<ip>:~/humanoidnavigation/models/

# on the Pi:
python scripts/deploy/deploy_standing.py --policy-npz models/real_standing_policy.npz
```
Output is byte-identical to the SB3 backend. Without `--policy-npz`, the loop falls back to
loading the `.zip` + `.pkl` (needs torch+SB3). The `.npz` is gitignored — `scp` it like the `.zip`/`.pkl`.

## Order of operations
1. **Validate wiring (any machine):** `python scripts/deploy/deploy_standing.py --dry-run`
   — loads model+vecnorm+map, builds the 228-dim obs, prints the first action + servo units. No hardware.
2. **Wire `hardware.py`** to your actual Pi servo/IMU calls (the `TODO`/`NotImplementedError` spots).
3. **Calibrate the IMU axis remap** (`IMU.axis_remap`) so accel/gyro land in the sim base frame
   (handoff item #2). With the robot held upright, `projected_gravity()` must read ≈ `[0,0,-1]`.
4. **Verify signs on the bench (robot supported):**
   `python scripts/deploy/verify_signs.py --apply`
   — turns the 13 `sign_verified: false` joints into tested entries and resolves the arm-DOF labels.
5. **First closed loop, supported/hanging:**
   `python scripts/deploy/deploy_standing.py --require-verified`
   — refuses to run until every sign is bench-verified.

## What is trustworthy now vs. needs the bench
- **Solid (from MJCF + tested memories):** all leg + waist DOF/servo mapping; the 4 hinge signs
  (idx 3,7,13,16 = knees/elbows); the obs construction; rad↔units math; tau=0.3 smoothing.
- **NOT yet verified (placeholders):** the 13 non-hinge signs are all `+1` guesses — your own
  notes say these need per-joint bench testing. **Do not drive under load before step 4.**
- **Ambiguous:** the middle arm joint (idx 12/15, "shoulder_roll?") DOF↔servo-ID — the MJCF arm is
  shoulder×2 + elbow (no wrist), but the servo memory labels IDs 12/13/14 shoulder/elbow/wrist.
  `verify_signs.py` resolves this by showing which joint actually moves.

## Critical invariants (do not change without retraining)
- **tau=0.3** action smoothing is part of the trained closed loop — `deploy_standing.py --tau` must stay 0.3.
- **Home keyframe** `default_joint_pos` (in the map) is a *bent* pose; obs joint angle is reported
  relative to it. The loop ramps to it before going closed-loop.
- **40 Hz** control rate (dt=0.025) matches sim.
- Feed `-normalize(accel)` as projected gravity (accel at rest reads +1g UP = −proj_grav).

## Safety in `deploy_standing.py`
- Open-loop ramp to home before closed loop (`--ramp-secs`).
- Tilt cutoff: cuts torque if `upright_cos < 0.5` (~60° tilt) — same threshold as the sim termination.
- Per-step servo move clamp (`--max-step-units`) rate-limits any wild command.
- Ctrl-C ramps to home and disables torque.
- FSR foot contact stays OFF (obs is 228, not 230) — matches the trained model.

## Every run is recorded (frame_log.py, schema 2)
`deploy_standing.py` and `pose_sweep.py` record by default (`--no-log` to skip; `--dry-run`
never records). Next to `logs/<YYYYmmdd_HHMMSS>_<tag>.csv`:

| File | What |
|------|------|
| `<stem>.csv` | one row per control frame (hold-pose runs too) |
| `<stem>.meta.json` | every CLI arg, mode, gyro bias, axis remap, prior gains, posture config, md5 of every file that shaped the run (map, policy, vecnorm, residual/posture/FSR yaml, imu calib, **each deploy source**), host, python/numpy, servo-telemetry decoding, FSR layout, row/drop counts |
| `<stem>.console.txt` | everything printed (stdout + stderr, tracebacks included), each line `unix_time \| text` |
| `<stem>.events.jsonl` | operator markers, one JSON object per line (see below; only written when a key was pressed) |

Columns added in schema 2 (schema-1 names and meanings unchanged; readers look columns up by name):
`acc_x/y/z` raw accel in g (remapped, not normalized) · `spd_raw/load_raw/volt_raw/temp_raw.<dof>`
RAW servo registers from the same read as `u_meas` (decoding in the meta; `--no-servo-telemetry`
falls back to the position-only read if `busy_ms` shows the loop can't hold 40 Hz. Before torque
comes on, every servo must answer the 8-byte read once (`ServoBus.check_telemetry`); if one does
not, the run prints `[warn]`, reads position only, and the meta says `telemetry_fallback: true`.
Telemetry is only read when the run is logged: `--no-log` and every other tool keep the 2-byte read) ·
`q_prior.<dof>` the feedback law's contribution (rad, NaN without one) · `posture_lean/zero/corr`
(deg, NaN with the posture controller off) · `fsr_v0..fsr_v{N-1}`, now LAST in the row.

What a log covers: the ramp-in is not recorded; every control frame is; a tilt cut records the cut
frame and then ~1 s of the fall with torque off (`u_cmd`/`q_cmd` NaN); a Ctrl-C or SSH hangup
records the 1 s ramp to home (`u_cmd` = the ramp, `q_cmd` NaN). On exit, torque comes off
BEFORE the log files are closed, so a slow SD card can never keep the robot stiff.
`pose_sweep.py` records ramps, dwells and the ramp back; a failed IMU or servo read there is NaN
in the row and counted in the meta's `sensor_errors`, never an abort.

## Marking what happened (events.py)
The CSV shows what the robot did, never what was done to it: a push, a catch or a kick looks
like the robot moving by itself. While `deploy_standing.py` or `pose_sweep.py` runs, press one
key the moment it happens (no Enter; a legend is printed at the start, each key is echoed):

| Key | label | Key | label |
|-----|-------|-----|-------|
| `f` `b` `l` `r` | `push_forward` `push_back` `push_left` `push_right` | `k` | `kick` (any impact) |
| `x` | `fell` | `m` | `mark` (generic "look here") |
| `c` | `caught` | `n` | `note` (stamps now; the text is asked for after the run) |

Each line of `<stem>.events.jsonl` is `{"t_wall", "t_rel", "step", "key", "label", "note"}`:
`t_rel` is on the CSV's clock and `step` is the logged frame nearest the keypress (before the
first frame -> the first, after the last -> the last). After the run, with torque already
off, it asks for each `n` note and a one-line "what did you see?" (Enter skips), stored as a
`run_note` line (`key` "", the run's last step) and as `events.operator_note` in the meta.
The meta's `events` block also says whether keys were on and, if not, why.

- **Needs a terminal:** run it as `ssh -t pi@<ip> 'cd ~/humanoidnavigation && python3 ...'` or from
  an interactive SSH shell. Piped, `nohup`, ssh without `-t`, Windows or a background job (`&`):
  one `[events] off: <why>` line and the run is unchanged.
- **The terminal is always given back:** cbreak mode, not raw, so Ctrl-C stops the run as
  before; Ctrl-Z is disabled while keys are read (suspending the loop would freeze the robot at
  torque). On a normal end, a tilt cut, Ctrl-C, SIGTERM, SIGHUP or a crash the mode is restored
  right AFTER torque-off. If a terminal is ever left without echo, type `reset` + Enter.
- **Costs the loop nothing:** keys are read on their own thread, and the step is matched on
  the frame logger's writer thread. Each key is appended to the file when pressed, so a run
  that dies keeps its markers.
- `--no-events` leaves the terminal alone; `--no-prompt` skips the after-run question.
  With `--no-log` there is nowhere to put them, so they are off too.

## FSRs: several boards, 4 per foot
Default is unchanged: `--log-fsr` samples `0,1` on the ADS1115 at 0x48 (left heel, left toe).
For more channels give ADDR:AIN pairs, e.g. `--fsr-channels "0x48:0,0x48:1,0x49:0"`, or a layout
file that also records each sensor's foot and position (for a per-foot centre of pressure later):

```bash
python3 scripts/deploy/deploy_standing.py ... --log-fsr --fsr-layout config/fsr_layout_4per_foot.yaml
```

Wiring for 4 FSRs per foot: two ADS1115 boards on I2C bus 1 (SDA/SCL shared with the IMU at
0x68), both powered from 3.3 V. **Board A: ADDR pin to GND = 0x48 (left foot). Board B: ADDR
pin to VDD = 0x49 (right foot).** Never leave ADDR floating -- the address is then undefined.
Each FSR is its own divider into one AIN pin, as `fsr_monitor.py` describes
(3.3V -- FSR -- AINx -- R_fixed -- GND), AIN0..3 = heel-out, toe-out, heel-in, toe-in, matching
the template. Check each board answers with `i2cdetect -y 1` (expect 48, 49, 68) and each
sensor with `fsr_monitor.py --addr 0x49 --channels 0,1,2,3`. Measure the sensor positions
and fill in `x_mm`/`y_mm`; until then they are recorded as null, never guessed. The boards
convert in parallel, so 8 channels cost 4 conversion waits per sweep on the FSR thread.

## Pi verification checklist (schema 2)
Everything above is tested on the laptop against fake servos, IMU and ADC boards. These can only
be checked on the robot; do them once after syncing, before trusting a schema-2 log.

1. **Sync + version.** scp `hardware.py`, `frame_log.py`, `events.py`, `deploy_standing.py`, `pose_sweep.py` and
   `config/fsr_layout_4per_foot.yaml` with the usual md5 check, then
   `python3 -c "import sys; sys.path.insert(0,'scripts/deploy'); import frame_log, hardware; print(frame_log.SCHEMA_VERSION)"`
   must print 2.
2. **The 8-byte read, torque OFF, robot supported:**
   `python3 -c "import sys; sys.path.insert(0,'scripts/deploy'); from hardware import ServoBus; b=ServoBus(telemetry=True).connect(); [print(s, b.read_state(s), b.read_pos(s)) for s in range(1,18)]"`.
   On every servo (arms 12-17 too): `read_state` is not None, its first value equals `read_pos`,
   voltage ~60-84 (0.1 V units), temperature ~room. A None means that servo ignores the long
   read; the deploy then falls back by itself (`[warn]` + `telemetry_fallback: true`), but note it.
3. **Load/speed sign bits, torque ON, one leg joint:** push it one way, then the other, and print
   `read_state`: `load_raw` 0..1000 one way and 1024..2024 (bit 10) the other. A quick hand move
   should set speed bit 15 in one direction only, matching the `u_meas` trend. Record whether the
   SDK decoding in `hardware.TELEMETRY_DECODING` held.
4. **Timing:** `--hold-pose` and a normal closed loop, each once default and once with
   `--no-servo-telemetry`, and once with `--no-log`, ~60 s each. Compare `busy_ms`/`loop_ms`
   p50/p99 in the CSVs. Expected: telemetry ~+1 ms, `loop_ms` ~25. If the loop goes over 25 ms,
   run with `--no-servo-telemetry` and report it.
5. **Default-on logging:** run `deploy_standing.py` with NO `--log`. `logs/<time>_standing.csv`,
   `.meta.json` and `.console.txt` all appear; meta has `schema_version` 2, `gyro_bias`, `args`,
   `hostname` = the Pi, and `files` md5s (incl. `source:hardware.py`) matching the laptop copies.
   Console lines start with a unix time close to the CSV's `t_wall`.
6. **Hold-pose rows:** `--hold-pose --residual-config config/stances/knee2.yaml --policy-npz <hold npz>`.
   One row per frame; `acc_z` ~+1.0 g upright, spikes when tapped; `temp_raw`/`volt_raw` filled.
   Let it tip into the tilt cut: the last ~40 rows have `u_cmd` NaN and show the fall in `acc_*`.
7. **Posture and law columns:** `--posture config/stances/posture.yaml` with an npz that has a prior.
   `posture_zero` NaN for ~3 s, then constant; `posture_corr` matches the printed
   `[posture] ... knees` lines; `q_prior.*` finite on the law's joints, 0 elsewhere.
8. **SSH hangup:** start a run, kill the laptop's SSH session, fetch the logs. `console.txt` has
   `Stopping: ramping to home` and the `Frame log: N rows` line; the CSV has the ~40 ramp rows; the
   meta has `rows_written`/`ended_unix`; the servos are released.

### Event markers (events.py)
9. **Keys during a run:** `ssh -t` in, `--hold-pose` as in 6, robot supported. The legend
   prints; press `m`, `f`, `x`, `c`, `n`, and one key that means nothing (`q`): each echoes at
   once (`q` prints a hint), none shows as a typed letter, and the hold keeps 40 Hz
   (`loop_ms` p99 unchanged). End it with Ctrl-C: torque comes off, the terminal echoes
   normally again, and it asks for the `n` note, then "What did you see?". Answer both.
   `.events.jsonl` has 6 lines (5 keys + `run_note`); each `step` is the CSV row whose `t_wall`
   is nearest the line's `t_wall`; the meta's `events.count` is 6.
10. **Ctrl-Z is ignored, Ctrl-C is not:** during a run Ctrl-Z does nothing (the run keeps
   going); Ctrl-C still stops it.
11. **Restored on a kill:** start a run, `kill <pid>` (SIGTERM) from a second SSH session:
   torque off, then the question appears in the first terminal and what you type there is
   echoed (after it, `stty -a` shows `icanon echo`). Repeat with a tilt cut (`--hold-pose`,
   let it tip into the cut).
12. **Hangup:** start a run, close the laptop's SSH session: servos released (as in 8),
   `.events.jsonl` has the keys pressed before the drop, and the run did not wait at the prompt
   (`ended_unix` within seconds of the drop; meta `events.operator_note` null). Log back in:
   the new shell is normal.
13. **No terminal:** `ssh pi@<ip> 'cd ~/humanoidnavigation && python3 scripts/deploy/deploy_standing.py --hold-pose ...'`
   (no `-t`) prints `[events] off: stdin is not a terminal ...`, runs, and never prompts.
   Same with `... &` in a shell: `[events] off: the run is in the background ...`, and the
   job is NOT stopped (`jobs` shows Running).
14. **pose_sweep:** a short `--probe 0` with keys `x`/`c` as it goes over and is caught; the
   question comes after the `--- PROBE` summary.
9. **FSR, default wiring:** `--log-fsr` alone: meta `fsr.channels` = L_heel/L_toe on 0x48 AIN0/1;
   `fsr_v0`/`fsr_v1` behave as before (press heel, then toe).
10. **FSR, two boards (after rewiring):** `i2cdetect -y 1` shows 48, 49, 68;
    `fsr_monitor.py --addr 0x49 --channels 0,1,2,3` responds. Fill in `x_mm`/`y_mm`, then
    `--fsr-layout config/fsr_layout_4per_foot.yaml` (implies `--log-fsr`): each `fsr_v0..7` responds
    to its own sensor, `fsr_age_ms` stays under ~40, meta `fsr_errors` 0, and the IMU still reads
    cleanly with the second board on the bus.
11. **pose_sweep:** `pose_sweep.py --joints waist_pitch --range -2 2 --step 2`, and once with
    `--probe 0`. The CSV has rows through ramp, dwell and ramp back; `sweep_points` and
    `sensor_errors` (expect 0) are in the meta; `console.txt` holds the RESULT block; the printed
    CoP and slope look as before. The extra reads per sample slow dwell sampling to ~16 Hz.
