# Deploy — proprioceptive standing on the real robot

Sim-to-real for the 19M tau=0.3 standing policy (`models/final_real_standing_model.zip`).
All files here run **on the Pi** except the `--dry-run` paths, which run anywhere.

## Files
| File | Role |
|------|------|
| `sim_real_map.py` | Single source of truth for index↔servo↔sign↔limit + rad/units conversion. Loads `config/joint_servo_map.yaml`. |
| `hardware.py` | **The hardware seam.** `ServoBus` (raw-serial SCS: position + speed/load/voltage/temperature in one read) + `IMU` (ICM-20948). Reconcile with your Pi driver HERE only. |
| `verify_signs.py` | Bench tool: confirm which servo each action index drives and its sign. Writes results back into the map. |
| `deploy_standing.py` | The closed-loop inference loop. |
| `export_policy.py` | **Dev box only:** dump the SB3 policy + vecnorm stats to a torch-free `.npz`. |
| `numpy_policy.py` | Pure-NumPy inference for the `.npz` (no torch/SB3/pickle). |
| `frame_log.py` | Run recording (schema 2): per-frame CSV + `.meta.json` + `.console.txt`, FSR sampling over several ADS1115 boards. |
| `pose_sweep.py` | Measure the pose the robot balances in (CoP vs trim); records its live phase like `deploy_standing.py`. |

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
| `<stem>.events.jsonl` | operator markers (reserved; one JSON object per line) |

Columns added in schema 2 (schema-1 names and meanings unchanged; readers look columns up by name):
`acc_x/y/z` raw accel in g (remapped, not normalized) · `spd_raw/load_raw/volt_raw/temp_raw.<dof>`
RAW servo registers from the same read as `u_meas` (decoding in the meta; `--no-servo-telemetry`
falls back to the position-only read if `busy_ms` shows the loop can't hold 40 Hz) ·
`q_prior.<dof>` the feedback law's contribution (rad, NaN without one) · `posture_lean/zero/corr`
(deg, NaN with the posture controller off) · `fsr_v0..fsr_v{N-1}`, now LAST in the row.

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
