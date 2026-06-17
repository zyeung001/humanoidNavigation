# Deploy — proprioceptive standing on the real robot

Sim-to-real for the 19M tau=0.3 standing policy (`models/final_real_standing_model.zip`).
All files here run **on the Pi** except the `--dry-run` paths, which run anywhere.

## Files
| File | Role |
|------|------|
| `sim_real_map.py` | Single source of truth for index↔servo↔sign↔limit + rad/units conversion. Loads `config/joint_servo_map.yaml`. |
| `hardware.py` | **The hardware seam.** `ServoBus` (scservo_sdk SCSCL) + `IMU` (ICM-20948). Reconcile with your Pi driver HERE only. |
| `verify_signs.py` | Bench tool: confirm which servo each action index drives and its sign. Writes results back into the map. |
| `deploy_standing.py` | The closed-loop inference loop. |
| `export_policy.py` | **Dev box only:** dump the SB3 policy + vecnorm stats to a torch-free `.npz`. |
| `numpy_policy.py` | Pure-NumPy inference for the `.npz` (no torch/SB3/pickle). |

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
