# Handoff — Real Humanoid Standing Deployment

_Last updated: 2026-06-22 (actuator-bandwidth session) · Branch: `deploy-imu-pelvis-recal` · PR #40_

Paste this into a fresh Claude Code chat to continue. Durable facts also live in
`~/.claude/.../memory/` (see MEMORY.md index) — this doc is the *current task state* on top.

## TL;DR

A MuJoCo-trained PPO standing policy stands **100% in sim** but **topples on the real robot
in ~0.25 s**. Root cause is now **quantified: actuator bandwidth.** The real SCS servos have
**~50 ms dead time + ~120 ms time constant (~2.1 rad/s max)**; the policy was trained on an
ideal actuator that tracks instantly, so it relies on fast corrections the servo can't follow.
A heavy mass/friction "robustness" retrain made it **worse** (wrong axis → more aggressive
control). The real fix — **modeling the measured servo lag in sim** — is built and committed.
**NEXT ACTION: USER runs the actuator-lag retrain**, then re-export the npz and redeploy.

> ⚠️ **The user runs all training/long runs themselves.** Prep configs/backups, hand over the
> exact command. Eval/diagnostics are fine to run directly.
> ⚠️ **Every commit must pass CI** (`ruff check .`). Lint before committing.
> ⚠️ **Pi is reachable by IP `192.168.86.36`.** MSYS/Git-bash can't resolve `ZachPi.local`
> mDNS (Windows `ping` can). scp/ssh by IP.

---

## The failure & the diagnosis (6/22)

1. **Closed-loop standing fails.** With all signs/limits/IMU verified clean, the robot
   seizes/jitters and would fall; `deploy_standing.py --debug` showed the **obs is clean**
   (`pg≈[0,0,-1]`, sane jvel). So it's not wiring/sensing.
2. **The mass/friction robust retrain (`final_real_standing_robust.zip`) was WORSE** — toppled
   in ~10 steps, tilt-cut at upright_cos 0.43 (clean was 0.49). It sim-evals 8/8 under
   randomization but learned *bigger, faster* corrections.
3. **Sim diagnostic:** the same policy in sim makes huge fast actions (raw `d_act` 3–4 rad/step)
   and stays pinned at h=1.40 → the **plant**, not the obs, is the problem.
4. **Bench step-response settled it** (`scripts/deploy/servo_step_response.py`): SCS servos =
   **~50 ms dead time, ~120 ms τ, ~2.1 rad/s**, identical on knee (servo 7) and hip (servo 6).
   The policy commands up to 16 rad/s (80-unit clamp); the EMA `tau=0.3` models only ~58 ms of
   lag and **zero dead time**. That gap is the whole failure.
5. **Not a compute problem.** The Pi 4 polled servo position at **2,638 Hz** in that test —
   the bottleneck is the servos, not the CPU. A Pi 5 / Jetson would not help; better actuators
   (Dynamixel / BLDC) would, but that's a hardware-cost decision, not needed for quiet standing.

**Sim check passed:** the deployed model stands 100% in sim (h=1.400, tilt_cos ~0.996) — we're
hardening a working controller, not chasing a broken one.

## The fix (built + committed, config-gated OFF by default)

- **`src/environments/standing_env.py` `actuator_lag`** — feeds the MuJoCo actuator a
  **delayed + first-order-lagged** version of the commanded target. The obs's `last_action`
  stays the commanded value and `jpos` reflects the lagged result, so the policy sees
  command-vs-actual divergence exactly like hardware and learns gentle, trackable control.
  Randomized per episode: `actuator_delay_ms [30,70]`, `actuator_tau_ms [90,220]`. Verified to
  reproduce the measured step response (dead-time 2 steps, τ ~120 ms). Sim control dt = 0.025 s
  = **exactly 40 Hz = deploy rate**. **Training-only** — deploy does NOT enable it (the real
  servo supplies the lag; enabling it there would double-lag).
- **Joint-zero map fix (partial):** `scripts/deploy/sim_real_map.py` now supports a per-joint
  `center`; `config/joint_servo_map.yaml` sets **waist_roll (servo 3) `center: 485`** (hand
  re-zero, ~8° lateral lean) — corrects both obs and command. Other joints' `home.json` offsets
  are ≤2° and eyeballed → left at 512. Already synced to the Pi.
- **`config/real_humanoid_actuator.yaml`** — resumes the **clean 19M** (not robust): actuator
  lag (the star) + mild mass/friction + actuator-gain + obs-noise; `action_rate_penalty` 15→25.
  Distinct `*_actuator` output paths — clean model preserved.

---

## NEXT ACTION — user runs the actuator-lag retrain

```bash
python scripts/train_standing.py --config config/real_humanoid_actuator.yaml \
  --model models/final_real_standing_model.zip \
  --vecnorm models/vecnorm_real_standing.pkl \
  --timesteps 35000000
```

Resumes the clean 19M. **Watch:**
- `approx_kl` < ~0.1
- `ep_len_mean` dips when the lag kicks in, recovers toward ~5000 in a few M
- **`d_act` (per-step action swing) should SHRINK** — the policy learning gentle control is the
  whole point. If it **collapses**, the lag is too hot → dial `actuator_tau_ms` toward
  `[80,150]`, then `actuator_delay_ms` toward `[30,50]`.

> CPU-only box (no NVIDIA GPU); 16M steps is multi-hour — run where there's compute/time.
> Capture output with `PYTHONUTF8=1`.

### After training finishes
1. Eval `*_actuator` in sim (sanity: still stands clean) — `scripts/evaluate.py --task standing
   --config config/real_humanoid_config.yaml --model models/final_real_standing_actuator.zip
   --vecnorm models/vecnorm_real_standing_actuator.pkl --episodes 5 --max-steps 5000`.
2. **Re-export the npz** from the new model + `vecnorm_real_standing_actuator.pkl`
   (`scripts/deploy/export_policy.py --model … --vecnorm … --out models/real_standing_policy.npz`);
   scp to Pi (`scp models/real_standing_policy.npz zyeung001@192.168.86.36:~/humanoidnavigation/models/`).
3. Redeploy: `python scripts/deploy/deploy_standing.py --policy-npz models/real_standing_policy.npz --debug`
   (EMA tau=0.3 default; do NOT pass `--imu-on-chest`). Hold the robot — watch `d_act` and tilt.
4. Expect to iterate on `actuator_tau_ms` / `action_rate_penalty`.

### Deferred
Re-zero the remaining joints by hand → fold straight positions into the map as per-joint
`center` values (only waist_roll done so far). The per-episode `bias_jpos` randomization partly
covers the residual offsets for now.

---

## Hardware state (verified)

- **Pi:** `ssh zyeung001@192.168.86.36` (ed25519 key). aarch64, Python 3.13, numpy 2.2.4,
  smbus2, **no `scservo_sdk`** (raw pyserial). Runs the numpy-only `real_standing_policy.npz`.
- **Servos:** 17× SCS, `/dev/ttyAMA0` @ 1 Mbaud, `scripts/deploy/hardware.py` (`ServoBus`).
  **Measured ~50 ms dead time + ~120 ms τ, ~2.1 rad/s.** 195 units/rad, center 512 (waist_roll
  485). All 17 signs + EEPROM limits verified; elbows are servos **14/17** (not 13/16).
- **IMU:** ICM-20948 on the **PELVIS**. `config/imu_calib.yaml` axis_remap
  `[[0,0,-1],[0,-1,0],[-1,0,0]]`, upright proj_grav ≈ `[0,0,-1]` (2.8° tilt). Chest transform
  code DELETED.

## Deploy scripts (`scripts/deploy/`)

- `deploy_standing.py` — closed loop. `--policy-npz`, `--tau 0.3`, `--jvel-alpha 0.35`,
  `--debug`. IMU read directly in obs frame.
- `servo_step_response.py` — **NEW** single-joint step-response / bandwidth logger (SSH-drivable,
  torques one joint only). `--joint R_knee --delta-units -50`.
- `export_policy.py` — SB3 → numpy `.npz` (parity-checked). `home.py`, `probe_sign.py`,
  `calibrate_imu.py`, `widen_hip_limits.py`, `fix_arm_limits.py` as before.

> Pi git fetch refspec is main-only. Sync a branch with
> `git fetch origin <branch> && git reset --hard FETCH_HEAD`. Capture on-Pi edits first (reset
> wipes them) — edit + commit locally, then reset Pi.

## Key reference numbers

- Proprioceptive obs = 228-dim: `proj_grav(3)|base_ang_vel(3)|jpos−default(17)|jvel(17)|
  last_action(17)` × history(4). default_joint_pos = v2 BENT keyframe.
- Position servos (action = target joint ANGLE in rad), kp=12/kv=2 legs+waist, kp=5/kv=0.5
  arms. xml `models/humanoid_real_v2.xml`. log_std clamp [-2.0,-1.5]. Sim control = 40 Hz.
- Clean model: `models/final_real_standing_model.zip` + `models/vecnorm_real_standing.pkl`
  (19M, tau=0.3; 100% sim eval). Robust (failed on HW): `*_robust`. npz backup of clean policy:
  `models/real_standing_policy_clean.npz`.

## Relevant memory files
real-deploy-bringup, real-standing-jitter-tau, proprioceptive-obs, v2-position-servo-std,
hinge-directions, servo-mapping, sensor-bringup, user-runs-training.
