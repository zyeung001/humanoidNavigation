# Handoff: warm-start obs expansion (228 → 239) for the turning policies

**Read this cold — it's self-contained.** Also see memory `project_real_turning_env.md`,
`project_waist_twist_paper.md`, `feedback_user_runs_training.md`.

---
## ⏩ SESSION UPDATE (current state — read this first)

**The warm-start task below is DONE, and so is everything else in the build pipeline.** Branch
`real-robot-mjcf` (off `deploy-imu-pelvis-recal`). Commits: 8fc9e43 MJCF, 574529c heading,
97cba01 env, c0205de train+configs, 0cb3a4c WTR-eval.

**Contribution-2 build pipeline = COMPLETE:**
- MJCF `humanoid_real_v2.xml` = real 2046 g + 120×80 feet (via `build_humanoid_mjcf.py::apply_measured_inertials`). Integrity 13/13.
- `src/core/heading.py HeadingYaw` — shared torso/pelvis code path, bit-identical no-op refactor of walking_env. Real body uses `root_is_torso=False`, torso=`0003_8` (objVelocity), pelvis=`base_link` (qvel[5]).
- `src/environments/real_turning_env.py RealTurningEnv` — 228+11=239 obs. Reward NOW includes turn-enabling terms (verified implemented): height+upright (stay up), yaw-track on heading source, **feet_air_time_weight** (reward foot-lift so planted feet can pivot), **turn_survival_floor 0.3** (scales stay-up reward by actual turn → kills stand-still exploit), **yaw_wrong_dir_penalty**, tilt-rate penalty (off-axis only, never the commanded z-yaw). Configs `config/turning/heading_torso.yaml` (HACK) + `heading_pelvis.yaml` (FIX) carry these.
- **Warm-start DONE**: `scripts/expand_turning_obs.py` → `models/turn_warmstart.zip` + `.pkl` (228→239, num_timesteps RESET to 0 so `--timesteps` is ABSOLUTE, parity 2e-7). Validated by `scripts/validate_turn_warmstart.py`.
- **WTR sim-eval DONE**: `scripts/measure_waist_twist_real.py` (commit 0cb3a4c) — WTR=std(torso_yaw−pelvis_yaw) over turn-in-place schedule; torso=`0003_8`, pelvis=`base_link`, feet `0003_2`/`0003_5`.

**What's left:**
1. **(USER runs)** train HACK + FIX: `python scripts/train_standing.py --config config/turning/heading_{torso,pelvis}.yaml --model models/turn_warmstart.zip --vecnorm models/turn_warmstart.pkl --timesteps 20000000`. (2 PPO @ n_envs 12 may not both fit one GPU — queue if OOM.)
2. **eval** each: `python scripts/measure_waist_twist_real.py --model models/final_real_turn_hack.zip --vecnorm models/vecnorm_real_turn_hack.pkl` (hack → high WTR/torso>pelvis; fix → low). Pick best seeds.
3. **LAST BUILD (not started): trajectory-record + open-loop replay tool.** Record the best policy's 17-joint target trajectory doing a commanded turn in sim → replay open-loop on the Pi (position targets on a clock, NO feedback → cannot seize) while logging pelvis IMU + waist_yaw. THAT run is the hardware figure (hack twists, fix steps). Supported replay is fine to disclose.
4. FIX-arm chicken-and-egg: clean pelvis-heading turn-in-place may want a real step-turner warm-start (→ real walker first). First pass: warm-start both from standing, see how far pelvis-heading gets; the air-time + survival-floor terms are meant to push it toward stepping.

**Standing / autonomous-balance state (BONUS, paper does NOT need it):**
- HW standing seizure was largely CONTAMINATED diagnosis: wrong map centers (±118u/35°) + off-distribution pose ref. Fixed: per-joint centers re-zeroed; deploy ramps to SIM-STRAIGHT (joints 0 = centers), where obs jpos=−default = the in-distribution settled standing obs. `deploy_standing.py` corrected (ramp-to-straight) — on the Pi.
- Clean straight-start closed-loop: calm ~1 s then diverges; with `--angvel-alpha 0.3 --max-step-units 12` it tames to a slow gentle lean (arms stay on) but still only ~2 s hands-off. Sim shows the balanced policy is symmetric+low-gain (0.035) — NOT the earlier "asymmetric/high-gain" (that was the contaminated obs).
- `scripts/find_balance_signs.py` + `src/core/balance_baseline.py` (uncommitted): sim shows straight+flat 120×80 feet is STATICALLY stable for small/moderate disturbances (lateral 14° self-recovers). Active tilt-feedback hip baseline has WEAK authority on pitch (no-ankle limit). **Verdict: autonomous stand = QUASI-STATIC (`home.py --hold --waist-lean N`, find the small trim), NOT a PD+residual build.** No-ankle is CORRECT (matches Humanoid-v5, which balances without ankles in sim).

**Uncommitted (intentionally, on the Pi / working tree):** deploy calibration `config/joint_servo_map.yaml` (re-zeroed centers), `scripts/deploy/deploy_standing.py` (ramp-to-straight + `--angvel-alpha`/`--arm-step-units`), `scripts/deploy/home.py` (`--waist-lean`); balance investigation `src/core/balance_baseline.py`, `scripts/find_balance_signs.py`. Pi = zyeung001@192.168.86.36.

---
## Why this task exists (original warm-start handoff — task now DONE, kept for reference)
Paper contribution 2 = validate the pelvis-vs-torso heading-source fix on the real robot via
**open-loop turn replay**: train two turning policies in sim (hack=torso-heading, fix=pelvis-
heading), record each one's joint trajectory during a commanded turn, replay open-loop on the
robot while logging the pelvis IMU + waist-yaw. Hack twists the waist, fix steps → the hardware
figure. (Sim training is fine — the hardware seizure is closed-loop-only; sim balance works.)

The turning env obs is **239-dim** (228 proprioceptive standing obs + 11 command block). The
standing model `models/final_real_standing_balanced.zip` is **228-dim**. To warm-start the
turning training from a body that already stands symmetric in sim, the 228-dim policy +
VecNormalize must be **expanded to 239** (the 11 command-block columns added, zero/identity-
init). From-scratch sim training also works but is slower/less reliable; warm-start is the move.

## Current state (branch `real-robot-mjcf`, off `deploy-imu-pelvis-recal`)
Committed and verified:
- `8fc9e43` MJCF rebuilt to real masses (2046 g) + 120×80 feet (`models/humanoid_real_v2.xml`,
  via `scripts/build_humanoid_mjcf.py::apply_measured_inertials`). Integrity 13/13.
- `574529c` `src/core/heading.py` `HeadingYaw` — shared heading-source code path, bit-identical
  no-op refactor of `walking_env`.
- `97cba01` `src/environments/real_turning_env.py` `RealTurningEnv` — 239-dim, verified.
- `c0205de` train path + `config/turning/heading_torso.yaml` (hack) + `heading_pelvis.yaml` (fix).
  `train_standing.py` routes `env_kind: real_turning` → `make_real_turning_env`, reads a
  `turning:` block.

## THE TASK — implement the 228→239 expansion
Produce expanded copies of the standing model + vecnorm that `train_standing.py --model/--vecnorm`
can resume the turning training from.

**Obs layout (critical):** turning obs = `[ 228 stacked proprioceptive dims | 11 command block ]`.
The 11 NEW dims are **appended at the END** (indices 228..238). See
`real_turning_env._process_observation`: `concatenate([super()._process_observation(obs), command_block])`.
Command block order = `[vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual, yaw_actual, err_vx, err_vy,
err_speed, err_angle, err_yaw]`.

**Policy expansion:** first layer weight `mlp_extractor.policy_net.0.weight` `[512,228]` → `[512,239]`;
the new 11 columns (228..238) **zero-init** so the expanded policy ignores the command block at
start (output == original on the shared 228 dims). Value net first layer same treatment. Bias,
all other layers, `action_net`, and `log_std` unchanged.

**VecNormalize expansion:** `obs_rms.mean` and `obs_rms.var` `(228,)` → `(239,)`; new 11 entries =
**identity** (mean 0, var 1) so the command block passes through unnormalized initially. `clip_obs`,
`epsilon`, etc. unchanged. `count` unchanged.

**Reuse, don't reinvent:** `scripts/expand_obs_dims.py` (walking 1493→1495) and
`src/training/transfer_utils.py` (the standing→walking **1484→1495 = +11 command block at the end**
transfer — VecNormalizeExtender / PolicyTransfer). The 1484→1495 case is the SAME operation as
228→239 (+11 appended). Adapt it. ⚠️ VERIFY the existing code appends at the END (not inserts mid-
array) — must match `real_turning_env` which concatenates the block last.

## Validation (before handing back to train)
1. Expanded model `observation_space.shape == (239,)`, loads via `PPO.load`.
2. **Parity:** for any obs whose last 11 dims are arbitrary, the expanded policy's deterministic
   action == the original 228-model's action computed on the first 228 dims (zero-init columns →
   command block contributes nothing). Require ~0 difference.
3. Smoke: a few `make_real_turning_env` steps with the expanded model+vecnorm — no crash, obs 239.

## Train commands (USER RUNS — never launch training yourself; see feedback_user_runs_training)
Hand the user, after expansion writes e.g. `models/turn_warmstart.zip` + `models/turn_warmstart.pkl`:
```
# HACK (torso heading -> expect waist twist)
python scripts/train_standing.py --config config/turning/heading_torso.yaml \
  --model models/turn_warmstart.zip --vecnorm models/turn_warmstart.pkl --timesteps <T>
# FIX (pelvis heading -> must turn lower body)
python scripts/train_standing.py --config config/turning/heading_pelvis.yaml \
  --model models/turn_warmstart.zip --vecnorm models/turn_warmstart.pkl --timesteps <T>
```
Note: `train_standing.py` treats `--timesteps` as CUMULATIVE (remaining = T − model.num_timesteps).
Check the expanded model's `num_timesteps` (expansion likely carries the standing model's count);
set `<T>` = that + the desired new steps (~20–30M), or reset it during expansion so T is absolute.

## Gotchas (do NOT "fix" these — they're correct)
- **Morphology inversion** is handled: real freejoint root = PELVIS (base_link), so
  `HeadingYaw(root_is_torso=False, torso_body='0003_8', pelvis_body='base_link')`. `qvel[5]` is the
  pelvis here; torso via `mj_objectVelocity('0003_8')`. Verified reads different bodies.
- `obs_include_com`/`obs_feature_norm` are pinned `false` in the turning configs to hold obs at 239.
- `net_arch [512,512,256] SiLU` in the configs matches the standing model — required for warm-start.

## The other remaining piece (after warm-start)
**WTR sim-eval for the real body:** adapt `scripts/measure_waist_twist.py` — it hardcodes Humanoid-v5
body names `'torso'/'pelvis'/'right_foot'/'left_foot'`. For humanoid_real_v2 use torso=`0003_8`,
pelvis=`base_link`, feet=`0003_2` (R) / `0003_5` (L). WTR = std(torso_yaw − pelvis_yaw) from xquat.

## Then
Train hack+fix → eval WTR in sim (pick best seeds) → record turn trajectories → open-loop replay on
the Pi (supported) + IMU log → the twist-vs-step hardware figure. Done = contribution 2.
