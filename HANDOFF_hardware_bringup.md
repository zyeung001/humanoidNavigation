# Handoff: real-humanoid hardware bring-up, 6/24 → 7/27

**Audience: an LLM (or person) joining cold.** This is the complete context for the
"robot seizures instead of standing" saga: what was tried, what each attempt proved,
the two root causes found, and exactly where things stand. Branch: `real-robot-mjcf`.

## The system in one paragraph

A 17-DOF hobby humanoid (Feetech SCS position servos on a half-duplex 1 Mbaud serial bus,
ICM-20948 IMU on the pelvis, Raspberry Pi at `zyeung001@192.168.86.36`) is driven at 40 Hz by
a PPO standing policy trained in MuJoCo (`models/humanoid_real_v2.xml`) with a deployable
228-dim proprioceptive obs: `proj_grav(3) + ang_vel(3) + (jpos-default)(17) + jvel(17) +
last_action(17)` = 57 dims × 4-frame history. Actions are joint-position targets (radians),
EMA-smoothed with `tau=0.3` (MUST match training), clipped, then mapped to servo units via
`scripts/deploy/sim_real_map.py` + `config/joint_servo_map.yaml`. Inference on the Pi is
NumPy-only from an npz bundle (`scripts/deploy/export_policy.py` bakes in VecNormalize
stats; `numpy_policy.py` runs it; parity vs SB3 verified <1e-4). Deploy loop:
`scripts/deploy/deploy_standing.py`.

## The failure

The policy stands 100% in sim but on hardware enters a ~1–3 Hz whole-body limit cycle
("seizure"). Open-loop replay of recorded trajectories is smooth — the failure is
closed-loop only. Several real code bugs were found and fixed along the way (listed below);
everything else was eliminated with data.

## READ THIS FIRST: the unifying diagnosis (8/5) — loop delay, one problem not three

Everything on this robot oscillates, and it is **one** problem: **loop delay / phase margin.**

| symptom | frequency |
|---|---|
| standing limit cycle (runK, 20 Hz sampling) | 1.96 Hz |
| arm, nobody touching it (`arm_hold4.log`) | 0.9 Hz |
| **sensor-free** offline emulation (`air_loop_emul.py`) | 1.6–1.8 Hz |

That third row is the decisive one: an ideal plant with perfect sensors still oscillates.
**A loop that oscillates with the sensors removed cannot be a sensor-noise problem, and no
reward term or clamp width fixes a phase-margin problem.**

The numbers are all measured:
- Servos: **~50 ms dead time + ~120 ms tau ≈ 170 ms**, bench-measured identically on
  R_knee and R_hip_pitch. Phase lag reaches 180° near **1/(2T) ≈ 2.9 Hz** — every observed
  oscillation sits just under it. At 40 Hz the policy issues ~7 commands before it sees the
  first response.
- 10-bit encoders at `units_per_rad: 195` → 1 count = 0.0051 rad → finite-differenced at
  40 Hz = **0.205 rad/s per count**. Quiet-standing joint speeds are *below one quantum*, so
  the jvel obs channel reads 0 / ±0.205 with nothing between — it measures nothing at the
  operating point, and the sensitivity probe found jvel is the channel the policy amplifies
  most.
- Plus real gear backlash (a limit-cycle generator on its own) and a position-commanded
  servo running its own hidden inner loop at P=15 with negligible damping.

This retrospectively explains two things that looked like separate mysteries: raising P
15→24 in June made the robot **worse** (more proportional gain, less margin), and every
residual iteration produced a control win with a stance failure (they were all tuning the
wrong variable).

**Fix ladder, cheapest first.** All tooling verified present 8/5.
1. **Raise servo D gain** — `servo_stiffness.py --kd` (reg 22), the closest analogue to the
   kd term that keeps other position-controlled robots quiet. No retrain. **Arms first,
   where nothing can fall.** Stock is D=15; step 24, then 32.
2. **Deadband** (regs 26/27). ⚠️ **MEASURED 8/5 and it changes this rung.** Read-back:
   **arms (12–17) are already at 16 units = 0.082 rad = 4.7°; legs+waist (1–11) are at 1
   unit ≈ 0.3°.** For the arms this rung is therefore already applied, and widening further
   would be actively harmful: 0.082 rad is the steady-state tracking-error *floor*, and the
   arm demo's entire claim is tracking + push-return (measured errors 0.056–0.18 rad
   straddle it). It also confirms the 7/31 "ship as-is" decision quantitatively — the
   "servo dead band" it cited as a hardware contribution IS 0.082 rad. The untried half of
   this rung is the legs, but standing is paused. (Both tools read the same registers:
   `servo_stiffness.py` prints them as cwMrg/ccwMrg, `servo_deadband.py` as CW/CCW_DEAD.)
3. **Rate-limit the command path** — `--arm-step-units 4` (from 8). No retrain. ⚠️ but see
   the correction below before touching `--jvel-alpha`.
4. **Drop or hard-filter jvel in the obs** — needs a retrain; principled, that channel is
   mostly counting noise.
5. **20 Hz control instead of 40** — with 170 ms of lag, slower control chases less.
   Retrain with matched config.
6. **Model backlash/deadband in training**, not just first-order lag.

**Two corrections found 8/5 checking the ladder against the code:**
- **`--jvel-alpha 0.15` moves the wrong way.** For `y ← (1-a)y + a·x`, *smaller* alpha is
  *heavier* smoothing: τ ≈ dt(1-a)/a, so at 40 Hz a=0.35 → τ=46 ms but a=0.15 → τ=**142 ms**
  — phase lag at 1.96 Hz goes −30° → −60°, on the channel the policy amplifies most, and it
  is lag the training loop never sees. It buys quiet by spending the margin you are trying
  to recover. The coherent move is to *remove* the channel, not delay it: `--zero-jvel`
  already exists in `deploy_standing.py`, free, no retrain. History agrees — run B (7/27)
  with `--zero-jvel` was calmer (tilt 0.19, d_act −2.4×) but still ~1 Hz: one noisy path
  gone, 170 ms of dead time still there. `deploy_arm_reach.py` has `--jvel-alpha` and
  `--jvel-clamp` but **no `--zero-jvel`** (would need adding).
- **`obs_filter` has never been enabled on the residual line.** The machinery that mirrors
  deploy's real jvel (quantize jpos to encoder resolution → finite-diff → clamp → EMA) is
  built at `standing_env.py:901-916` but is `False` in residual v1/v2/v3. All three trained
  on exact MuJoCo `qvel` + Gaussian `noise_jvel: 0.25`, then deployed against a quantized
  staircase — magnitude roughly right, *character* wrong (white and zero-mean vs
  deterministic and position-correlated). **Ladder item 4 is largely already built, just
  switched off.**

## Timeline of what was done

### 6/24 — smoothness + balance retrains, servo calibration
- Servo horns re-zeroed so each joint's straight pose ≈ its per-joint `center` (not flat 512;
  arms off by up to ~35°). `home.py` ramps to these centers.
- Jitter lever identified: `action_smoothing_tau` fine-tuned AT 0.3 (NOT action_rate_penalty,
  which is dwarfed by the +300 alive bonus). 19M "smooth" model: chatter −56%, 20/20 eval.
- "Balanced" retrain launched (`config/real_humanoid_balanced.yaml`): stance_balance reward
  (even foot loading + centered COM) + push-recovery kicks (lin 0.3 m/s, yaw 1.0 rad/s),
  resumed from the smooth model. Passed the then-current sim gates.

### 6/25 — MJCF mass rebuild (commit `8fc9e43`) — for the turning track only (the later trap)
- Robot weighed for real: **2.046 kg vs 1.303 kg in the MJCF (~36% light)**. Chest
  electronics (Pi, DC-DC, motor controller, terminal) and pelvis battery/breadboard added as
  point masses; feet corrected 80×80 → 120×80 mm. Integrity 13/13
  (`scripts/test_model_integrity.py`).
- **Critically: this corrected model was used only for the turning experiments.** The
  standing policy deployed to hardware still came from the pre-rebuild wrong-mass model.

### 6/25–7/5 — turning track (paper contribution 2) — DONE except Pi replay
- `src/core/heading.py` (`HeadingYaw`), `src/environments/real_turning_env.py` (239-dim =
  228 + 11-command block appended at END). Morphology inversion handled: on the real MJCF the
  freejoint root is the PELVIS (`base_link`), torso is `0003_8` — opposite of Humanoid-v5.
- Warm-start: 228→239 expansion of the (old) balanced standing model → `models/turn_warmstart.zip`
  (zero-init new columns; both arms share it, so the comparison stays controlled).
- Run 1 had a stand-still exploit; fixed via turn-gated reward floor + feet-air + wrong-direction
  penalty. **Run 2 PASSED (7/3): waist-twist ratio hack 0.70 vs fix 0.10 (6.9×, expected
  direction).** Models: `final_real_turn_hack.zip` / `final_real_turn_fix.zip`.
- Record (sim) → replay (Pi, open-loop) → `analyze_turn_replay.py` pipeline committed 7/5.
  **Only remaining: run the replay on the Pi and log the IMU.** (Open-loop, so the standing
  seizure does not block this.)

### 7/1 — ROOT CAUSE #1 found: all standing retrains trained on the wrong-mass MJCF
- Controlled sim test (old vs rebuilt XML, same policy, 3 seeds, deterministic):
  - Quiet standing transfers to the corrected model (3/3, 900/900 steps, symmetric, tilt≈0.99).
  - **Kick at the trained magnitude (lin 0.3 / yaw 1.0): OLD model stood (dip ~0.98); corrected
    model FELL 3/3 in ~1 s (dip ~0.53).** The push-recovery basin the balanced retrain was built
    to create did not exist on real masses → any real perturbation diverges into the limit cycle.
  - First sim configuration reproducing a hardware-consistent failure (overturned the earlier
    "cause is structurally absent from sim" conclusion).
  - Domain-rand mass range [0.8, 1.2] around the wrong nominal cannot cover a 1.57× error.
- Also established: the policy's true settled attractor is **straight, symmetric joints ≈ 0**;
  the earlier "15° forward-bent pose" finding was an artifact of feeding synthetic
  off-distribution obs (jpos=0 at the keyframe). The deploy ramp now targets sim-straight.

### 7/2–7/3 — corrected-mass retrain + gates + export
- Balanced model resumed on the rebuilt XML (config already pointed at it). Note:
  `train_standing.py --timesteps` is CUMULATIVE.
- **Kick gate now PASSES 6/6** on the corrected model (was 3/3 FAIL). Gate is permanent:
  `scripts/gate_standing_kick.py` (settle 100, kick lin 0.3/yaw 1.0 in 6 directions, tilt
  from yaw-invariant projected-gravity cosine — never quat_w, which conflates yaw with tipping).
- Exported `models/real_standing_balanced_policy.npz` (7/3).

### 7/27 — Pi deployment + ROOT CAUSE #2 found: unvalidated servo-bus reads
- New npz pushed to the Pi (the Pi had a **same-named stale npz from 6/24** — backed up as
  `real_standing_balanced_policy_OLDMASS_0624.npz`; md5-verify after every copy). Dry-run OK.
- **Held closed-loop test still seizured** (`--freeze-after`, robot hand-held). Since a held
  robot can't fall, this is not the recovery-basin problem → loop oscillation.
- Sim noise gates re-run on the retrained model: sensitivity probe action-std 0.059 (<0.1
  target), chatter modest → the policy is NOT a noise amplifier in sim. So the driver is
  hardware-side, in the sensing loop.
- Debug logs (run A baseline; run B with `--zero-jvel --zero-angvel`, both hand-held):
  - A: tilt |pg_xy| mean 0.34 rad, body osc ~1.5–1.9 Hz, |av| mean 1.3 rad/s, actions LEAD
    tilt by 2 frames (policy drives it). **76% of frames showed finite-diff joint speeds
    above the servo's physical max (~2.1 rad/s), up to 12.5 rad/s** — impossible → corrupted
    position reads. Also one 171 ms loop stall.
  - B: calmer (tilt 0.19, d_act −2.4×) but still ~1 Hz rocking; 25% impossible-speed frames.
    Velocity channels were zeroed, so the corruption was entering via the **jpos obs** — the
    jvel clamp had been hiding the glitches from jvel while the garbage positions passed through.
- **The bug: `hardware.py read_pos()` accepted any reply starting `FF FF` — no checksum, no
  servo-ID echo, no length check.** On a busy half-duplex bus, corrupted replies delivered
  garbage joint positions into the obs on a quarter to three-quarters of frames.
- Fixes applied (on both Windows tree and Pi; dry-run re-verified):
  1. `hardware.py`: `read_pos`/`read_reg` validate length, ID echo, and checksum; retry on
     mismatch (error byte may be nonzero — position still valid).
  2. `deploy_standing.py ObsBuilder.frame`: plausibility gate on POSITION — a reading implying
     |speed| > `jvel_clamp` (2.5 rad/s) or NaN holds the last good value; a value persisting
     3 frames is accepted (real motion, e.g. a shove, is not suppressed).
  3. NaN-safe ramp reads (`read_units_or`), and a `rej=` counter in the `--debug` line to
     watch the glitch rate live.

### 7/27 (cont.) — bus fix insufficient → ROOT CAUSE #3: servo P-gain left raised since 6/23
- Retest with the bus fix (run D) still seized and ended in a genuine sideways fall (tilt-cut).
  Reject gate active on 67% of frames (partly real violent motion, not just glitches).
- Key signature: at a perfectly calm start (pg upright, |av|=0) the commanded action ramps
  0.29→0.78 rad in <10 frames BEFORE the body moves → the policy's commands drive the plant.
- Deploy-boot hypothesis tested in sim (settle → flush history with zero-action frames →
  resume policy, i.e. the exact Pi startup state): sim stays CALM. Startup protocol is fine.
- The decisive number from that experiment: **in sim the settled policy outputs the same
  0.4–0.7 rad action magnitudes as on hardware, but sim |av|≈0.09 vs hardware |av|≈1.0–1.5.
  Same commands, >10× plant response** → the real actuators execute commands far more
  aggressively than the trained actuator model.
- Register read confirmed why: **all load-bearing servos (1–11) still had P=24 (waist_roll
  also D=30)** — the raised stiffness from the failed 6/23 experiment was never reverted,
  while the training actuator-lag model (τ≈120 ms) was fitted at stock P=15 BEFORE that
  change. Every test since 6/23 ran on a ~60% stiffer plant than trained.
- **Reverted to stock P=15 D=15 on servos 1–11 (7/27, read-back confirmed).** Arms were
  already stock. Next: re-run the held `--freeze-after` test; optionally re-measure one leg's
  step response (`servo_step_response.py`) to confirm τ≈120 ms again.
- Note the compounding-cause structure: wrong mass + corrupted reads + gain mismatch were all
  simultaneously active; each fix looked "ineffective" because the others still forced the
  limit cycle. Also note: every closed-loop test so far was hand-held — a hand grip adds
  coupled human-arm dynamics and unloads the feet, itself off-distribution. The definitive
  test once held behavior is calm: feet on ground, loose overhead tether (not a rigid grip).

### 7/27 (final) — seizure regime gone; balance NOT achieved (run F was hand-held)
- Post-revert run E: fell in 1 s but with NO seizure — a clean forward fall from an 8.6°
  starting lean. Sim static-lean test says the recovery envelope is ≤ ~5.7° (falls ≥ 6.9°),
  and sim's fall-from-8.6° timeline (44 steps) matches hardware (39 frames): **sim now
  quantitatively predicts the hardware.** The seizure phenomenon is gone.
- The 8.6° start came from the deploy startup dead time (~4 s of open-loop ramp + gyro
  calibration during which the robot free-stands uncontrolled and drifts). IMU verified
  honest (held-straight reads pg=[+0.01,+0.03,-1.00]); it stays on the pelvis — that is the
  trained root-body obs frame, and chest lean is visible to the policy via waist/hip jpos.
- **Run F — CORRECTED: the robot was LIGHTLY HAND-HELD the entire run** (initial report
  mistook it for hands-off). Honest reading: the seizure regime is gone (no limit cycle,
  moderate action deltas), but **balance was NOT achieved** — 6 s of wobble (tilt mean
  ~0.2 rad, |av| 0.72 vs sim 0.09) ending in a leftward fall despite light assistance.
  How much wobble is hand-coupling vs plant is unknown from this log.
- Next: (1) ONE genuinely unassisted run using a slack overhead tether — no hand contact —
  as ground truth for what the policy can actually do; (2) 10 s `imu_monitor` average at
  held-straight to check a static left-heavy bias (both falls were leftward → possible
  hip_roll center trim); (3) short robustness fine-tune, user-run (stronger/varied pushes,
  spike obs noise, wider actuator DR); (4) if the unassisted tether run still cannot stand,
  switch to the PD-baseline + RL-residual plan (bounded corrections, structurally unable
  to command large excursions).

### 7/28 — the held-in-AIR test is INVALID; deploy stack proven faithful three ways
- User ran a held-in-the-air check (runG): the robot thrashed at ~1.4 Hz, which looked like
  "the seizure is back." It is not — the test itself cannot discriminate:
  1. **npz parity is exact.** Weights, biases, and VecNormalize stats in
     `real_standing_balanced_policy.npz` are bit-identical to `final_real_standing_balanced.zip`
     (max diff 0 on every layer; post-clip action diff 0.000000). Caution: a naive compare
     shows ~0.9 rad differences because SB3's `predict` clips to the action space and the raw
     npz forward pass doesn't — deploy clips immediately, so always compare post-clip.
  2. **The oscillation reproduces fully offline** (scratchpad `air_loop_emul.py`): the exact
     deploy code path (ObsBuilder → npz → clip/EMA/clip → unit quantize → step clamp) against
     an ideal first-order servo plant, still base, perfect sensors, self-oscillates at
     1.6–1.8 Hz. With servos frozen (pure last_action feedback) it converges dead quiet →
     the loop is sustained by jpos feedback of unloaded servos free-tracking the policy's
     own commands.
  3. **Sim does it too** (scratchpad `sim_air_test.py`: root kinematically clamped, floor
     dropped, trained actuator-lag model active): the trained policy oscillates at 0.8 Hz
     (action osc std 0.14, both seeds). A known-good policy hunts when held rigidly in the
     air — free-tracking joints are outside the training distribution; on the ground the
     loaded plant absorbs the dither.
- runG's 39% "impossible-speed" frames are also explained: unloaded servos genuinely exceed
  the 2.5 rad/s plausibility clamp in air, and the gate's hold-3-frames-then-accept turns
  real fast motion into apparent multi-frame jumps (up to 18.5 rad/s). Not bus garbage.
- **Conclusion: do not use held-in-air as a seizure test.** The code-bug seizure (garbage
  obs, 76% glitch frames) remains fixed. The remaining hardware gap is the policy's own
  ~1 Hz dither plus a real plant livelier than sim. Ground truth is still the unassisted
  slack-tether ground run.
- Mentor-suggested ladder adopted: before more balance attempts, an **arm-stretch policy**
  (6 arm servos, hold a commanded pose, robot seated/supported) is the safe, unfakeable
  closed-loop stack demo — push the arm down, it returns. Trains fast; nothing can fall.

### 8/3–8/4 — residual v3 trained, run, and FAILED (runO): standing is now a negative result
- v3 trained 8/3 (`final_real_standing_residual3.zip`), npz exported 8/4. Sim gates pass as
  always.
- **runO (8/4, tether): 75 frames ≈ 1.9 s**, safety cut at `upright_cos 0.24`, fell BACKWARD
  (pg_x → −0.94) — same direction as runJ. Loop health was fine (39.9 Hz; rej 22/75, much of
  it real fall motion).
- The per-joint widen did exactly half its job: **waist_pitch saturation 25% → 0%** (mean dev
  −0.096 inside its ±0.30 box), but **waist_yaw did not resolve** (mean dev −0.294, 21% at
  the −edge even in a 0.35 box — it simply consumed the extra room), and saturation MOVED to
  new joints: **R_hip_yaw 7% → 31% at edge**, L_shoulder_pitch 20%.
- Widening the box **relocates** saturation instead of removing it — which is what the
  caveat recorded when v3 was built predicted, and what the phase-margin diagnosis explains.
- **STANDING IS PAUSED AS A NEGATIVE RESULT.** v1/v2/v3 all gate 6/6 in sim and fall in ~2 s
  on hardware; deploy `--trim` saturates too. **Do not build a residual4.** The mechanical
  finding stands alongside it: the 3-servo waist stack deflects forward under load with no
  servo leaving position — invisible to an obs of pelvis proj-grav + joint angles.

### 8/4 — arm track: four stacked faults, three fixed (same compounding pattern as 7/27)
- **Both arms had roll↔elbow servos swapped in their slots** by the rebuild (13↔14 right,
  16↔17 left), caught with `probe_sign.py --idx N --delta 0.5` (probing idx 15 articulated
  the forearm). Bus IDs live in servo EEPROM, not cabling — a slot swap, not a wiring error.
- **EEPROM angle limits travelled with the swapped servos**, so each joint got the wrong
  range. `fix_arm_limits.py` LIMITS now follow physical slots.
- **A latching plausibility gate in `deploy_arm_reach.py`**: rejecting a fast reading left
  `prev_jpos` unchanged, so the next frame was equally far away, stayed "bad", and jpos froze
  with jvel railed at the clamp **forever** — the policy could not see a push at all.
  Signature: err locked (0.518 / 0.359 rad) with jvel pinned 2.50 on 93–100% of frames.
  `deploy_standing.py` had the 3-frame-accept escape hatch; the arm script didn't.
- After the fixes: jvel rail 0%, **best err 0.056 rad — better than sim's 0.079**. The stack
  tracks at sim parity.
- ⚠️ **Every arm run before 8/4 was recorded against a frozen observation and measured
  nothing.** There is still **no push-recovery result**.
- **Still open (a): arm zeros are wrong** — arms hang visibly bent at `home` while the policy
  reports only 0.06–0.11 rad error, i.e. servos and policy agree they are on target and the
  target isn't straight. A joint's zero is a horn-mount property, so the slot swap
  invalidated it. `scripts/deploy/calibrate_arm_centers.py` exists for this (torque off, pose
  by hand, medians of 9 reads/joint, prints `center:` YAML).
- **Still open (b): the 0.9 Hz self-oscillation** — see the diagnosis section above.
  **Do zeros BEFORE oscillation tuning: a wrong zero can itself cause oscillation, so tuning
  first means chasing a symptom.**

### 8/5 — arm zeros measured; THREE joints could not reach straight; 6/24 re-zero found missing
- `calibrate_arm_centers.py` had never been run (Pi was offline 8/4) and had a never-executed
  bug: it built `ServoBus()` but called `bus.open() if hasattr(bus, "open") else None` — there
  is no `open()`, the real method is `connect()`, and the `hasattr` guard turned a wrong API
  guess into a silent no-op until the first write hit `self._ser = None`. Fixed; the `finally`
  block is now guarded too (its cleanup threw and buried the real traceback). All ten other
  deploy scripts already used `ServoBus().connect()` — this was the only unrun one.
- **Measured straight pose** (torque off, hand-posed, 15 reads/joint, **spread 0 on every
  joint**): R_sh_pitch 380 (−132 = −38.8°), R_sh_roll 530 (+18), R_elbow 418 (−94),
  L_sh_pitch 648 (+136 = +40.0°), L_sh_roll 509 (−3), L_elbow 385 (−127). The shoulder
  pitches are near-perfect mirrors (−132 / +136); the rolls were already near nominal.
- **THREE of six joints could not reach straight at all** under the firmware limits, because
  every previous limit table assumed straight = 512: servo 12 short by 32 units (9.4°),
  servo 13 short by 94 (27.6°), servo 15 over by 36 (10.6°). The shoulder pitches could only
  travel in a ±30° window centred ~39° away from straight, so **the arm physically could not
  straighten** — exactly the reported symptom. Same silent-clip bug class as the 7/28
  L_shoulder_roll `ctrlrange` fault. Hold this as a candidate contributor to the arm's 0.9 Hz
  oscillation too: a joint pinned on a firmware limit while the policy commands past it is a
  saturation nonlinearity, which generates limit cycles.
- **ORDER MATTERS:** with the old centre of 512 nothing clips (512 is inside every old limit).
  Writing the new centres while the old EEPROM limits are still in the servos is what would
  *create* the clipping. So: `fix_arm_limits.py` FIRST, then the map.
- `fix_arm_limits.py` rewritten: covers all six arm servos (it only did four), every limit is
  the old span shifted by that joint's measured offset (physical arc preserved, spans
  deliberately unchanged), and it now READS BACK each write and refuses to bless a servo whose
  straight pose falls outside the verified range.
- Map updated with the six `center:` values and shifted `servo_limit`s. Verified numerically:
  every span preserved exactly, every centre inside its box, both elbows resolve to clean
  one-sided ranges anchored at straight (R `+0.000..+1.569`, L `−1.569..+0.000`), and the new
  boxes strictly contain the deploy reach box on all six joints — all four demo poses
  (home/stretch/forward/bent) reachable.
- **APPLIED AND VERIFIED ON HARDWARE 8/5.** `fix_arm_limits.py` wrote all six EEPROM ranges
  and read them back clean (elbows report `-0/+306` and `-306/+0`, i.e. straight sits exactly
  on the one-sided stop, as it should). Map pushed, md5 `2953ac0ba36891e3b86ca8c5ca3b9227`.
  End-to-end check through the real deploy code path (`SimRealMap` + live encoder reads, with
  the arms still hand-posed straight): **worst joint 0.005 rad = 0.3° = one encoder count**,
  the resolution floor. The same physical pose read up to 0.68 rad of phantom bend before.
  Confirmed `sim_real_map.py` honours per-joint `center:` in BOTH directions (`rad_to_units`
  and `units_to_rad` both use `self.centers`), so command and obs are corrected together.
- ⚠️ **THE 6/24 WHOLE-BODY RE-ZERO IS MISSING FROM THE REPO.** That session recorded
  re-zeroing all 17 joints into per-joint `center:` values with shifted limits plus a
  `joint_servo_map.yaml.bak`. Today the map at HEAD *and* in the working tree has exactly ONE
  `center:` (`waist_roll: 485`, from 6/18) and there is no `.bak`; the Pi's copy is
  byte-identical, so it is gone there too. The magnitudes match what 8/5 just re-measured
  (6/24 said "±118 units, ~35° on the arms"; 8/5 measured 38.8°/40.0°), i.e. **we re-measured
  what 6/24 already had.** Mechanism is the one that session flagged itself: the calibration
  was left uncommitted and Pi-only, and this repo syncs the Pi by scp, so a later local→Pi
  push of the same file overwrites it. **Open question with real stakes: if the LEGS also had
  offsets, runs J–O ran on wrong leg zeros and the standing negative result is confounded.
  Unverified — `calibrate_arm_centers.py` covers only idx 11–16.**

## Verified-and-eliminated list (do not re-investigate without new evidence)

Actuator bandwidth (clean 1st-order step response; modeled as `actuator_lag` delay+tau in
training), npz↔SB3 parity, VecNormalize baking, obs frame order/history padding, IMU axis
remap + units + gyro bias, servo signs/limits/centers (bench-verified; IDs 7,16 sign-inverted),
40 Hz loop feasibility (busy ~21 ms/25 ms), tau mismatch, log_std explosion (clamped;
position-servo action space needs tight std ~0.22 rad — wide std gives 0% deterministic eval).

## Current state + next steps (in order)

**AS OF 8/5 the track order is:**

| track | status | next action |
|---|---|---|
| **ARM** | closest to a result; 3 faults fixed 8/4, tracks at sim parity | **zeros first** (`calibrate_arm_centers.py`, on the Pi, md5 `23b2f136d5a7ee92ee35a0d5a995b1a0`), **then** the D-gain ladder. Goal = push-return demo. |
| **TURNING** | passed in sim 7/3; cheapest remaining deliverable | one open-loop Pi replay + IMU log. Doesn't need balance, can't fall. |
| **STANDING** | **PAUSED — negative result** | none. Do not build a residual4. |

Pi is back online (was down 8/4). Everything below is the historical detail behind that table.


0. **RESULT 7/28: the tether run FAILED** — drops within a couple of seconds every time.
   Decision taken: pure end-to-end RL cannot stand this hardware → **PD-baseline +
   RL-residual, now implemented** (see below). Design probes (`sim_residual_probe.py`,
   scratchpad): the settled standing COMMAND (≈ pose + gravity offsets) stands frozen
   open-loop 400/400; the balanced policy clamped to baseline ± 0.10–0.25 rad stands
   QUIETER than unclamped → warm start valid; kick recovery at lin 0.3 is impossible under
   any clamp → residual training uses milder pushes (lin 0.15 / yaw 0.5). Bonus bug: the
   post-mass-fix settled attractor is NOT straight (one knee −0.505 rad) — deploy's
   ramp-to-straight started every hardware run off-attractor; residual mode ramps to the
   baseline pose.
   Implemented (lint-clean; deploy + config pushed to Pi, md5-verified):
   - `standing_env.py`: `residual_clamp` + `residual_baseline` — clips the action to
     baseline ± clamp at the top of `_process_action` (before the EMA); off by default,
     action space unchanged so the balanced model resumes directly.
   - `config/real_humanoid_residual.yaml`: clamp 0.2, measured baseline vector, milder
     pushes, `*_residual` outputs, +10M steps.
   - `deploy_standing.py --residual-config <yaml>`: mirrors the clamp before the EMA and
     ramps to the baseline pose.
1. **RESIDUAL FINE-TUNE DONE 7/28-29 (user ran it, 90,021,888 steps) — GATED AND SHIPPED.**
   Note the base accounting: the balanced model was at 80,019,456 steps (35M smooth +
   ~25M balanced 6/24 + ~20M corrected-mass retrain 7/2 — that last 20M IS the model in
   use, nothing was wasted), so `--timesteps 90000000` added ~10M.
   Sim verification (all PASS):
   - Kick gate at the trained magnitude (lin 0.15 / yaw 0.5): quiet PASS + **6/6 kicks
     STOOD, tilt dips >= 0.994** (vs the balanced model's 0.975-0.989 — calmer).
   - Quiet standing 800 steps: tilt min 0.994, **|av| mean 0.030 — 3x calmer than the
     balanced model's 0.09**.
   - Clamp verified: steady-state max|applied − baseline| = 0.200 exactly (the policy uses
     its full envelope). A transient 0.295 in the first ~30 steps is only the tau-EMA
     warming up from zeros — identical in training and deploy, not a mismatch.
   - `models/real_standing_residual_policy.npz` exported (parity 5.7e-07), **md5-verified
     on the Pi** together with deploy_standing.py and the residual config.
2. **RUN I RESULT (7/30, slack tether): residual v1 = control WIN, stance FAIL.** 17 s on
   the tether (previous unassisted best ~2 s), d_act mean 0.060 (half of every prior run),
   |av| 0.34 (vs 0.7–1.0), glitch frames 5% — the bounded-corrections mechanism works. BUT
   it hung on the tether CROOKED: steady ~13–16 deg lean, never centered, "legs look bent
   backwards" (that is the baseline's R_hip_pitch +0.297 — 85% of the joint limit — plus a
   29 deg waist_yaw twist; note an earlier record mislabeled the −0.505 baseline entry as
   a knee — it is WAIST_YAW). Diagnosis: v1 anchored the clamp to sim's measured settled
   stance, and sim's static balance point is ~10 deg off the real robot's. A ±0.2 rad
   policy cannot fix a structurally leaning anchor.
   **RESIDUAL v2 BUILT (7/30): straight baseline + baseline randomization + on-robot trim.**
   - Sim 7/30: frozen STRAIGHT (all-zeros command) stands 500/500 (tilt 0.999, more
     upright than v1's baseline); frozen symmetrized-v1 also stands; the v1 policy moved
     to a new baseline WITHOUT retraining FALLS (old/new waist_yaw boxes are disjoint) —
     so a ~10M retrain is required, resumed from v1.
   - `standing_env.py residual_baseline_rand`: per-episode uniform baseline offset
     (±0.05 rad) — the policy must find true vertical from proj_grav instead of trusting
     the anchor (exactly the sim-vs-real static error runI exposed), and it makes small
     on-robot trims in-distribution.
   - `deploy_standing.py --trim "R_hip_pitch=-2,L_hip_pitch=-2"`: named-joint degree
     offsets added to the baseline (shifts ramp target + clamp box together) — the
     on-robot static calibration knob. Keep <=3 deg per iteration.
   - `config/real_humanoid_residual2.yaml` (zeros baseline, rand 0.05, bias_jpos 0.04,
     *_residual2 outputs). Deploy script + config md5-verified on the Pi.
   **ORDER OF OPERATIONS:**
   a. 2-minute HARDWARE pre-check BEFORE the retrain (user): `python3 scripts/deploy/home.py --hold`
      ramps to straight and holds stiff — does the robot stand at straight (possibly with
      a fingertip / slight lean)? If yes, the zeros baseline is validated on the REAL
      robot, not just sim. If it clearly cannot, find the standing trim first (home.py
      --waist-lean or physically note the lean) and bake it into the baseline instead.
   b. USER RUNS (cumulative; v1 is at 90,021,888):
      `python scripts/train_standing.py --config config/real_humanoid_residual2.yaml --model models/final_real_standing_residual.zip --vecnorm models/vecnorm_real_standing_residual.pkl --timesteps 100000000`
   c. **DONE 7/31 — residual2 GATED AND EXPORTED; only the tether run remains.**
      Retrain loaded at **100,024,320 steps** (v1 was 90,021,888 -> ~10M added, as intended);
      obs 228 / act 17, log_std parked at the -2.0 floor (sigma ~0.135 rad).
      - Kick gate (lin 0.15 / yaw 0.5, residual2 config): **quiet PASS + 6/6 kicks STOOD**,
        tilt dips 0.991-0.997. Baseline confirmed loaded as zeros (`baseline max|.|=0.000`).
      - Quiet standing 800 steps: survived 800/800, tilt min 0.9941, **root |av| mean 0.019
        rad/s** (v1 0.030, balanced 0.09 — calmest model yet), d_act 0.0007 rad/frame.
      - **The policy saturates its envelope: steady-state max|applied - baseline| = 0.2000
        exactly.** Settled sim stance rides the box edges asymmetrically: waist_roll +11.8,
        waist_yaw -11.1, R_hip_yaw +11.2, R_hip_pitch +11.1, L_hip_roll -11.5, L_hip_yaw
        +6.7, waist_pitch -6.0 deg (arms +-11 deg). So on hardware the robot will visibly
        move away from straight when the policy takes over after the ramp — that is normal,
        not a fault. It also means a correction needed beyond 0.2 rad in an already-saturated
        direction is unreachable: that is exactly what `--trim` is for (shifts the box).
      - `models/real_standing_residual2_policy.npz` exported, parity 5.4e-07.
        md5 `fc0183ba14b4cb17cbcd4ef1316363a6`.
      Remaining: md5-push the npz, then tether run with
      `--residual-config config/real_humanoid_residual2.yaml`, iterating `--trim` (<=3 deg
      per step) against whichever way it leans.
   d. **HARDWARE 7/31-8/1, runs J/K/L/M/N -- residual2 does NOT stand; cause localized to
      the WAIST by two independent routes.** All on a fall-catch tether, feet on ground.
      - runJ (no trim, 13 s): pg_x -0.140 (leaning BACK ~8 deg) vs sim's +0.094 (leaning
        FORWARD) -- a 13 deg disagreement about where this robot balances. Lateral was
        FIXED though: pg_y +0.011 vs v1's 13-16 deg crooked hang, so v2's symmetric straight
        baseline did its job on the roll axis. |av| 0.346, d_act 0.058 -- identical to v1,
        i.e. no control regression.
      - runK (-3 deg hip-pitch trim, 11 s): lean -0.071, tilt 10.6->7.0 deg, |av| 0.252.
        Trim gain ~1.33 body-tilt per unit trim, near-linear. At 20 Hz sampling the
        oscillation question resolved: a real ~1.96 Hz LATERAL limit cycle exists (pg_y 60%
        of power in the 0.8-2.5 Hz band), bounded, far below the old seizure amplitude.
      - runL (-6 deg, 45 s -- longest run to date): **trim SATURATED.** Doubling it changed
        the lean not at all (-0.075 vs -0.071) while clamp saturation went 96% -> 100%.
        1.96 Hz power dropped (pg_y 60->40%). Arms detached at t~40 s (unscrewed horns).
        Deploy-time trim has a ceiling; the clamp box, not the anchor, became binding.
      - runM: instrumented script was NOT pushed (md5 check skipped) -- run lost, and a
        printed part snapped in the fall. **Always verify the md5 BEFORE running.** It did
        show the fingertip-through-startup fix works: first frame pg=[+0.02,-0.02,-1.00]
        vs -0.17 on every prior run. It then diverged forward ~1 s after release, which
        largely kills the "it only fails because it inherits a fall" explanation.
      - **runN (8/1, post arm-screw fix, 299 frames) -- THE MEASUREMENT.** Per-joint
        saturation summary: waist_yaw mean dev -0.171 and pinned at its edge **46%** of
        frames, waist_pitch -0.121 / 25%, L_shoulder_roll 13%, R_shoulder_pitch 11%,
        R_hip_yaw 7%, R_hip_roll 3%, rest ~0%. **Saturation is entirely ONE-SIDED** (no
        joint alternates edges) -> not a relay/bang-bang limit cycle, so more authority is
        a coherent fix rather than a dangerous one. And it is the WAIST, only the waist.
      - **Converging mechanical evidence (user, hands on the robot): the chest deflects
        forward under load with NO servo leaving position** -- structural compliance in the
        3-servo waist stack (horn splines, printed brackets, gear lash in series). That is
        INVISIBLE to a policy whose obs is pelvis proj_grav + joint angles, and in a rigid
        sim the chest pose is a pure function of joint angles. It also predicts the sign of
        runJ's puzzle: chest sags forward -> COM forward -> pelvis must rotate BACK to
        compensate -> IMU reads a backward lean, which is exactly what was measured.
        Waist horn screws have since been added. Robot is also slightly back-heavy from a
        pelvis-mounted breadboard, but that is quantitatively minor: real offset 44 mm vs
        30 mm modelled, 68 g in a 2046 g robot = **0.47 mm** of whole-body COM shift.
      - Also settled: `home.py --hold` at straight stands only if hand-angled, and the
        robot pitches back during the ~3.5 s open-loop startup (ramp + gyro calib) unless
        steadied. Steady it by fingertip through startup, release at `Closed loop running`.
   e. **RESIDUAL v3 (per-joint clamp) — BUILT 8/2, TRAINED 8/3, RUN 8/4, FAILED.** See the
      "8/3–8/4 — residual v3 … negative result" section below for runO. The caveat recorded
      when it was built turned out to be the right call. Detail of what was built:
      `residual_clamp` now accepts a scalar OR a 17-vector in both `standing_env.py` and
      `deploy_standing.py` (4 call sites in the env; the `reset()` one is easy to miss).
      `config/real_humanoid_residual3.yaml`: waist_pitch 0.30, waist_yaw 0.35, the other 15
      unchanged at 0.20 -- a uniform widen would hand the legs excursion room they never
      asked for, and bounded leg excursion is what made v1 a control win. Baseline stays
      zeros and baseline_rand stays 0.05, so exactly one variable changes vs v2. Verified:
      clamp applies per joint in both directions (waist reaches +-0.30/+-0.35, others stop
      at 0.20 or their ctrlrange), residual2 resumes (obs 228, act 17, 100,024,320 steps),
      repo lint clean. **Caveat recorded honestly: if the waist COMPLIANCE is the real
      driver, no clamp width fixes it and the measured saturation is a symptom -- commanded
      waist motion not producing body response, so the policy pushes until it saturates.**
      Also note `deploy_standing.py` now md5 `ab2e9474154c387a988c440e33675f1a` (per-joint
      logging + summary table on exit; the table prints via `finally`, so stop runs with
      Ctrl-C, not by closing the SSH session).
   **Real-world-data doctrine (user asked):** full on-robot RL is out (10M steps @ 40 Hz =
   70+ days of robot time); what IS practical: on-robot trim calibration (minutes/trial,
   no retrain), real runs falsifying sim's balance point (what runI just did), and later
   real2sim COM fitting from logs (FSRs/center-of-pressure are the right instrument —
   user is deferring FSRs to the walking stage). Real data for the static truth, sim for
   the dynamic samples.
3. **Arm-stretch stack-demo rung — NOW THE ACTIVE TRACK.** Status as of 8/5 is the
   "8/4 — arm track" section below (3 faults fixed, tracks at sim parity, zeros + 0.9 Hz
   oscillation open, no push-recovery result yet). History of how it was built: (`src/environments/arm_reach_env.py`, `scripts/train_arm_reach.py`
   [CPU, ~3M steps], `scripts/deploy/deploy_arm_reach.py` — the deploy script IS now
   md5-verified on the Pi.) Obs 96 / act 6 (new fingerprint); jpos ABSOLUTE (0 = straight).
   - The user's first 3M-step train plateaued at 0.13-0.27 rad steady tracking error (even
     at the trivial home pose). Root cause: the original tracking reward `3*exp(-6*mse)`
     keeps ~80% of its value at 0.2 rad error — no gradient to finish. FIXED in the env
     defaults: two-scale kernel `2*exp(-8*mse) + 3*exp(-80*mse)` (the sharp term only pays
     near zero error). A 400k sanity train of the new reward was running as of this
     writing; regardless of it, the fix is committed in the env.
   - RETRAIN DONE 7/31 (3.0M steps, sharpened two-scale reward). Steady pose error:
     home 0.079 / stretch 0.117 / forward 0.182 / bent 0.107 rad (old model
     0.27/0.27/0.22/0.13 — better everywhere); push recovery excellent (arm kicked away
     returns under 0.1 rad in 0.1-0.5 s). Misses the 0.05 rad bar.
     **DECISION 7/31: SHIP AS-IS, do not retrain.** 0.08-0.18 rad is 4.5-10.4 deg, which is
     at or below what this hardware contributes anyway (servo dead band, horn-zero error,
     and gravity droop on 0.226 N*m arm servos) — a sim policy at 0.05 rad would not be
     distinguishable on the real arm. The demo's claim is closed-loop push-return, and that
     is the strong part. The worst pose (forward, 0.182) is also the most gravity-loaded, so
     it reads as a torque/feasibility limit rather than a reward-shaping one; more steps are
     unlikely to buy it. `models/arm_reach_policy.npz` exported (parity 9.5e-07,
     md5 `37fe801ef5f77dcc4acc1362ffc3f402`).
     If a later cycle wants precision anyway: `python scripts/train_arm_reach.py --timesteps
     6000000` (fresh run, NOT cumulative — the script always starts a new PPO; it would
     overwrite `models/final_arm_reach.zip`, so back the current one up first).
   - Then: `python scripts/deploy/export_policy.py --model models/final_arm_reach.zip --vecnorm models/vecnorm_arm_reach.pkl --out models/arm_reach_policy.npz`
     -> md5-push the npz to the Pi -> robot seated ->
     `python3 scripts/deploy/deploy_arm_reach.py --policy-npz models/arm_reach_policy.npz --pose stretch --debug`
     (then `--cycle 4`; push the arm away, it must return — the unfakeable closed-loop demo).
   - Env/model context: arm servos are weak (0.226 N*m force cap) so training targets are
     filtered by a reach-and-hold feasibility sim; sim "arm out" is R negative / L positive
     roll. **MODEL BUG (user's catch) fixed:** v2's left shoulder roll range was
     mis-converted to [-2.007, 0] = zero outward travel (commands silently
     ctrlrange-clipped: zero force, zero motion — how to spot this bug class); the arm
     track uses `models/humanoid_real_v2_armfix.xml` (Revolute 5 -> [-0.4363, +1.5708],
     mirror of the right arm; integrity gate 13/13). The STANDING track keeps the old v2
     xml on purpose — changing ctrlrange changes the action space and would break resuming
     the balanced/residual models; fold the fix in at the next from-scratch standing cycle.
     First HW arm run: verify the left arm swings OUT on "stretch" (a map comment says
     servo16 "+ = in"; if it swings inward, the map sign for servo 16 is wrong).
4. Static left-bias check: 10 s `imu_monitor` average at held-straight; if pg_y biased,
   trim hip_roll centers in `config/joint_servo_map.yaml`.
5. Turning contribution: open-loop turn replay on the Pi + IMU log (pipeline committed),
   independent of standing.
6. Model renders (straight / training keyframe / settled residual stance / arm poses) are
   published at https://claude.ai/code/artifact/b50f71f0-da84-46db-bdb4-e5c9ff4cdd5f .

## Gotchas for whoever picks this up

- Obs dims fingerprint the track: 228 = deployable standing (this), 239 = real turning,
  1484/1495 = the separate Humanoid-v5 std pipeline. Don't mix models/vecnorms across tracks.
- `--timesteps` is cumulative when resuming.
- The user runs ALL training launches themselves — including short sanity/smoke trains
  (re-affirmed 7/29); agents prep configs + hand over the exact command. Read-only
  eval/diagnostics are fine to run.
- Reward-shaping lesson (arm task, 7/29): a lone soft exp kernel lets PPO plateau far from
  the goal; pair a coarse kernel with a sharp near-zero precision term.
- Old wrong-mass XML recoverable via `git show 8fc9e43^:models/humanoid_real_v2.xml`
  (`models/humanoid_real_v2.xml.bak` is NOT that version).
- Upright/tip-over metrics: always yaw-invariant projected-gravity tilt, never quat_w.
- The Pi repo (`~/humanoidnavigation`) is synced by scp, not git pull — always md5-compare
  after copying; a same-named stale npz already caused one wasted hardware test.
- **Recompute the md5 at push time; never quote a remembered one.** A stale quoted hash and a
  skipped scp each cost a hardware run, one of which broke a printed part. Proof it keeps
  happening: on 8/5 the hashes recorded in memory for `calibrate_arm_centers.py`
  (`9ccb436c…`) and `deploy_standing.py` (`ab2e9474…`) were *both* already stale.
- Deploy summary tables print via `finally` — stop runs with **Ctrl-C**, not by closing the
  SSH session, or you lose the table.
- Standing runs need a fingertip through the ~3.5 s open-loop startup (ramp + gyro calib);
  release at `Closed loop running`. Without it the robot pitches back before the policy
  engages and inherits a fall.
- Comparing a file across machines: md5 differs on line endings alone (the Windows tree is
  CRLF, the Pi copy LF). `servo_stiffness.py` looked out of sync on 8/5 for exactly this
  reason and was byte-identical in content — diff before re-pushing.
