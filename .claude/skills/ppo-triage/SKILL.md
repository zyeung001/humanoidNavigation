---
name: ppo-triage
description: Diagnose PPO training health from a pasted rollout/train log dump. Use when the user pastes an SB3 metrics table and asks whether the run is healthy, should be killed, or "i dont think vN is working" / "ep_len not >300".
---

# ppo-triage

## When to trigger

- "i dont think v5b is working" + a pasted `| rollout/ | ... |` block
- "version v5b finished but ep_len_mean was not > 300"
- "[Step 30,000,000] Walking Metrics: Avg velocity ..." style dumps
- "is this working?" / "should I kill this?" with metrics attached
- "why is it not working still?" after a launch

Do NOT trigger for: a finished model the user wants scored (→ `nav-eval`), or a request for the next command (→ `nav-train-launch`).

## Diagnostic order — check in THIS sequence (do not jump to reward shaping)

1. **explained_variance < 0** → VF harmful. Freeze policy, train VF only. Not a reward problem.
2. **clip_fraction > 0.9 sustained** → gradient signal clipped to noise. Cause = NUMBER of gradient steps, not LR (12 minibatch×3 epoch=36 steps→95.8% clip; full-batch 1×3=3→healthy). LR cuts alone won't fix; need full-batch + permanent policy scaling.
3. **approx_kl > 50** → high KL; re-check EV and clip for the real cause.
4. **approx_kl < 0.001 AND clip_fraction = 0 sustained** → policy frozen: LR too low / past convergence ([[fresh_lr_overtraining]]).
5. **std > 1.5** → clamp log_std (must be 0.0 for walking transfer — [[log_std_max_transfer_must_be_zero]]).
6. **ep_rew_mean > 2000** → rewards mis-scaled (target per-step ~2–5, episode ~500).

## Critical caveats (apply BEFORE declaring failure)

- **Ramp/scaling phases**: during Phase 2 (vf_rampup) and Phase 3 (policy_max_scale=0.5), SB3's reported clip_fraction/approx_kl are PRE-interpolation and look alarmingly high. **Ignore them.** Judge on vel_error, ep_rew_mean, ep_len_mean, curriculum stage. Applied KL ≈ scale²×reported.
- **Healthy PPO can still learn nothing.** Stage-0 / standing-prior trap: KL≈0.02, clip≈0.04, EV≈0.97 (textbook healthy) while vel_err flat, 0% Stage-0, height stuck, pg_loss positive. Healthy PPO + flat task signal = trap, not mechanics ([[walking_stage0_trap_real_cause]], [[nav_phase3_v8_collapse]]).
- **From-scratch bootstrap**: ~0–120k steps, clip≈0, height rising 0.8→0.98 is NORMAL ([[fromscratch_walking_bootstrap]]). Don't kill before ~300k.

## Steady-state healthy ranges

approx_kl 0.01–0.03 · clip_fraction < 0.3 · explained_variance > 0.8 · std ≈ 1.0 · per-step reward 2–5.

## Verification step (before giving the verdict)

1. **Identify the phase first.** Get current timestep and config'd vf_warmup_steps / vf_rampup_steps (Read the run's config or training.jsonl). A "KL=34000" verdict is wrong if the step count puts it inside ramp — confirm the phase before reading KL/clip at all.
2. **Confirm the task metric, not just PPO internals.** Locate vel_error / nav_progress_arc / curriculum stage in the metrics file. If you only have PPO internals and they look healthy, you CANNOT rule out the trap — say so and ask for the task metric rather than declaring HEALTHY.
3. **Check the trend, not one row.** One pasted row is a point; pull the last N rows from `metrics/training.jsonl` (Read/Bash) to confirm flat-vs-rising. "Flat" needs ≥2 samples ≥0.5M apart.
4. **State the single deciding metric.** The verdict must name one metric and its value; if you can't, verification failed — gather more before answering.

## Output

Verdict: **HEALTHY / KILL / WATCH** + the one deciding metric + concrete next action. If KILL, hand the corrected launch to `nav-train-launch`.

## Tools this skill needs

- **Read** — the run's config YAML (phase boundaries) and tail of `metrics/training.jsonl` (trend, task metric). This is what makes verification possible vs. guessing from one pasted row.
- **Bash / PowerShell** — line-count / extract last N jsonl rows when the file is large.
- **Glob** — locate the run's metrics dir when the user gives only a run name.
- Gap: the user often pastes a single SB3 console block with NO task metric (vel_err/progress absent from the default print). The skill's main failure mode is over-trusting that block — verification step 2 forces asking for the metrics file instead.
