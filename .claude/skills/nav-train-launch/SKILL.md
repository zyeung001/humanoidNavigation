---
name: nav-train-launch
description: Construct the correct next training launch command — walking transfer from standing, nav phase resume, or obs-dim expansion. Use when the user asks for the command to run or what to do next after an eval.
---

# nav-train-launch

## When to trigger

- "whats the command" / "give me the command" / "the command in one line"
- "what do i do next?" after an eval verdict
- "i want something that will work. give me the best option"
- "ok now with the trained walker how do i start the nav phases?"
- A `ppo-triage` KILL verdict that needs a corrected relaunch

Do NOT trigger for: scoring a finished model (→ `nav-eval`), or interpreting live logs (→ `ppo-triage`).

## Output rules (always)

One line, no `\` continuations ([[feedback_command_format]]). Prefix `$env:PYTHONUTF8=1; `. Output to D: ([[feedback_output_to_d_drive]]). Confirm model/vecnorm paths exist before emitting.

## Case A — Walking from standing (b91a244 recipe, FROZEN — do not "improve")

Proven walker = `b91a244`; recipe lives in `config/training_config.yaml` `walking:` section. **Do NOT pass `--reinit-action-head`; do NOT raise Stage-0 speed or tighten termination** — those three deltas ARE the Stage-0 trap ([[walking_stage0_trap_real_cause]]). Causal params must equal b91a244: speed stages `[0.15,0.3,0.6,1.0]`, termination_height_threshold 0.60, early_termination_protection 150, recovery_window 80, ent_coef 0.005, policy_max_scale 0.5, log_std_max 0.0, vf_warmup 250000, vf_rampup 500000.

```
$env:PYTHONUTF8=1; $env:HN_MODELS_DIR="D:\hn_models"; python scripts/train_walking.py --from-standing --model models/best_standing_model.zip --vecnorm models/vecnorm.pkl --timesteps 30000000
```
(train_walking.py reads HN_MODELS_DIR; train_nav.py does NOT.)

## Case B — Nav phase resume

`--model` WITHOUT standing flags + matching `--vecnorm`. b91a244 walker is 1493-dim → run `scripts/expand_obs_dims.py` first (1493→1495). train_nav.py ignores HN_MODELS_DIR → pass `--output-dir D:\hn_models\runs\<name>` explicitly.

```
$env:PYTHONUTF8=1; python scripts/train_nav.py --model <prev>\model_final.zip --vecnorm <prev>\vecnorm_final.pkl --output-dir D:\hn_models\runs\<name> --timesteps 30000000 --procedural --upright-bonus-weight 1.0
```

## Phase knobs

- **Phase 1 (fresh from walker)**: RAMP — `--upright-bonus-start 0.2 --upright-bonus-weight 1.0 --upright-ramp-steps <N> --fall-height-threshold 0.85`. Crawl is born in Phase 1; ramp prevents v8's standing-local-optimum ([[nav_crawl_fix_implemented]]).
- **Phase 2+**: CONSTANT `--upright-bonus-weight 1.0`, no ramp.
- **Best reward profile (v7)**: `--progress-weight 15.0 --goal-bonus 200.0 --vel-proj-weight 3.0`, keep `--collision-penalty 25.0`.
- **Mazes**: pure `--procedural` unless deliberately specializing — any fixed-maze mix from a procedural model catastrophically forgets.

## Verification step (before emitting the command)

1. **Paths exist.** Glob/Read-check the `--model` and `--vecnorm` files resolve. A command with a typo'd path wastes a long run — confirm, don't assume.
2. **Obs-dim match.** If source is the 1493-dim walker and target is nav (1495), the command MUST be preceded by an `expand_obs_dims.py` step. Emit both, in order, or the run dies at load.
3. **Recipe diff against b91a244 (Case A only).** Read `config/training_config.yaml` `walking:` and confirm the causal params above are unchanged. If config drifted, say so and give the one-line fix — do NOT emit a launch on a drifted config.
4. **Output dir is on D: and new.** Reject `runs/...` (C:) defaults; confirm `<name>` isn't an existing dir you'd overwrite.
5. **One line, copy-paste clean.** Re-read the emitted command: no `\`, no stray newline, correct `;` separators for PowerShell.

## Watchdog to tell the user

No auto-abort by design. Watch `reward/progress` / `nav_progress_arc` in `D:\hn_models\runs\<name>\metrics\training.jsonl`; if not rising by ~1M steps (Phase 1), abort — PPO can look healthy while learning nothing.

## Tools this skill needs

- **Read** — `config/training_config.yaml` to diff the live recipe vs b91a244 (verification 3); this is the single highest-value check, it directly prevents the repeated Stage-0-trap relaunches.
- **Glob** — confirm model/vecnorm paths and detect an existing `<name>` output dir (verification 1, 4).
- **Bash / PowerShell** — only to stat files / inspect a checkpoint's obs dim if ambiguous; the user runs the actual training, not this skill.
- Gap: obs dimensionality (1493 vs 1495) isn't always evident from the filename. If unverifiable from context, instruct the user to run `expand_obs_dims.py` defensively rather than guessing.
