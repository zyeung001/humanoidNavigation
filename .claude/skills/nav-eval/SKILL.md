---
name: nav-eval
description: Evaluate a finished navigation/walking model across the full maze suite (corridor, L, U, procedural), compute the composite, record video, and apply the gait gate. Use whenever the user reports a finished run, asks whether it worked, asks to run eval, or pastes a "Saved final model:" line.
---

# nav-eval

## When to trigger

Fire when the user has a trained nav model and wants a verdict. Concrete examples (drawn from real sessions):

- "i finished phase 1 ... how do i see if it worked?"
- "i finished running this" / "i finished the training"
- "run eval" / "just evaluate" / "whats the command" (after a run)
- A pasted line like `Saved final model: runs/nav_phase3_v4\model_final.zip`
- "i dont think v5b is working — take a look" (eval first, then hand to `ppo-triage`)

Do NOT trigger for: mid-training log dumps with no model to score (→ `ppo-triage`), or pre-launch command requests (→ `nav-train-launch`).

## Inputs

Model `.zip` + matching `vecnorm .pkl` (same dir, `model_final.zip` ↔ `vecnorm_final.pkl`). If the user named only a run dir, use **Glob** to resolve the pair; if ambiguous, ask once.

## Run the suite

All four maze classes, 50 episodes each. One line per command ([[feedback_command_format]]), output to D: ([[feedback_output_to_d_drive]]), `$env:PYTHONUTF8=1` once per shell (eval_nav.py prints `❯` → cp1252 crash). Eval is long; launch with **run_in_background** and poll.

```
$env:PYTHONUTF8=1; python scripts/eval_nav.py --model <M> --vecnorm <V> --maze-type corridor --episodes 50 --record --log D:\hn_models\eval\<run>\corridor.jsonl --video-dir D:\hn_models\eval\<run>\video_corridor
```
Repeat with `--maze-type L`, `--maze-type U`, `--maze-type procedural`.

## Pass criteria (hard-coded in eval_nav.py — do not invent)

- **PASS**: goal-reach ≥ 80% AND mean torso height ≥ 1.20 m
- **HOLLOW**: goal-reach ≥ 80% but mean height < 1.20 m — crawling to goals, NOT a pass ([[nav_phase3_v7b_gait_collapse]])
- **FAIL**: goal-reach < 80%
- **Composite** = mean of the 4 per-maze goal-reach rates

## Verification step (run before reporting — do not skip)

1. **All four logs exist and are non-empty.** Glob `D:\hn_models\eval\<run>\*.jsonl`; expect 4 files. A missing/zero-byte log = that maze crashed (often the cp1252 `❯` error or a bad vecnorm/model dim mismatch) — re-run that maze, don't report partial.
2. **Episode count matches.** Each jsonl has 50 lines (Read/Bash count). Fewer → eval aborted early; investigate before scoring.
3. **Height field is populated.** Confirm `nav_height` is present and non-null in the logs. If null on every episode, the gait gate is meaningless → flag it, don't silently report PASS (this is exactly the hole that hid the v7b crawl).
4. **Verdict string matches the numbers.** Recompute PASS/HOLLOW/FAIL yourself from goal-rate + mean height and confirm it equals eval_nav.py's printed verdict. Mismatch = stale binary or arg error.
5. **Sanity vs. prior version.** A ±0 composite with a totally different failure profile, or a >0.3 jump, usually means wrong model/vecnorm pair — re-check inputs before believing it.

Only after 1–5 pass: emit the `| Maze | prev | this | Δ | Pass |` table + composite + X/4, with per-maze mean/min height and termination breakdown. Lead with HOLLOW if any maze is hollow.

## Handoff

→ `log-run-memory` to persist the `nav_phaseX_vN_eval.md` entry. → `ppo-triage` if the user also pasted training logs.

## Tools this skill needs

- **Bash / PowerShell** — run eval_nav.py; prefer `run_in_background` (4×50 episodes is slow) then poll.
- **Glob** — resolve model↔vecnorm pairs from a run dir; verify the 4 expected jsonl outputs exist.
- **Read** (or Bash line-count) — verification step: episode counts, `nav_height` presence, verdict line.
- **Monitor** (deferred tool) — would improve this: tail the background eval and wake on completion instead of fixed-interval polling. Fetch via ToolSearch when needed.
- Gap / nice-to-have: no programmatic way to view recorded `.mp4` gait. The 1.20 m height gate is the proxy; if the user disputes a HOLLOW/PASS call, ask them to watch the video — the skill cannot inspect it.
