# Archived Variant Configs

These configs are kept for research history. **Do not use them for new training.**
The current canonical pipeline lives at `config/variants/01..05_*.yaml` —
see `config/README.md` for the step-by-step recipe.

## Yaw exploration (3/31 — 4/15/2026)

Iterative attempts at adding yaw control to a walking model. The lineage that
worked is collapsed into `03_yaw.yaml`.

| File | What it tried | Why archived |
|------|---------------|--------------|
| `aggressive_yaw.yaml`     | Isaac Lab G1-style strong yaw (yaw_rate_weight=5.0, 30% TIP, fast cmd switching) | Early A/B test, superseded by yaw_final once wrong-direction problem was identified |
| `balanced_yaw.yaml`       | Default-style yaw (5.0 weight, 30% TIP, capped at curriculum stage 1) | Early A/B test, same reason |
| `yaw_dominant.yaml`       | Sacrifice walking quality for turning (6.0 weight, ent_coef=0.03) | Made walking unstable; partial rollback |
| `fresh_start.yaml`        | First successful yaw fine-tune recipe (3.0 weight, 10% TIP, no scaling) | Worked, but lacked wrong_dir_penalty — got stuck on Gaussian zero-gradient |
| `yaw_gentle_range.yaml`   | Cap max_yaw_rate at 0.5 to consolidate within agent's existing range | Stepping stone — folded into yaw_ramp_up |
| `yaw_low_std.yaml`        | Reduce action noise (log_std_max=-0.5) so yaw signal isn't drowned | Helped, but not the primary fix |
| `yaw_ramp_up.yaml`        | After gentle range succeeded, widen to ±1.0 and unlock all stages | Reached 28M steps, still produced wrong-direction turns in maze |
| `yaw_nav_ready.yaml`      | Lock at stage 3, max_yaw_rate=0.6, kill TIP | Same wrong-direction problem persisted |
| `yaw_final.yaml`          | Added `yaw_wrong_dir_penalty=1.5` + `log_std_max=-0.3` — fixed wrong-direction | Worked. Recipe folded into `03_yaw.yaml`. |
| `yaw_sustained.yaml`      | Long command intervals + height penalty for sustained turns | Worked. Recipe folded into `04_omni_sustained.yaml`. |
| `yaw_polish.yaml`         | Conservative steady-state continuation from a 145M-step model | Used full transfer-style scaling — only useful for resuming a specific lineage |

## Omnidirectional exploration (4/16 — 4/17/2026)

Fixing the +x velocity drift by training on negative vx commands.

| File | What it tried | Why archived |
|------|---------------|--------------|
| `omni_sustained.yaml`     | Add `omnidirectional: true` on top of yaw_sustained          | Worked. Folded into `04_omni_sustained.yaml`. |
| `omni_direction.yaml`     | Added `vel_wrong_dir_penalty=1.5` (same pattern as yaw fix)  | Worked. Folded into `04_omni_sustained.yaml`. |

## TIP exploration (4/17 — 4/18/2026)

A/B test of turn-in-place probability. 5% was insufficient; 20% became canonical.

| File | What it tried | Why archived |
|------|---------------|--------------|
| `tip_5pct.yaml`           | 5% TIP — claimed sufficient by `yaw_finetuning_success.md` | Agent never learned (0,0,yaw) action pattern |
| `tip_20pct.yaml`          | 20% TIP — current default                                   | Worked. Recipe folded into `05_tip.yaml` (vel_error=0.124 at 9M). |
