# Training Pipeline

This is the canonical, step-by-step recipe for producing a navigation-ready
humanoid policy from scratch. **Each stage takes the output of the previous
stage as its input model.** Don't skip stages — every `04_*` config assumes
the policy already has the capability `03_*` adds.

```
Stage 1: Standing       (training_config.yaml)
   |
   v
Stage 2: Walking        (training_config.yaml + --from-standing)
   |
   v
Stage 3: Yaw control    (config/variants/03_yaw.yaml)
   |
   v
Stage 4: Omnidirectional + sustained turning   (config/variants/04_omni_sustained.yaml)
   |
   v
Stage 5: Turn-in-place  (config/variants/05_tip.yaml)
   |
   v
Maze navigation         (scripts/run_maze_nav.py)
```

Active configs live at the top level of `config/variants/`. Anything in
`config/variants/archive/` is research history — see its README for what
each archived config tried and why it was abandoned.

---

## Stage 1 — Standing

Train balance from scratch.

```bash
make train-standing ARGS="--timesteps 5000000"
```

**Inputs:** none.
**Output:** `models/best_standing_model.zip` + matching vecnorm.
**Success:** `evaluate.py --task standing` shows the agent holds 1.40m height for 10s.

---

## Stage 2 — Walking (transfer from standing)

Transfer the standing policy into a walking policy via the three-phase
warmup pipeline (VF warmup → policy ramp → permanent scaling).

```bash
make train-walking ARGS="--from-standing --model models/best_standing_model.zip --timesteps 30000000"
```

**Inputs:** `models/best_standing_model.zip`.
**Output:** `models/walking/best/model.zip` (best by `velocity_error`).
**Success:**
- `velocity_error < 0.20` sustained
- Reaches walking curriculum stage 3 (0.6 m/s commands)
- Episodes regularly run > 4000 steps

If KL diverges or `clip_fraction > 0.9`, see the PPO troubleshooting notes
in the project root `CLAUDE.md`.

---

## Stage 3 — Add yaw control

Fine-tune the walker to track commanded yaw rate.

```bash
python scripts/train_walking.py \
    --model models/walking/best/model.zip \
    --vecnorm models/walking/best/vecnorm.pkl \
    --override config/variants/03_yaw.yaml \
    --fresh-lr --timesteps 15000000
```

**Inputs:** Stage 2 best model.
**Output:** Same path (overwrites `models/walking/best/`). Back up first if
you want to keep Stage 2 weights.
**Success:**
- Agent turns the commanded direction (no wrong-direction turns in eval)
- `velocity_error < 0.15`
- Episode length > 4000 steps at curriculum stage 3

**Key parameters** (full list in the file): `yaw_rate_weight=5.0`,
`yaw_wrong_dir_penalty=1.5`, `max_yaw_rate=0.5`, curriculum locked at stage 3.

---

## Stage 4 — Omnidirectional + sustained turning

Two related fixes for navigation:
- **Omnidirectional commands** kill the +x velocity drift (Stage 3 only saw `vx ≥ 0`).
- **Long command intervals + height penalty** prevent the gait from decaying
  during multi-second yaw commands like the maze controller issues.

```bash
python scripts/train_walking.py \
    --model models/walking/best/model.zip \
    --vecnorm models/walking/best/vecnorm.pkl \
    --override config/variants/04_omni_sustained.yaml \
    --fresh-lr --timesteps 15000000
```

**Inputs:** Stage 3 model.
**Output:** Same path.
**Success:**
- Maze corridor traversal > 2000 steps
- L-maze: turns the corner without wall collision (TIP not yet trained — agent
  uses smooth pure-pursuit turns)
- `velocity_error < 0.13`

---

## Stage 5 — Turn-in-place (TIP)

Train the agent to rotate in place when commanded `vx=0, vy=0, |yaw|>0`.
Required for sharp maze corners.

```bash
python scripts/train_walking.py \
    --model models/walking/best/model.zip \
    --vecnorm models/walking/best/vecnorm.pkl \
    --override config/variants/05_tip.yaml \
    --timesteps 5000000
```

**Inputs:** Stage 4 model.
**Output:** Same path — final navigation-ready policy.
**Success:**
- Clean rotation when commanded `(0, 0, 0.5)`
- L-maze full traversal without wall collision
- `velocity_error < 0.15` (slight regression from Stage 4 is expected — TIP
  trades a little walking quality for stop-and-turn capability)

---

## Running maze navigation

Once Stage 5 is done:

```bash
python scripts/run_maze_nav.py \
    --maze-type l_maze \
    --model models/walking/best/model.zip \
    --record
```

See `src/maze/CLAUDE.md` for the navigation controller's parameters.

---

## What if a stage fails?

- **Stage 2 KL explosion:** Check `batch_size` is full-buffer (24576), `n_epochs=3`,
  `policy_max_scale=0.5`. See project root `CLAUDE.md` for the full PPO checklist.
- **Stage 3 wrong-direction turns:** Confirm `yaw_wrong_dir_penalty=1.5` is set.
  Without it, Gaussian yaw reward has zero gradient beyond err=0.5 rad/s.
- **Stage 4 height decay:** Confirm `reward_height_weight=1.0` and
  `command_switch_interval=[2, 15]`.
- **Stage 5 vel_error spike:** Expected — TIP introduces (0,0,yaw) command pattern.
  If spike is large (>0.3), drop `turn_in_place_prob` to 0.10 and retrain.

Don't tweak parameters mid-stage. If a stage fails, restart from the previous
stage's checkpoint with the override that addresses the specific failure.
