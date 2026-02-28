# scripts/ — Entry Points

## Training Scripts

### `train_standing.py`

Train the standing (balance) controller from scratch.

```bash
# Fresh training
python scripts/train_standing.py --timesteps 5000000

# Resume from checkpoint
python scripts/train_standing.py --model models/best_standing_model.zip \
    --vecnorm models/vecnorm.pkl --timesteps 5000000
```

Key args: `--timesteps`, `--model` (resume from), `--vecnorm` (VecNormalize stats).

Includes callbacks for entropy scheduling, log_std clamping, and VecNormalize saving. Uses `StandingCurriculumEnv` with 6 stages.

### `train_walking.py`

Train the walking controller, typically via transfer from a standing model.

```bash
# From standing (recommended)
python scripts/train_walking.py --from-standing \
    --model models/best_standing_model.zip \
    --timesteps 30000000

# Resume walking training
python scripts/train_walking.py \
    --model models/walking/final/final_walking_model.zip \
    --vecnorm models/walking/final/vecnorm_walking.pkl \
    --timesteps 30000000
```

Key args: `--from-standing` (enable transfer), `--model`, `--vecnorm`, `--timesteps`.

Adds `CommandStatsProtectorCallback` (re-pins command stats every 10k steps) and `WalkingMetricsCallback` (curriculum progress every 50 episodes). Uses `WalkingCurriculumEnv` with 3 stages.

## Evaluation

### `evaluate.py`

Unified evaluation and video recording for both tasks.

```bash
# Evaluate standing
python scripts/evaluate.py --task standing \
    --model models/best_standing_model.zip

# Evaluate walking + record video
python scripts/evaluate.py --task walking \
    --model models/walking/final/final_walking_model.zip --record

# Walking with specific velocity command
python scripts/evaluate.py --task walking \
    --model models/walking/final/final_walking_model.zip \
    --vx 1.0 --vy 0.0 --record
```

Videos saved to `data/videos/{task}_eval.mp4`.

### `record_video.py`

Legacy video recording script. Prefer `evaluate.py --record` instead.

## Debug Scripts (`scripts/debug/`)

| Script | Purpose |
|--------|---------|
| `test_standing.py` | Standalone standing model evaluation |
| `test_walking.py` | Standalone walking model evaluation |
| `analyze_rewards.py` | Reward component analysis |
| `diagnose_transfer.py` | Debug standing → walking transfer |
| `standing_plot.py` | Plot standing training metrics |
| `walking_plot.py` | Plot walking training metrics |
