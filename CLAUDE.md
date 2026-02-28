# Humanoid Navigation

Reinforcement learning for humanoid locomotion using PPO (Stable Baselines3) and MuJoCo physics simulation. Two-stage pipeline: train standing balance first, then transfer to command-conditioned walking.

## Architecture

```
Standing (5M steps)  →  Transfer Learning  →  Walking (30M steps)
  StandingCurriculumEnv      VecNormalizeExtender      WalkingCurriculumEnv
  6-stage height curriculum  PolicyTransfer             3-stage speed curriculum
  0.80m → 1.40m target      1484 → 1493 obs dims      0.15 → 0.80 m/s
```

## Directory Map

| Directory | Purpose | Details |
|-----------|---------|---------|
| `src/environments/` | Gym wrappers + curriculum | [CLAUDE.md](src/environments/CLAUDE.md) |
| `src/core/` | Reward calculator + command generator | [CLAUDE.md](src/core/CLAUDE.md) |
| `src/training/` | Transfer learning, model management, callbacks | [CLAUDE.md](src/training/CLAUDE.md) |
| `src/agents/` | High-level agent wrappers + diagnostics | — |
| `src/utils/` | Visualization + WandB utilities | — |
| `scripts/` | Training, evaluation, debug entry points | [CLAUDE.md](scripts/CLAUDE.md) |
| `config/` | Single YAML config for both tasks | [CLAUDE.md](config/CLAUDE.md) |
| `models/` | Trained weights + VecNormalize stats | — |

Source code details: [src/CLAUDE.md](src/CLAUDE.md)

## Key Numbers

| Constant | Value | Notes |
|----------|-------|-------|
| Target height | 1.40m | Both standing and walking |
| Base obs dims | 365 | Humanoid-v5 |
| COM features | +6 dims | Center of mass position + velocity |
| Command dims | +9 dims | Walking only (cmd, actual, error) |
| History frames | 4 | Temporal stacking |
| Total obs (standing) | 1484 | (365 + 6) × 4 |
| Total obs (walking) | 1493 | 1484 + 9 command block |
| Action dims | 17 | Joint torques |
| Standing curriculum | 6 stages | 0.80m → 1.40m |
| Walking curriculum | 3 stages | 0.15 → 0.80 m/s |

## Dev Workflow

```bash
# Install
pip install -r requirements.txt

# Train standing (fresh)
python scripts/train_standing.py --timesteps 5000000

# Train walking (from standing model)
python scripts/train_walking.py --from-standing \
    --model models/best_standing_model.zip \
    --timesteps 30000000

# Evaluate + record video
python scripts/evaluate.py --task walking \
    --model models/walking/final/final_walking_model.zip --record
```

## Common Pitfalls

- **VecNormalize mismatch**: Always load the matching `.pkl` file with the model. Standing and walking have different observation dimensions.
- **log_std explosion**: Without clamping, `log_std` can reach ~9.0 (std ~8000), causing action explosion. Training scripts clamp to `[-2.0, 0.5]` (standing) or `[-2.0, 1.0]` (walking).
- **Command stat collapse**: VecNormalize can shrink command variance to ~0.001, making commands invisible to the policy. `CommandStatsProtectorCallback` re-pins command block to identity stats (mean=0, var=1) every 10k steps.
- **Standing exploit**: Walking Stage 0 requires `actual_speed ≥ 0.5 × commanded_speed` to prevent the agent from earning rewards by standing still.
- **Reward scale**: Walking rewards were scaled down 10× from original values. High rewards cause PPO KL divergence explosion.
- **Value function re-init**: During transfer, the standing value function must be re-initialized — standing value estimates corrupt walking gradients.
