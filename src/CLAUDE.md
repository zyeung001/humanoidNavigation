# src/ — Source Library

Package version: 0.2.0. Import from submodules directly (e.g., `from src.environments import make_walking_env`).

## Module Dependencies

```
environments/  ←  core/rewards.py, core/command_generator.py
training/      ←  environments/, core/
agents/        ←  environments/, training/
scripts/       ←  all of the above
```

## Key Patterns

- **Gym wrappers**: Environments subclass `gym.Wrapper` around `Humanoid-v5`. Curriculum envs subclass the base env.
- **Dataclass configs**: `RewardWeights`, `RewardMetrics` in `core/rewards.py` for structured reward parameters.
- **Callback-based monitoring**: Training uses SB3 callbacks for entropy scheduling, log_std clamping, WandB logging, and VecNormalize protection.
- **VecNormalize everywhere**: Both observations and rewards are normalized. Stats must be saved/loaded with the model.

## Data Flow

```
Humanoid-v5 raw obs (350 dims)
  → StandingEnv/WalkingEnv processes to 365 dims
  → Append COM features (+6)
  → [Walking only] Append command block (+11)
  → Stack 4 frames → 1484 or 1495 dims
  → VecNormalize → normalized obs to PPO policy
  → PPO outputs 17-dim action
  → Action smoothing (EMA) → env.step()
  → Reward computed by env (standing) or RewardCalculator (walking)
  → VecNormalize normalizes reward
```

## Subdirectory Docs

- [environments/CLAUDE.md](environments/CLAUDE.md) — Observation/action spaces, curriculum, termination
- [core/CLAUDE.md](core/CLAUDE.md) — Reward math, command generator
- [training/CLAUDE.md](training/CLAUDE.md) — Transfer learning, model management, safety callbacks
