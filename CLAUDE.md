# Humanoid Navigation

Reinforcement learning for humanoid locomotion and navigation using PPO (Stable Baselines3) and MuJoCo physics simulation. Three-stage pipeline: standing balance → walking → maze navigation.

## How we work: every change goes through a pull request

`main` only changes by merging a PR. Nobody commits or pushes to `main` directly.

1. **Start from fresh `main`, on a new branch named for the one thing it does:**
   `git switch main && git pull && git switch -c posture-deadband`
2. **Commit as you go** (a commit per logical step, message = what changed and why) and
   **push at least at the end of every work session**: `git push -u origin <branch>`.
   Work that exists only on this laptop is not backed up and nobody can review it.
3. **Open a PR into `main`** as soon as the branch has something to show
   (`gh pr create --base main`). Keep pushing to the same branch; the PR updates itself.
4. **CI must be green** (`.github/workflows/ci.yml`: `ruff check .` + `python -m pytest`).
   Run both locally before pushing.
5. **Merge** (merge commit, not squash) once CI passes and the user agrees the PR is done,
   then delete the branch. The next piece of work starts again at step 1.

One topic per PR: a reward change, a deploy fix and a new eval script are three PRs.
Hardware runs, evals and training results belong in the PR whose code produced them.

**For agents: the user is learning GitHub, so keep them on this workflow.** Before acting,
and briefly (one line, then help), point it out when a request or the repo state deviates
from it, and offer the PR way:
- asked to commit or push to `main`, or about to start work while `main` is checked out
- uncommitted or unpushed work left from an earlier session (check `git status` first)
- new work starting on a branch that belongs to a different topic, or a PR growing a second topic
- merging with CI red or not yet run, or a branch left behind after its PR merged

Do not commit, push, open or merge PRs unless asked, but when asked to "save" or
"ship" work, the answer is a branch + PR, not a commit on `main`.

## Architecture

```
Standing (5M steps)  →  Transfer Learning  →  Walking (30M steps)  →  Maze Navigation
  StandingCurriculumEnv      VecNormalizeExtender      WalkingCurriculumEnv      NavigationController
  6-stage height curriculum  PolicyTransfer             3-stage speed curriculum  Pure pursuit + A*
  0.80m → 1.40m target      1484 → 1495 obs dims      0.15 → 0.80 m/s          Frozen walking policy
```

## Directory Map

| Directory | Purpose | Details |
|-----------|---------|---------|
| `src/environments/` | Gym wrappers + curriculum | [CLAUDE.md](src/environments/CLAUDE.md) |
| `src/core/` | Reward calculator + command generator | [CLAUDE.md](src/core/CLAUDE.md) |
| `src/training/` | Transfer learning, model management, callbacks | [CLAUDE.md](src/training/CLAUDE.md) |
| `src/maze/` | Procedural maze generation + navigation | [CLAUDE.md](src/maze/CLAUDE.md) |
| `src/agents/` | High-level agent wrappers + diagnostics | [CLAUDE.md](src/agents/CLAUDE.md) |
| `src/robustness/` | Scenarios (starts, pushes, sensor/plant faults) for training mixes + the eval suite and its score | [CLAUDE.md](src/robustness/CLAUDE.md) |
| `config/experiments/` | Standing experiments: a base config + only the keys that change | [config/CLAUDE.md](config/CLAUDE.md) |
| `runs/` | One directory per training run (gitignored): config, checkpoints, best/, final/, logs | — |
| `tests/` | `python -m pytest` (robustness mechanics, training pipeline, maze) | — |
| `src/utils/` | Visualization + WandB utilities | [CLAUDE.md](src/utils/CLAUDE.md) |
| `scripts/` | Training, evaluation, debug entry points | [CLAUDE.md](scripts/CLAUDE.md) |
| `config/` | Single YAML config for all tasks | [CLAUDE.md](config/CLAUDE.md) |
| `models/` | Trained weights + VecNormalize stats | — |

Source code details: [src/CLAUDE.md](src/CLAUDE.md)

## Key Numbers

| Constant | Value | Notes |
|----------|-------|-------|
| Target height | 1.40m | Both standing and walking |
| Base obs dims | 365 | Humanoid-v5 |
| COM features | +6 dims | Center of mass position + velocity |
| Command dims | +11 dims | Walking only (cmd, actual, error for vx/vy/yaw) |
| History frames | 4 | Temporal stacking |
| Total obs (standing) | 1484 | (365 + 6) × 4 |
| Total obs (walking) | 1495 | 1484 + 11 command block |
| Action dims | 17 | Joint torques |
| Standing curriculum | 6 stages | 0.80m → 1.40m |
| Walking curriculum | 3 stages | 0.15 → 0.80 m/s |

## Dev Workflow

```bash
# Install
make setup        # or: pip install -r requirements.txt

# Lint
make lint         # ruff check

# Train standing (fresh)
make train-standing ARGS="--timesteps 5000000"

# Train walking (from standing model)
make train-walking ARGS="--from-standing --model models/best_standing_model.zip --timesteps 30000000"

# Evaluate + record video
make evaluate ARGS="--task walking --model models/walking/final/final_walking_model.zip --record"

# Maze navigation
python scripts/run_maze_nav.py --maze-type corridor --model models/walking/best/model.zip --record

# Real-robot standing: experiment -> robustness score -> compare (see scripts/CLAUDE.md)
python scripts/train_standing.py --config config/experiments/stand_pushes.yaml --wandb online
python scripts/eval_standing.py --run runs/<name>
python scripts/compare_models.py

# Tests
python -m pytest
```

## Common Pitfalls

- **VecNormalize mismatch**: Always load the matching `.pkl` file with the model. Standing and walking have different observation dimensions.
- **log_std explosion**: Without clamping, `log_std` can reach ~9.0 (std ~8000), causing action explosion. Training scripts clamp to `[-2.0, 0.5]` (standing) or `[-2.0, 1.0]` (walking).
- **Command stat collapse**: VecNormalize can shrink command variance to ~0.001, making commands invisible to the policy. `CommandStatsProtectorCallback` re-pins command block to identity stats (mean=0, var=1) every 10k steps.
- **Standing exploit**: Walking Stage 0 requires `actual_speed ≥ 0.5 × commanded_speed` to prevent the agent from earning rewards by standing still.
- **Reward scale**: Walking rewards were scaled down 10× from original values. High rewards cause PPO KL divergence explosion.
- **Value function re-init**: During transfer, the standing value function must be re-initialized — standing value estimates corrupt walking gradients.
- **Judge standing models by `eval/score`, against the no-feedback poses**: training reward is not comparable across runs. On 10/7 the deployed 195M policy scored 70.5, indistinguishable from commanding the straight pose with no feedback (70.8), which is also what an UNTRAINED net does.
- **Phantom self-collision** (fixed 10/16, suite -> standing-v2): the real-robot MJCF excluded the LEFT hip-yaw link vs the pelvis but not the right, so sim R_hip_roll sat on a stop the robot does not have. Every policy since 8/14 braced on it and fell sideways on hardware. If a policy works in sim and not on the robot, compare each joint's command vs position in sim first.
- **Entropy schedule on resume** (fixed 10/7): it divided cumulative steps by this call's steps, so every fine-tune ran ent_coef ~ -0.024. LR and clip follow SB3's progress, which on a resume starts near 0.1, so a fine-tune runs at close to the FINAL lr/clip.
- **Yaw bandwidth too soft**: `reward_yaw_bandwidth` controls yaw tracking sharpness. At 4.0 (old hardcoded value), the agent retained 91% of yaw reward while producing zero turning. Default is now 10.0 (configurable). Curriculum advancement also gates on yaw error.
