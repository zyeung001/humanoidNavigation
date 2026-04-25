# Navigation Rebuild Plan

## Why

Current architecture decomposes navigation into `walking_policy + A* + pure_pursuit`. The walking policy was trained on random world-frame velocity commands in an open arena. It never sees walls, goals, or paths. Every L-maze failure traces back to this gap. Six nav-controller variants and four heading_align reward variants have failed to close it.

The fix is to train end-to-end navigation: goal in, torques out. One unified policy. No mode-switching controller.

## Target system

- **Obs additions**: 6 dims = body-frame `(dx, dy)` to next 3 waypoints from an A* path computed at episode reset.
- **Action**: 17-dim torques (unchanged).
- **Reward**: progress along path, waypoint-reached bonus, time penalty, collision penalty. No velocity tracking.
- **Env**: procedurally generated mazes (`src/maze/maze_generator.py` already exists), random goal each episode.
- **Eval**: held-out maze types (corridor, L, U, dfs_3x3 — already in `src/maze/maze_maps.py`).

## Reward design (with anti-hacking)

| Component | Purpose | Anti-exploit |
|-----------|---------|--------------|
| Progress along path arc-length | Forward motion toward goal | Arc-length, not Euclidean — closes "ghost waypoint through wall" |
| Waypoint reached bonus | Sparse reinforcement of correct subgoal | Monotonic advance only; cannot retrigger |
| Time penalty (small) | Faster is better | Bounded so reaching goal is always net positive vs giving up |
| Collision penalty + termination | Walls are not speed bumps | Episode ends on collision — same severity as a fall |
| Fall termination penalty | Don't sacrifice posture for progress | Strictly worse than continuing |

Forbidden: survival reward (causes freezing local optimum), Euclidean progress (ghost waypoints), mid-episode A* re-plan (encourages straying).

## Phases

Each phase ships an end-to-end working slice. Pass criteria are defined before the phase starts.

### Phase 0 — De-risk env (1-2 days, no training)
- Build the new env: maze, A* path, body-frame waypoint obs, new reward.
- Drive it with a hand-coded controller (head straight toward next waypoint).
- **Pass:** reward curve is sane, no obvious exploit, episode terminates correctly on goal/collision/fall.

### Phase 1 — Open arena, single goal (3-5 days compute)
- No walls. Random goal point ~5m away. Warm-start from existing walking model (`models/walking/best/`).
- **Pass:** >80% goal reach within timeout. Confirms the new reward + obs can drive locomotion at all.

### Phase 2 — Fixed simple mazes (1-2 weeks compute)
- Corridor + L-maze + U-maze, fixed seeds.
- **Pass:** >70% goal reach on each. Reward-hacking exploits surface here — actively look for them.

### Phase 3 — Procedural mazes (2-4 weeks compute)
- Random maze each episode (`generate_maze_dfs`, `generate_maze_prims`).
- **Pass:** >60% goal reach on held-out maze types.

### Phase 4 — Robustness (open-ended)
- Domain randomization, harder mazes, varying start orientations.

## Parallel tracks

- **Eval harness**: consistent suite of held-out mazes, run nightly on latest checkpoint.
- **Reward-hack watchlist**: log per-episode `progress / collisions / time / final_distance`; alert on outliers.
- **Use existing JSONL metrics logger** (`src/training/metrics_logger.py`).

## Reuse

- Standing model and walking model — warm-start, don't retrain locomotion from scratch.
- `src/maze/` infrastructure (generator, A*, MJCF) — already works.
- `src/training/model_manager.py`, JSONL logger, callbacks — unchanged.

## Out of scope (don't do)

- Velocity-command interface (replaced by waypoint obs).
- Nav controller mode-switching (TIP vs walk-and-steer vs brake) — the policy handles all modes.
- New heading_align reward variants.
- New nav controller variants (see `nav_controller_cant_fix_corners.md`).

## Principles

1. Each phase ships something working end-to-end, not a layer.
2. Cost-of-failure scales with phase number — front-load cheap discovery.
3. Define pass criteria before starting a phase.
4. If Phase N fails, fix it in Phase N — don't paper over it and continue.
5. Every reward must have its reverse path closed: if "go to goal" pays X, "give up" must pay strictly less than X under all conditions.
