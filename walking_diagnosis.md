# Walking Training Diagnosis: Root Cause Analysis

**Status:** 0% success rate after 5M timesteps | Mean velocity error: 1.34 m/s | Mean height: 0.893m (target 1.40m)

---

## Executive Summary

The walking policy is trapped in a **standing-still local optimum**. Five independent analyses (reward function, curriculum/environment, command generator, transfer learning, PPO/config) converge on the same conclusion: **standing still is more profitable than walking** under the current reward structure, termination conditions, and observation pipeline. The agent has rationally learned to avoid movement because movement triggers height penalties and termination, while standing accrues safe positive reward indefinitely.

The root cause is not a single bug but a **compounding chain of design conflicts** across four systems: reward shaping, termination logic, observation normalization, and transfer initialization.

---

## Critical Issues (Fix First)

### 1. Reward Structure Incentivizes Standing Over Walking

**Files:** `walking_env.py:360-720`, `rewards.py`, `config/training_config.yaml`

The reward function has a fundamental hierarchy conflict:

| Component | Standing Still (0.3 m/s error) | Shuffling (0.09 m/s error) | Walking (0.0 error) |
|-----------|-------------------------------|---------------------------|---------------------|
| Velocity tracking (w=25.0, bw=5.0) | +15.96/step | +24.6/step | +25.0/step |
| Standing penalty | -15.0/step (if speed < 0.05) | 0.0 | 0.0 |
| Height penalty (if COM drops during gait) | 0.0 | -4.0 to -30.0/step | -4.0 to -30.0/step |
| Termination risk | None | Low | High |
| **Net reward** | **+0.96/step (safe)** | **~+20.6 minus fall risk** | **~+25 minus fall risk** |

**Key findings:**
- The modular `RewardCalculator` in `rewards.py` is instantiated but **mostly bypassed**. Only `vel_error` and `jerk_penalty` are used from it. The actual reward is computed by 350+ lines of ad-hoc logic in `walking_env.py:_compute_task_reward()`.
- Height maintenance penalty (`-50.0 * height_velocity`) fires during natural gait oscillation. Walking naturally causes COM to drop 0.1-0.15m during swing phase, triggering penalties that **exceed the maximum velocity tracking reward**.
- The `standing_penalty` of -15.0 only fires when `actual_speed < 0.05`. The agent can shuffle at 0.06 m/s to avoid it entirely while still not really walking.
- The `consistency_bonus` (w=8.0) and `sustained_bonus` (up to +25.0) require good performance to activate, making them **unavailable during the critical early learning phase** when the agent needs the strongest learning signal.

**Recommendations:**
- Increase standing penalty to -25.0 and raise the speed threshold to 0.15 m/s
- Reduce height maintenance penalty weight from -50.0 to -10.0 (natural gait oscillation should not be penalized)
- Decouple height reward from velocity tracking quality in early stages
- Either use the modular `RewardCalculator` consistently or remove it to avoid maintenance confusion

---

### 2. Termination Conditions Kill Walking Attempts

**Files:** `walking_env.py:570-609`, `walking_curriculum.py`

The termination system punishes walking attempts more than standing:

| Parameter | Value | Problem |
|-----------|-------|---------|
| Height threshold | 0.70m | Natural walking crouches reach ~0.75m during leg extension |
| Early protection | 50 steps (0.5s) | One gait cycle takes 0.7-1.0s; protection expires mid-stride |
| Recovery window | 20 steps (0.2s) | Shorter than one stride; robot dies mid-step |
| Termination penalty | -50.0 | Catastrophic compared to +25 max tracking reward |

**Result:** The agent discovers that any walking attempt risks a -50 penalty, while standing safely accumulates +16/step. After a few hundred episodes, it converges on standing.

**Recommendations:**
- Lower height threshold to 0.55m (allow natural gait oscillation)
- Extend early protection to 200 steps (2 seconds, ~2 gait cycles)
- Extend recovery window to 80-100 steps (allow one full stride to recover)
- Reduce termination penalty to -20.0 (still negative but not catastrophic relative to tracking reward)

---

### 3. VecNormalize Crushes Command Visibility (Variance Collapse)

**Files:** `transfer_utils.py:191-198`, `walking_curriculum.py:205-213`, `walking_env.py:821-831`

This is the **smoking gun** for why the policy ignores velocity commands entirely:

**Stage 0 uses fixed commands** with minimal variance:
```
vx: 0.27-0.31 m/s (nearly constant)
vy: -0.08 to 0.08 m/s (near zero)
yaw_rate: always 0.0
```

During warmup (10k steps), VecNormalize collects statistics on these near-constant values:
```
After warmup: mean ~ [0.30, 0.00, 0.00], var ~ [0.001, 0.001, 0.000]
```

Then during training, VecNormalize applies: `(obs - mean) / sqrt(var + eps)`

For the command dimensions: `(0.30 - 0.30) / sqrt(0.001) = 0.0`

**The command signal becomes zero.** The policy literally cannot see what velocity is being commanded.

Additionally, the observation pipeline has a **scale mismatch**:
- Body features after `tanh(0.1 * x)`: magnitude ~0.1-0.3
- Command block after `clip(x/max_speed, -1, 1)`: magnitude ~0.1 for Stage 0
- After VecNormalize: command dims further compressed relative to 1484 body dims
- 9 command features vs 1484 body features = commands are 0.6% of the observation space

**Recommendations:**
- Do NOT use fixed commands during warmup; use the full command generator with diverse sampling
- Alternatively, skip VecNormalize for command dimensions entirely (they're already normalized to [-1, 1])
- Add command variance monitoring to training logs
- Consider scaling command block by 3-5x before concatenation to increase signal strength

---

### 4. Transfer Learning: Policy Ignores New Command Dimensions

**Files:** `transfer_utils.py:227-427`

When the standing policy (1484 dims) is extended to walking (1493 dims), the 9 new command dimensions get Xavier-initialized weights with std ~0.032. The 1484 standing weights are fully trained and coordinated.

**Impact on policy network first layer (512 units):**
- 1484 standing weights: precisely tuned, produce strong activations
- 9 command weights: random noise at 0.032 std, produce negligible activations
- Commands contribute ~0.2% of total activation magnitude

The "velocity" initialization strategy copies weights from velocity observation dims (0-2) to command dims. This is semantically wrong: the standing policy learned to **observe** its velocity, not **follow** a commanded velocity. The relationship is fundamentally different.

**Value function is also not re-initialized.** The critic still predicts standing-task returns, causing value estimation errors that destabilize PPO updates early in walking training.

**Recommendations:**
- Initialize command weights with 5-10x larger scale (std ~0.15-0.30)
- Re-initialize the value function (critic) after transfer with orthogonal init
- Add a gradient flow diagnostic: perturb command inputs and verify non-zero action change
- Consider freezing body weights for the first 100k steps to force learning through command dims

---

## High-Priority Issues

### 5. Action Smoothing Too Aggressive for Walking

**File:** `walking_env.py:730-732`

Action smoothing uses `tau=0.2`, meaning:
```
new_action = 0.8 * prev_action + 0.2 * policy_output
```

The policy's current decision is only **20% influential**. For balance recovery during walking (which requires fast corrective actions), this creates a ~200ms effective delay. The humanoid cannot react fast enough to prevent falls.

**Recommendation:** Reduce tau to 0.05-0.10 for walking stages (keep 0.2 for standing where stability matters more).

---

### 6. Success Definition Creates a Dead Zone

**File:** `walking_curriculum.py:325-335`

Stage 0 success requires `is_moving = actual_speed >= commanded_speed * 0.30` (i.e., 0.09 m/s for a 0.3 m/s command). But the standing-still policy achieves ~0.06-0.08 m/s from natural oscillation, just barely failing the check.

Combined with 0% success rate, the curriculum never advances. The agent is stuck:
- Too fast to be "standing" (avoids standing penalty at 0.06+ m/s)
- Too slow to pass "is_moving" (needs 0.09 m/s)
- No incentive to bridge the 0.03 m/s gap (reward difference is negligible)

**Recommendation:** Lower the is_moving threshold to 0.05 m/s OR increase the standing penalty threshold to 0.15 m/s to create a clearer gap.

---

### 7. VecNormalize Clipping Too Permissive

**File:** `config/training_config.yaml`

```yaml
vecnormalize_clip_obs: 50.0   # Effectively no clipping (50 std devs)
vecnormalize_clip_reward: 50.0
```

Standard practice is clip_obs=5-10, clip_reward=1-10. At 50.0, normalization is ineffective and extreme outliers (reward spikes of 10k-20k) can destabilize the value function.

**Recommendation:** Set `clip_obs: 10.0`, `clip_reward: 10.0`.

---

### 8. Action Std Anomaly in Training Logs

Training logs show action_std increasing from 7e4 to 2.88e5, which is physically impossible for bounded action spaces. This indicates either:
- A logging/scaling bug (most likely: reporting unnormalized variance instead of std)
- Unbounded log_std in the policy head
- Missing action space clipping

If the policy output variance is truly unbounded, inference actions will be erratic regardless of learned mean behavior.

**Recommendation:** Verify SB3 policy log_std bounds. Add explicit log_std clamping if needed.

---

## Medium-Priority Issues

### 9. Entropy Decay Schedule

Entropy decays from 0.10 to 0.015 over training. By 5M steps it's at ~0.026 — low enough that the policy is too deterministic to discover walking from a standing-still optimum.

**Recommendation:** Slow the decay or set `final_ent_coef: 0.005`. Consider resetting entropy to 0.08 when stuck at 0% success for >500k steps.

### 10. Batch Size / Gradient Stability

Batch size 256 with rollout 24,576 (2048 steps x 12 envs) means 1% minibatch ratio. This creates high-variance gradient estimates for a 17-dim action space.

**Recommendation:** Increase to `batch_size: 512` or `n_epochs: 20`.

### 11. Stabilization Window Never Triggers

**File:** `walking_curriculum.py:267-277`

Velocity error is measured using the stabilized portion of the episode (after `stabilization_steps`). But for short episodes (<50 steps), the stabilization window never activates, and noisy early-phase errors are used instead.

**Recommendation:** Set a minimum stabilization window of 20 steps regardless of episode length.

---

## Root Cause Chain (How All Issues Compound)

```
Transfer from Standing
  |
  v
Policy ignores 9 command dims (tiny weights, variance-collapsed VecNormalize)
  |
  v
Policy outputs standing-still actions regardless of commanded velocity
  |
  v
Walking attempts trigger height oscillation from gait mechanics
  |
  v
Height penalties (-50 * height_velocity) exceed velocity tracking reward (+25 max)
  |
  v
Termination at 0.70m fires during natural gait crouches
  |
  v
Agent learns: walking = -50 penalty, standing = +16/step safe reward
  |
  v
Policy converges to standing-still optimum (locally rational)
  |
  v
Success rate stays at 0% (agent shuffles at 0.06-0.08 m/s, needs 0.09)
  |
  v
Curriculum never advances from Stage 0
  |
  v
Action smoothing (tau=0.2) prevents corrective actions during rare walking attempts
  |
  v
Low entropy (0.026) prevents escape from local optimum
  |
  v
STUCK
```

---

## Recommended Fix Priority

| Priority | Fix | Files to Change |
|----------|-----|-----------------|
| 1 | Reduce height maintenance penalty from -50 to -10 | `walking_env.py` |
| 2 | Lower termination height threshold to 0.55m, extend recovery to 80 steps | `walking_env.py`, `training_config.yaml` |
| 3 | Fix VecNormalize variance collapse: use diverse commands during warmup OR skip normalization for command dims | `transfer_utils.py`, `walking_curriculum.py` |
| 4 | Increase command weight initialization scale (5-10x) | `transfer_utils.py` |
| 5 | Re-initialize value function after transfer | `transfer_utils.py` |
| 6 | Reduce action smoothing tau to 0.05 for walking | `training_config.yaml` |
| 7 | Increase standing penalty threshold to 0.15 m/s | `walking_env.py` |
| 8 | Fix VecNormalize clipping (50 -> 10) | `training_config.yaml` |
| 9 | Slow entropy decay (final 0.015 -> 0.005) | `training_config.yaml` |
| 10 | Investigate action_std anomaly in logs | `train_walking.py` |

---

## Files Referenced

| File | Issues Found |
|------|-------------|
| `src/environments/walking_env.py` | Reward bypass (#1), height penalties (#1), termination (#2), action smoothing (#5), standing penalty (#6) |
| `src/environments/walking_curriculum.py` | Fixed command variance collapse (#3), success definition (#6), stabilization window (#11) |
| `src/training/transfer_utils.py` | Command weight init (#4), value function (#4), VecNormalize extension (#3) |
| `src/core/rewards.py` | Modular calculator bypassed by walking_env (#1) |
| `config/training_config.yaml` | VecNormalize clipping (#7), action smoothing (#5), entropy schedule (#9) |
| `scripts/train_walking.py` | Action std logging (#8) |
