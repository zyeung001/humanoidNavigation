# Why Your Humanoid Won't Walk: A Stage 0 Diagnosis

Your agent has learned something remarkable: **how to pass the test without doing the work**. It stands perfectly still, survives long enough, and the curriculum says "good job, you're walking." This document explains why, and how to fix it.

---

## Issue 1: The Standing-Still Exploit

### High Level

The curriculum's success criteria accidentally reward survival over movement. When you set a velocity tolerance of 0.9 m/s for a 0.3 m/s command, you've created a world where standing still (velocity error = 0.3) is well within the "acceptable" range. The agent discovers this loophole immediately—why risk falling while walking when standing still passes the test?

This is a classic reinforcement learning failure mode: the agent optimizes exactly what you measure, not what you intended.

### Simply Explained

Imagine telling a student "you pass if your answer is within 90 points of correct" on a test where the right answer is 30. Getting zero is only 30 points off—that's a passing grade! Your agent figured out that doing nothing scores better than trying and failing.

**The fix:** Tighten the tolerance. If you command 0.3 m/s, require the agent to achieve at least 0.1 m/s to pass. Standing still should fail.

> **✅ VERIFIED** — `walking_curriculum.py:53` shows `velocity_tolerances = [0.9, 0.75, 0.6, ...]`. Stage 0 tolerance is exactly 0.9 m/s as claimed. However, note that the code at lines 279-291 does include an `attempted_movement` check requiring `actual_speed > 0.08` for Stage 0-1, which partially mitigates this exploit.

---

## Issue 2: The Invisible Command

### High Level

Your observation space is 1,496 dimensions. The velocity command—the entire point of the task—lives in the last 3 dimensions. After your feature normalization applies `tanh(0.1 * x)`, a 0.3 m/s command becomes 0.03 in magnitude. This is below the noise floor of a randomly initialized neural network.

The agent literally cannot see what you're asking it to do. It's like handing someone a 500-page book and telling them the instructions are in the last three letters, written in 2-point font.

### Simply Explained

You're whispering the instructions in a loud room. The command signal is so small and so buried that the network treats it as noise. The agent has no idea it's supposed to walk forward versus backward versus sideways—all it sees is static.

**The fix:** Make the command loud and clear. Put the commanded velocity and actual velocity side-by-side at the end of the observation, normalized to a sensible range like [-1, 1]. Let the network see "target: 0.3, actual: 0.0" as an obvious error signal.

> **✅ VERIFIED** — `walking_env.py:708-709` confirms feature normalization: `if self.feature_norm: feat_vec = np.tanh(feat_vec * 0.1)`. This applies to ALL features including commands. With tanh(0.1 × 0.3) ≈ 0.03, commands are indeed nearly invisible. Observation dimension is 1496 (`walking_env.py:186`) with command at positions 371-374 in each of 4 history frames.

---

## Issue 3: Death Before Learning

### High Level

Your termination conditions create a catch-22. The agent needs to attempt walking to learn, but attempting walking causes natural height oscillations (crouching during steps), which trigger termination. The "early protection" window of 50 steps expires right when the agent would start discovering coordinated movement. The 20-step recovery window is shorter than a single gait cycle.

You've built a driving school where students fail their license test if they ever touch the brake pedal.

### Simply Explained

Walking involves bobbing up and down—that's physics. Your termination says "if you bob down too far for too long, you're dead." The "too long" threshold is 0.2 seconds. A natural walking step takes 0.7 seconds. The agent dies mid-stride, every time, and learns that attempting to walk is dangerous.

**The fix:** Be patient. Extend the protection window to 150+ steps. Extend the recovery window to at least one full gait cycle (60-80 steps). Lower the height threshold to catch actual falls (0.60m), not natural crouches (0.85m).

> **✅ VERIFIED** — `training_config.yaml:229-232` confirms exact values:
> - `termination_height_threshold: 0.70` (not 0.85 as suggested—already lowered)
> - `early_termination_protection: 50` steps
> - `termination_height_recovery_window: 20` steps
>
> The 20-step window at ~100Hz simulation = 0.2 seconds, matching the claim. The recovery window is indeed shorter than a typical gait cycle.

---

## Issue 4: Too Many Tasks at Once

### High Level

Stage 0 asks the agent to simultaneously learn: forward walking, diagonal walking, lateral movement, backward walking, and random directions—all while speed commands vary from 0.1 to 0.3 m/s within a single episode. This is curriculum anti-pattern. The agent sees a different task every few seconds and cannot form stable associations between actions and outcomes.

Curriculum learning works by mastering simple tasks before combining them. You're asking for the combination before any component is learned.

### Simply Explained

You're teaching someone to juggle by throwing five balls at them immediately. They need to learn to catch one ball first. Your agent needs to learn "move forward" as a single, stable concept before you add "move diagonally" or "vary your speed."

**The fix:** Stage 0 should be boring. Fixed speed (0.3 m/s), fixed direction (forward), no surprises. Let the agent master one thing. Introduce variety in stage 1.

> **✅ VERIFIED** — `walking_curriculum.py:63` shows `direction_diversity = cfg.get('direction_diversity', True)` defaulting to enabled. Lines 172-187 confirm 5 direction types: forward (50%), diagonal (25%), lateral (15%), backward (2%), random (8%). Speed sampling at line 175: `np.random.uniform(0.1, self.max_commanded_speed)` with max=0.3. The agent faces high task variance from step one.

---

## Issue 5: Penalties That Punish Progress

### High Level

When the agent's height drops during a walking attempt, it accumulates a penalty that scales linearly: -3 per step becomes -60 after 20 steps. Meanwhile, the maximum velocity tracking reward is +25. The penalty for *trying to walk* exceeds the reward for *walking successfully*. The agent learns that the safest strategy is motionlessness.

Your reward function punishes exploration more than it rewards success.

### Simply Explained

Every time the agent bends its knees to take a step, it gets fined. The fine grows every moment it stays bent. Eventually the fine is larger than any possible paycheck from walking. Rational response: never bend your knees.

**The fix:** Cap the penalty at a small fixed value (-0.5 per step, not -3 × counter). The penalty should discourage prolonged falling, not punish natural gait mechanics.

> **✅ VERIFIED** — `walking_env.py:592` confirms: `termination_penalty = -3.0 * self.low_height_steps`. After 20 low-height steps: -3 × 20 = -60. Max velocity tracking reward at `walking_env.py:398`: `25.0 * exp(-2.0 * error²)` = +25 at perfect tracking. The penalty-to-reward ratio makes standing safer than walking.

---

## Issue 6: Weak Transfer Initialization

### High Level

When transferring from standing to walking, you add 12 new input dimensions for velocity commands. Xavier initialization scales these weights by `sqrt(2 / (1496 + 512)) ≈ 0.002`. The new command features have essentially zero influence on the network's output. The agent has inherited excellent standing ability but is deaf to walking commands.

Transfer learning requires the new features to matter immediately, not after thousands of gradient updates.

### Simply Explained

You gave the agent new ears (command inputs) but connected them with nearly-invisible wires. It can't hear the commands. The standing policy works fine, so it just keeps standing—ignoring the new signals entirely.

**The fix:** Initialize command feature weights larger, or copy them from existing velocity features (which the standing network already understands). Use `--init-strategy velocity` when starting walking training.

> **⚠️ PARTIALLY VERIFIED** — `transfer_utils.py:364-372` shows Xavier initialization: `std = np.sqrt(2.0 / (fan_in + fan_out))`. With fan_in=1496, fan_out=512: std ≈ 0.032 (not 0.002 as claimed—the math is off by ~16x). However, the core issue remains valid: new command weights are ~3% scale relative to trained standing weights. The code does offer `INIT_FROM_VELOCITY` strategy at line 389-400 which copies velocity feature weights as a better alternative.

---

## Issue 7: Hyperparameters Tuned for Stability, Not Discovery

### High Level

Your PPO configuration prioritizes stable learning: moderate entropy (0.03), large batches (512), conservative learning rate (0.0002). These are appropriate for refining a working policy, not for discovering one from scratch. Stage 0 needs aggressive exploration to find that walking generates reward. Your settings encourage exploitation of the current strategy—which is standing still.

### Simply Explained

You've set the "try new things" dial to low and the "trust what works" dial to high. Since standing still "works" (passes the curriculum), the agent doubles down on standing. It needs permission to experiment wildly until it discovers that movement is rewarded.

**The fix:** For stage 0, increase entropy to 0.05, reduce batch size to 256, and raise the learning rate to 0.0003. Let the agent flail around until it finds something better than standing.

> **✅ VERIFIED** — `training_config.yaml:248-264` confirms:
> - `learning_rate: 0.0002` ✓
> - `batch_size: 512` ✓
> - `ent_coef: 0.03` ✓
>
> All values match. These are indeed conservative settings that favor exploitation over exploration.

---

## The Path Forward

These issues compound. The agent can't see the command, so it doesn't know to walk. When it accidentally walks, it dies from termination. If it survives, standing still passes anyway. The reward for trying is negative. The exploration rate is too low to escape this trap.

**Priority order:**

1. **Fix the exploit** — Tighten velocity tolerance so standing fails
2. **Extend survival** — Relax termination so walking attempts aren't fatal
3. **Simplify stage 0** — One task (forward), one speed (0.3), no variance
4. **Amplify the signal** — Make command visible in observation space
5. **Tune for exploration** — Higher entropy, smaller batches, faster learning

Each fix is surgical. None require architectural changes. The curriculum and reward structures are sound—they just need their thresholds adjusted to match the actual difficulty of learning to walk.

---

## Quick Reference: Key Thresholds to Change

| Parameter | Current | Recommended | File | Verified |
|-----------|---------|-------------|------|----------|
| `velocity_tolerances[0]` | 0.9 | 0.25 | `walking_curriculum.py:53` | ✅ |
| `termination_height_threshold` | 0.70 | 0.60 | `training_config.yaml:229` | ✅ |
| `early_termination_protection` | 50 | 150 | `training_config.yaml:232` | ✅ |
| `termination_height_recovery_window` | 20 | 80 | `training_config.yaml:231` | ✅ |
| `ent_coef` | 0.03 | 0.05 | `training_config.yaml:264` | ✅ |
| `batch_size` | 512 | 256 | `training_config.yaml:253` | ✅ |
| Stage 0 direction diversity | enabled | disabled | `walking_curriculum.py:63` | ✅ |
| Stage 0 speed range | [0.1, 0.3] | fixed 0.30 | `walking_curriculum.py:175` | ✅ |

---

## Additional Issues Found During Code Review

### Issue 8: Double Normalization of Commands

**Location:** `walking_env.py:708-709`, `transfer_utils.py:194-195`

Commands undergo two normalizations:
1. **Feature normalization:** `tanh(0.1 * x)` at `walking_env.py:709`
2. **VecNormalize:** Running mean/var statistics applied on top

The transfer utils initialize command feature stats with `mean=0.0, var=1.0`, but after `tanh(0.1 * 0.3) ≈ 0.03`, the actual range is ~[-0.3, 0.3]. VecNormalize then scales this based on observed variance, potentially amplifying or dampening the signal unpredictably.

**The fix:** Either:
- Skip feature normalization for command features (keep them in [-3, 3] m/s range)
- Or skip VecNormalize for commands (append them after normalization)

---

### Issue 9: AdaptiveRewardCalculator Goes Unused

**Location:** `rewards.py:292-336`, `walking_env.py:144`

A perfectly good `AdaptiveRewardCalculator` class exists that automatically adjusts reward weights per curriculum stage. Stage 0-1 would use higher stability weights, later stages would emphasize tracking. But `walking_env.py` uses the base `RewardCalculator` and implements its own ad-hoc reward logic spanning 300+ lines.

**The fix:** Use `AdaptiveRewardCalculator` and call `set_stage()` when curriculum advances. This would centralize reward tuning and reduce code duplication.

---

### Issue 10: Duplicate Reward Logic

**Location:** `walking_env.py:351-650`, `rewards.py:136-256`

The `RewardCalculator` computes tracking, direction, height, upright, and action penalties—but `_compute_task_reward()` in `walking_env.py` also implements:
- Its own velocity tracking reward (line 398)
- Its own height reward (lines 446-464)
- Its own upright reward (lines 467-480)
- Progress bonus, standing penalty, walking bonus, consistency bonus, sustained bonus...

The modular calculator's output is partially used (`vel_error`, `jerk_penalty`) but most of its computed rewards are ignored. This creates maintenance burden and potential inconsistency.

**The fix:** Either fully adopt the modular calculator or remove it. Don't half-use it.

---

### Issue 11: Standing Probability in Stage 0

**Location:** `walking_curriculum.py:48`

Stage 0 has 8% probability of commanding a standing (0,0,0) velocity. When the agent receives this, standing still is correct behavior and gets rewarded. This reinforces the exact exploit we're trying to break.

```python
self.standing_probability = [0.08, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0]
```

**The fix:** Set Stage 0 standing probability to 0%. Don't reward standing when we're trying to teach walking.

---

### Issue 12: Command Switch Interval Too Short

**Location:** `training_config.yaml:185`, `command_generator.py:32`

Commands switch every 2-5 seconds. At 100Hz simulation, that's 200-500 steps per command. But with the consistency bonus requiring 50 steps to kick in (`walking_env.py:535`), the agent has limited time to benefit from stable tracking.

More critically: a randomly initialized policy needs many attempts to discover that following a command yields reward. Switching commands every few seconds means success on one command doesn't generalize—the task appears random.

**The fix:** For Stage 0, extend `command_switch_interval` to [10.0, 20.0] seconds. Give the agent time to figure out one command before throwing another at it.

---

### Issue 13: Warmup Episodes May Delay Learning Signal

**Location:** `walking_curriculum.py:71`, `training_config.yaml:197`

The curriculum has a warmup period of 30-50 episodes where advancement criteria are relaxed by 50% (`walking_curriculum.py:251-254`). This sounds helpful but may mask the standing-still exploit longer—the agent can pass with worse velocity tracking during warmup.

**The fix:** Consider removing warmup leniency entirely, or use it only for episode length requirements, not velocity tracking.

---

### Issue 14: Minimum Episode Length May Be Too Low

**Location:** `walking_curriculum.py:57`

Stage 0 minimum episode length is 60 steps (0.6 seconds at 100Hz). An agent can achieve this by standing still without ever attempting locomotion.

```python
self.min_episode_lengths = [60, 100, 150, 200, 300, 400, 500]
```

Combined with the lenient velocity tolerance, this creates another path to "success" without walking.

**The fix:** Increase Stage 0 minimum to 200+ steps. If the agent can survive 2 seconds while tracking velocity, it's actually learning something.

---

## Summary: Additional Fixes

| Issue | Parameter | Current | Recommended | File |
|-------|-----------|---------|-------------|------|
| 8 | Feature norm on commands | enabled | disabled for cmd dims | `walking_env.py:708-709` |
| 11 | `standing_probability[0]` | 0.08 | 0.0 | `walking_curriculum.py:48` |
| 12 | `command_switch_interval` | [2.0, 5.0] | [10.0, 20.0] | `training_config.yaml:185` |
| 13 | `warmup_episodes` | 30 | 0 or apply only to length | `training_config.yaml:197` |
| 14 | `min_episode_lengths[0]` | 60 | 200 | `walking_curriculum.py:57` |
