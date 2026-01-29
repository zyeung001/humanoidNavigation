HUMANOID VIDEO RECORDING
============================================================
[visualization] MUJOCO_GL=egl
Creating custom walking environment...
 Loaded walking config from /content/humanoidNavigation/config/training_config.yaml
  Config: obs_history=4, tau=0.2
Using Humanoid-v5 for walking task
Walking environment observation space configuration (NEW):
  Base from env.observation_space: 350
  + Position inclusion adjustment: +15 → 365
  + COM features: 6
  = Body dim per frame: 371
  × History stack: 4
  = Stacked body dim: 1484
  + Command block (ONCE): 9
  = FROZEN dimension: 1493
  Velocity weight: 5.0
  Max commanded speed: 3.0
  Action smoothing tau: 0.2
 InferenceActionWarmup enabled: 10 warmup steps
Pre-warming environment to freeze observation dimension...
Environment observation space after reset: (1493,)
Loading VecNormalize from models/vecnorm_walking.pkl
 VecNormalize loaded and configured for inference
Observation space (walking): (1493,)
Observation space: Box(-inf, inf, (1493,), float32)
Loading SB3 PPO model from: models/final_walking_model.zip
/usr/local/lib/python3.12/dist-packages/stable_baselines3/common/on_policy_algorithm.py:150: UserWarning: You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU when not using a CNN policy (you are using ActorCriticPolicy which should be a MlpPolicy). See https://github.com/DLR-RM/stable-baselines3/issues/1245 for more info. You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU.Note: The model will train, but the GPU utilization will be poor and the training might take longer than on CPU.
  warnings.warn(
Trying codec: mp4v
Successfully initialized VideoWriter with mp4v
Starting video recording...

Recording episode 1/3
   Episode terminated at step 260, height=0.5855878675880866
  Episode 1: 260 steps, avg vel error: 1.9642 m/s, avg height: 0.921 m

Recording episode 2/3
Step  500: h=0.857, cmd=(0.00,0.00), actual=(0.11,0.05), yaw_err=1.060, vel_err=0.118, r=  19.8 [track=24.3, prog=0.0, stand_pen=0.0]
   Episode terminated at step 865, height=0.5481486465350625
  Episode 2: 865 steps, avg vel error: 1.0099 m/s, avg height: 0.872 m

Recording episode 3/3
   Episode terminated at step 116, height=0.3924836114064845
  Episode 3: 116 steps, avg vel error: 2.3703 m/s, avg height: 0.987 m

Video saved: data/videos/walking_performance.mp4 (1241 frames, 41.4s)

=== Walking Performance Summary ===
  Total velocity errors recorded: 1241
  Mean velocity error: 1.3370 m/s
  Std velocity error: 0.9921 m/s
  Mean height: 0.893 m
  Height std: 0.1531 m
============================================================
RECORDING COMPLETE!
Output: data/videos/walking_performance.mp4
============================================================

# Quadruped Walking Training Summary (Stage 0)
**Target velocity:** 0.3 m/s  
**Algorithm:** PPO  
**Period covered:** ~2.31M to ~4.42M total timesteps  
**Date of log:** (assumed recent as of Jan 2026)

## Overall Progress

| Timesteps     | Avg Velocity Error (m/s) | Best Recorded Error | Avg Episode Reward | Avg Ep Length | Success Rate | Action Std (approx) |
|---------------|---------------------------|----------------------|--------------------|---------------|--------------|---------------------|
| ~2.3M         | 0.46 – 0.70              | —                    | ~4,000–4,500      | 160–180      | 0%           | ~7e4                |
| ~3.0M         | 0.44 – 0.50              | —                    | ~5,000–6,000      | 200–250      | 0%           | ~1e5                |
| ~3.75M (peak) | **0.406** (new best)     | 0.406 m/s            | ~7,000–9,000      | 280–350      | 0%           | ~2e5                |
| ~4.0M         | 0.33 – 0.42              | ~0.30–0.33           | ~8,000–11,000     | 300–400      | 0%           | ~2.2e5              |
| ~4.4M (latest)| **0.30 – 0.36**          | ~0.30 m/s (multiple) | ~10,000–13,500    | 400–500+     | 0%           | ~2.55e5             |

## Key Trends & Observations

1. **Velocity Tracking Improvement** (primary success signal)
   - Started: 0.65–0.70 m/s average error
   - Mid: reached 0.406 m/s best at 3.75M steps
   - Latest: frequently 0.30–0.36 m/s average, with best 500-step windows often **0.10–0.25 m/s**
   - Clear downward trend → policy learning to match commanded 0.3 m/s more precisely

2. **Stability & Episode Length**
   - Early: falls quickly (~160–180 steps)
   - Latest: routinely survives 400–600+ steps
   - Still **0% success rate** → eventually loses balance or deviates too far, but survival time has roughly tripled

3. **Reward Growth**
   - From ~4k → recent peaks >13k
   - Strongly correlated with lower velocity error + longer episodes

4. **PPO Training Signals**
   - approx_kl: ↓ from ~0.02 → ~0.004–0.006 (smaller, more stable updates)
   - clip_fraction: ↓ from ~0.37 → ~0.16–0.20 (less aggressive clipping)
   - entropy_loss: slowly decreasing → policy becoming more deterministic
   - value_loss: consistently very low (~0.0015–0.003)
   - action std: steadily ↑ (~7e4 → 2.55e5) → exploration still active

5. **Entropy Coefficient Schedule**
   - Reduced multiple times: 0.059 → 0.026
   - Expected behavior: policy gradually becomes less random

## Current Status (as of ~4.42M steps)

Last updated from log: ~4.42M timesteps

# Quadruped Walking Training Summary - Part 2
**Target velocity:** 0.3 m/s  
**Algorithm:** PPO  
**Period covered:** ~4.425M → ~5.010M total timesteps  
**Entropy coefficient schedule:** Reduced to 0.0235 → 0.02095 → 0.0184 → 0.01585  
**Date range in log:** roughly late Jan 2026

## Overall Progress Snapshot

| Timesteps     | Avg Vel Error (m/s) | Best 500-step Vel Err | Avg Ep Len | Avg Ep Reward | Ep Len Mean (rollout) | Success Rate | Action Std     |
|---------------|----------------------|------------------------|------------|---------------|------------------------|--------------|----------------|
| ~4.425M       | 0.304 – 0.359       | ~0.30–0.33            | 416–507   | 10.2k–11.6k  | 494–595               | 0%           | ~2.57e5–2.63e5 |
| ~4.500M       | 0.351 (new best 0.337) | 0.337                | 637       | 14.35k       | ~591                  | 0%           | ~2.65e5        |
| ~4.650M       | 0.268–0.328         | ~0.27–0.30            | 610–721   | 15.2k–16.7k  | 645–776               | 0%           | ~2.72e5–2.85e5 |
| ~4.875M       | 0.280–0.332         | ~0.27–0.33            | 843–862   | 20.3k–20.3k  | 716–749               | 0%           | ~2.83e5–2.88e5 |
| ~5.010M (end) | 0.328–0.345         | ~0.27–0.33            | 763–768   | 18.8k–19.3k  | 749                   | 0%           | ~2.88e5        |

**Key milestone at 4.5M:** New best velocity error = **0.3373 m/s** → checkpoint saved  
**Training concluded at ~5.01M:** Final model saved

## Key Trends & Observations (4.4M → 5.0M)

1. **Velocity Tracking**
   - Frequent 500-step windows now achieve **0.05–0.25 m/s error** (very good for 0.3 m/s target)
   - Rolling average error fluctuates 0.27–0.39 m/s, but best segments consistently <0.35 m/s
   - Clear improvement in command following, especially forward velocity matching

2. **Stability & Survival Time**
   - Average episode length increased from ~450–550 → **700–850+ steps**
   - Many episodes now routinely reach **1500–2000 steps** without falling
   - Still **0% success rate** — robot eventually deviates (yaw drift, backward velocity, height collapse, or explosive instability)

3. **Reward Growth**
   - Avg episode reward climbed from ~10–14k → **18–20k+ peaks**
   - Strongly driven by longer episodes + better velocity tracking
   - Highest recorded mean reward ~20.3k at ~4.875M

4. **PPO Training Signals**
   - approx_kl: continued gentle decline (~0.004 → ~0.0027–0.0029)
   - clip_fraction: decreased to ~0.12–0.13 (very stable updates)
   - entropy_loss: stable around -235 → -236 (policy fairly deterministic)
   - value_loss: low and stable (~0.003–0.004)
   - action std: continued slow increase (~2.57e5 → 2.88e5) — exploration still present

5. **Common Failure Modes (from Step 500/1000/1500/2000 logs)**
   - **Yaw divergence** (yaw_err > 3–5 rad) → spinning or drifting sideways
   - Sudden **backward velocity spikes** (actual vx < -0.3 or vy large)
   - **Height collapse** (h < 0.3–0.5 m) → falling forward/backward
   - Rare explosive instability (very high vel/yaw errors in one step)
   - Occasional **standing exploit / penalty** still appears but much less frequent
