# config/ — Training Configuration

**The step-by-step training recipe lives in [`README.md`](README.md). Read it first.**

## Layout

| File | Purpose |
|------|---------|
| `training_config.yaml` | Base config — defaults for `standing`, `walking`, `maze` |
| `variants/03_yaw.yaml` | Stage 3 override: add yaw control to a walker |
| `variants/04_omni_sustained.yaml` | Stage 4 override: omnidirectional + sustained turning |
| `variants/05_tip.yaml` | Stage 5 override: turn-in-place |
| `variants/archive/` | Research history — see its README, do not use for new training |

Stages 1 (standing) and 2 (walking) use `training_config.yaml` directly with no override.

## `training_config.yaml`

Single YAML file with three top-level keys: `standing`, `walking`, and `maze`. Standing and walking share most parameter names but with different values.

## Parameter Groups

### Environment
- `n_envs`: Parallel environments (12 for both)
- `device`: `cuda` or `cpu`
- `max_episode_steps`: 10000 (standing), 5000 (walking)

### Observation Processing
- `obs_history`: Frame stacking count (4)
- `obs_include_com`: Include COM features (+6 dims)
- `obs_feature_norm`: Tanh feature normalization

### Curriculum
- `curriculum_start_stage` / `curriculum_max_stage`: Stage range
- `curriculum_advance_after`: Episodes before advancement check
- `curriculum_success_rate`: Required success rate to advance
- Walking-specific: `curriculum_max_speed_stages` list

### PPO Hyperparameters
- `learning_rate` / `final_learning_rate`: Linear decay schedule
- `n_steps`, `batch_size`, `n_epochs`: Standard PPO
- `clip_range` / `final_clip_range`: Clipping decay
- `ent_coef` / `final_ent_coef`: Entropy coefficient decay
- `gamma`: 0.995 (long horizon)

### Reward Weights (walking-specific)
- `reward_tracking_weight`: Velocity tracking (dominant signal)
- `reward_tracking_bandwidth`: Gaussian kernel sharpness
- `reward_direction_weight`, `reward_height_weight`, etc.
- `reward_arm_posture_weight`: Arm posture penalty

### Network Architecture
- `policy_kwargs.net_arch`: `[512, 512, 256]` for both policy and value
- `activation_fn`: SiLU
- `ortho_init`: Orthogonal initialization

### VecNormalize
- `vecnormalize_clip_obs` / `vecnormalize_clip_reward`
- Standing: 50.0 / 50.0
- Walking: 10.0 / 10.0 (tighter for stability)

## Tuning Guidance

**Safe to adjust:**
- `reward_tracking_weight` / `reward_tracking_bandwidth` — primary walking signal
- `reward_arm_posture_weight` — cosmetic (arm positioning)
- `curriculum_success_rate` — controls advancement speed
- `total_timesteps` — more training time

**Adjust with care:**
- `learning_rate` — too high causes instability, too low stalls
- `ent_coef` — too low kills exploration (agent gets stuck standing)
- `batch_size` / `n_epochs` — controls gradient steps per update (n_epochs × data/batch_size). 2048 batch + 2 epochs = 24 steps (stable). 512 batch + 10 epochs = 480 steps (KL explosion)
- `reward_caps` — affect termination/recovery behavior

**Do not change without understanding implications:**
- `vecnormalize_clip_obs/reward` — affects entire normalization pipeline
- `termination_height_threshold` — too strict kills walking, too loose masks failures
- `obs_history` — changes observation dimensions, breaks saved models
- `policy_kwargs.net_arch` — changes model architecture, breaks saved models
