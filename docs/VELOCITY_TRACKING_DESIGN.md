# Humanoid Velocity Tracking Training Design

## Overview

Train a humanoid to follow a **direction vector** (unit direction in world frame) and a **speed scalar** (magnitude in m/s). The policy receives commands and must achieve the target velocity while maintaining stability.

---

## 1. Command Representation

### Input Format
```python
# Option A: Direction + Speed (Recommended)
direction = [dx, dy]  # Unit vector, ||direction|| = 1
speed = s             # Scalar in m/s

# Converted to velocity command
vx = dx * s
vy = dy * s
```

### Command Bounds (Curriculum-Based)

| Stage | Speed Range | Direction | Notes |
|-------|-------------|-----------|-------|
| 0 | 0.0 - 0.3 m/s | Any | Very slow walk, learn balance |
| 1 | 0.0 - 0.6 m/s | Any | Slow walk |
| 2 | 0.0 - 1.0 m/s | Any | Normal walk |
| 3 | 0.0 - 1.5 m/s | Any | Fast walk |
| 4 | 0.0 - 2.0 m/s | Any | Light jog |
| 5 | 0.0 - 2.5 m/s | Any | Jog |
| 6 | 0.0 - 3.0 m/s | Any | Run |

### Direction Sampling Strategy
```python
# During training, sample directions with bias toward forward
direction_types = {
    'forward': 0.45,      # ±30° from +X
    'diagonal': 0.25,     # 45°, 135°, -45°, -135°
    'lateral': 0.15,      # ±90°
    'backward': 0.05,     # ±150-180°
    'random': 0.10        # Uniform [0, 2π]
}
```

---

## 2. Refined Reward Function

### Total Reward
```
R_total = R_tracking + R_stability + R_alive + R_action_penalty + R_smoothness
```

### Component 1: Velocity Tracking (Primary)
```python
# Gaussian kernel for velocity matching
v_target = np.array([vx_cmd, vy_cmd])
v_actual = np.array([vx_actual, vy_actual])

velocity_error = np.linalg.norm(v_target - v_actual)
R_tracking = 10.0 * np.exp(-4.0 * velocity_error**2)  # Max: 10

# Direction bonus (when speed > 0.1 m/s)
if speed_cmd > 0.1:
    # Project actual velocity onto commanded direction
    projected_speed = np.dot(v_actual, direction_cmd)
    direction_reward = 5.0 * np.clip(projected_speed / speed_cmd, 0, 1)
    R_tracking += direction_reward  # Max: +5 bonus
```

### Component 2: Stability (Upright + Height)
```python
# Height reward (target: 1.35-1.45m during walking)
height_error = abs(height - 1.40)
R_height = 5.0 * np.exp(-10.0 * height_error**2)  # Max: 5

# Upright orientation (quaternion w component)
quat_w = abs(quaternion[0])
R_upright = 3.0 if quat_w > 0.85 else 1.0 * quat_w  # Max: 3

R_stability = R_height + R_upright  # Max: 8
```

### Component 3: Alive Bonus
```python
R_alive = 1.0  # Constant per-step survival reward
```

### Component 4: Action Penalty (Effort)
```python
action_magnitude = np.sum(action**2)
R_action_penalty = -0.005 * action_magnitude  # Penalize large actions
```

### Component 5: Smoothness (Jerk Penalty)
```python
# Penalize abrupt action changes
action_delta = action - prev_action
jerk_penalty = np.sum(action_delta**2)
R_smoothness = -0.01 * jerk_penalty  # Penalize jerky control

# Track for WandB logging
self.jerk_history.append(jerk_penalty)
```

### Summary of Reward Components

| Component | Weight | Range | Purpose |
|-----------|--------|-------|---------|
| R_tracking | 10.0 | [0, 15] | Match commanded velocity |
| R_stability | 8.0 | [0, 8] | Stay upright and at target height |
| R_alive | 1.0 | [1, 1] | Survival bonus |
| R_action_penalty | -0.005 | [-∞, 0] | Energy efficiency |
| R_smoothness | -0.01 | [-∞, 0] | Smooth control |

**Expected per-step reward**: ~15-20 when tracking well, ~5-10 when learning

---

## 3. Parallelization Strategy

### Current Setup
- `n_envs: 8` with SubprocVecEnv
- `n_steps: 2048` (steps per env before update)
- **Effective batch**: 8 × 2048 = 16,384 timesteps per PPO update

### Recommended Improvements

```yaml
# For RTX 3090/4090 with 24GB VRAM
walking:
  n_envs: 32         # 4x increase (32 parallel environments)
  n_steps: 2048      # Keep same
  batch_size: 512    # 2x increase for larger effective batch
  
# Effective batch: 32 × 2048 = 65,536 timesteps per update
# This is 4x more data per update, significantly faster training
```

### Multi-GPU (Optional Future)
```python
# If multiple GPUs available, use torch distributed
# Each GPU runs its own set of environments
from stable_baselines3.common.vec_env import SubprocVecEnv

n_gpus = torch.cuda.device_count()
envs_per_gpu = 16
total_envs = n_gpus * envs_per_gpu  # e.g., 32 for 2 GPUs
```

---

## 4. Logging with WandB

### Metrics to Track

```python
# Every step (aggregate every 1000 steps)
"train/velocity_error": float,           # ||v_target - v_actual||
"train/velocity_error_x": float,         # vx error
"train/velocity_error_y": float,         # vy error
"train/direction_error": float,          # Angular error in radians
"train/jerk_penalty": float,             # Action smoothness metric
"train/action_magnitude": float,         # ||action||

# Every episode
"episode/length": int,
"episode/reward": float,
"episode/velocity_error_mean": float,
"episode/jerk_penalty_total": float,
"episode/curriculum_stage": int,
"episode/commanded_speed": float,

# Every eval (every 100k steps)
"eval/success_rate": float,              # Episodes meeting tracking threshold
"eval/mean_velocity_error": float,
"eval/mean_episode_length": float,
"eval/height_stability": float,

# Curriculum tracking
"curriculum/stage": int,
"curriculum/max_speed": float,
"curriculum/success_rate": float,
"curriculum/advancement_progress": float,
```

### WandB Integration Pattern

```python
import wandb

class WalkingWandBCallback(BaseCallback):
    def __init__(self, log_freq=1000):
        super().__init__()
        self.log_freq = log_freq
        self.velocity_errors = []
        self.jerk_penalties = []
        
    def _on_step(self):
        # Collect metrics
        for info in self.locals.get("infos", []):
            if 'velocity_error' in info:
                self.velocity_errors.append(info['velocity_error'])
            if 'jerk_penalty' in info:
                self.jerk_penalties.append(info['jerk_penalty'])
        
        # Log aggregated metrics
        if self.num_timesteps % self.log_freq == 0:
            wandb.log({
                "train/velocity_error": np.mean(self.velocity_errors[-100:]),
                "train/jerk_penalty": np.mean(self.jerk_penalties[-100:]),
                "train/timesteps": self.num_timesteps,
            })
        return True
```

---

## 5. Weight Storage Organization

### Directory Structure
```
models/
├── walking/
│   ├── latest/
│   │   ├── model.zip              # Latest checkpoint
│   │   └── vecnorm.pkl            # VecNormalize stats
│   │
│   ├── best/
│   │   ├── model.zip              # Best performing model
│   │   └── vecnorm.pkl
│   │
│   ├── checkpoints/
│   │   ├── stage_0/
│   │   │   ├── model_100k.zip
│   │   │   ├── model_200k.zip
│   │   │   └── ...
│   │   ├── stage_1/
│   │   │   └── ...
│   │   └── ...
│   │
│   └── final/
│       ├── model.zip              # Final production model
│       └── vecnorm.pkl
│
├── standing/                       # Same structure for standing
│   └── ...
│
└── config/
    ├── training_config.yaml       # Symlink to active config
    └── run_20241217_143022.yaml   # Archived config per run
```

### Checkpoint Naming Convention
```
{task}_{stage}_{timesteps}_{velocity_error:.3f}.zip
# Example: walking_s3_5M_0.152.zip
```

---

## 6. Curriculum Velocity Bounds

### Update Strategy

```python
class VelocityBoundsCurriculum:
    """
    Progressive velocity bounds that expand with training success.
    """
    
    def __init__(self):
        self.stages = [
            # (min_speed, max_speed, vel_tolerance, min_episode_len)
            (0.0, 0.3, 0.5, 100),    # Stage 0: Baby steps
            (0.0, 0.6, 0.45, 150),   # Stage 1: Slow walk
            (0.0, 1.0, 0.4, 200),    # Stage 2: Normal walk
            (0.0, 1.5, 0.35, 300),   # Stage 3: Fast walk
            (0.0, 2.0, 0.3, 400),    # Stage 4: Light jog
            (0.0, 2.5, 0.28, 500),   # Stage 5: Jog
            (0.0, 3.0, 0.25, 600),   # Stage 6: Run
        ]
        
    def get_bounds(self, stage: int):
        stage = min(stage, len(self.stages) - 1)
        return self.stages[stage]
    
    def should_advance(self, success_rate: float, avg_vel_error: float, 
                       avg_episode_length: float, current_stage: int) -> bool:
        """
        Advancement criteria:
        1. Success rate >= 60%
        2. Avg velocity error < tolerance * 1.2
        3. Avg episode length > min_length * 1.5
        """
        _, _, tolerance, min_len = self.stages[current_stage]
        return (
            success_rate >= 0.60 and
            avg_vel_error < tolerance * 1.2 and
            avg_episode_length > min_len * 1.5
        )
```

---

## 7. Implementation Tickets

### Ticket 1: Refactor Reward Function
**Priority**: HIGH | **Estimate**: 2 hours

- [ ] Create `RewardCalculator` class in `src/utils/rewards.py`
- [ ] Implement Gaussian kernel tracking reward
- [ ] Add jerk penalty (action smoothness)
- [ ] Add direction bonus for speed > 0.1 m/s
- [ ] Track individual reward components for logging
- [ ] Unit tests for reward calculation

### Ticket 2: Enhance Command Generator Integration
**Priority**: HIGH | **Estimate**: 1 hour

- [ ] Replace `WalkingEnv` velocity sampling with `VelocityCommandGenerator`
- [ ] Support direction + speed input format
- [ ] Add yaw rate tracking (optional)
- [ ] Implement stop command probability (15%)

### Ticket 3: Improve Parallelization
**Priority**: MEDIUM | **Estimate**: 1 hour

- [ ] Increase `n_envs` from 8 to 32 in config
- [ ] Test memory usage with 32 envs
- [ ] Adjust `batch_size` to 512
- [ ] Add config option for auto-detecting optimal n_envs based on GPU memory

### Ticket 4: WandB Integration
**Priority**: HIGH | **Estimate**: 2 hours

- [ ] Create `WalkingWandBCallback` class
- [ ] Log velocity error (per-axis and magnitude)
- [ ] Log jerk penalty (action smoothness)
- [ ] Log curriculum progression
- [ ] Add video logging every 500k steps
- [ ] Dashboard template for walking metrics

### Ticket 5: Weight Storage Refactor
**Priority**: MEDIUM | **Estimate**: 1.5 hours

- [ ] Create `ModelManager` class in `src/utils/model_manager.py`
- [ ] Implement directory structure from design
- [ ] Auto-save best model based on eval metrics
- [ ] Checkpoint cleanup (keep last N per stage)
- [ ] Config archiving per run

### Ticket 6: Velocity Bounds Curriculum
**Priority**: HIGH | **Estimate**: 1.5 hours

- [ ] Create `VelocityBoundsCurriculum` class
- [ ] Integrate with `WalkingCurriculumEnv`
- [ ] Add advancement logging to WandB
- [ ] Test curriculum progression with mock episodes

### Ticket 7: Action Smoothness Enforcement
**Priority**: MEDIUM | **Estimate**: 1 hour

- [ ] Implement jerk penalty in reward function
- [ ] Add `action_smoothing_tau` curriculum (start high, decrease)
- [ ] Log smoothness metrics to WandB
- [ ] Visualize action trajectories in eval

### Ticket 8: Config Cleanup
**Priority**: LOW | **Estimate**: 0.5 hours

- [ ] Consolidate walking config with new parameters
- [ ] Add validation for config values
- [ ] Document all config options
- [ ] Remove deprecated options

---

## 8. Quick Start Commands

```bash
# Train from scratch
python scripts/train_walking.py --timesteps 15000000

# Resume training
python scripts/train_walking.py --model models/walking/latest/model.zip \
                                --vecnorm models/walking/latest/vecnorm.pkl

# Test with specific velocity
python scripts/record_video.py --task walking \
                               --model models/walking/best/model.zip \
                               --vx_target 1.0 --vy_target 0.0

# Visualize command generator
python src/utils/plot_velocity_commands.py
```

---

## 9. Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Velocity tracking error | < 0.25 m/s | At all curriculum stages |
| Episode length | > 2000 steps | At max speed (3 m/s) |
| Direction accuracy | < 15° error | When speed > 0.5 m/s |
| Action smoothness | Jerk < 0.1 | Low action discontinuity |
| Training time | < 24 hours | 15M steps on RTX 3090 |

---

## 10. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐     ┌──────────────────────────────────┐ │
│  │ VelocityCommand  │────>│         WalkingEnv               │ │
│  │    Generator     │     │  ┌────────────────────────────┐  │ │
│  │                  │     │  │     RewardCalculator       │  │ │
│  │ - dir + speed    │     │  │  - R_tracking (Gaussian)   │  │ │
│  │ - 15% stop prob  │     │  │  - R_stability             │  │ │
│  │ - 2-5s intervals │     │  │  - R_smoothness (jerk)     │  │ │
│  └──────────────────┘     │  └────────────────────────────┘  │ │
│                           └──────────────────────────────────┘ │
│                                         │                       │
│                                         ▼                       │
│  ┌──────────────────┐     ┌──────────────────────────────────┐ │
│  │   SubprocVecEnv  │◄────│          PPO Policy              │ │
│  │   (32 parallel)  │     │  - [512, 512, 256] MLP           │ │
│  └──────────────────┘     │  - SiLU activation               │ │
│                           │  - Entropy decay                 │ │
│                           └──────────────────────────────────┘ │
│                                         │                       │
│                                         ▼                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    WandB Logging                         │  │
│  │  - velocity_error, jerk_penalty, curriculum_stage        │  │
│  │  - episode_reward, action_magnitude                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

