# Training Checklist - Humanoid Standing Task

Use this checklist to monitor your training and verify the fixes are working.

---

## Pre-Training Checklist

### ☐ Environment Setup
- [ ] All dependencies installed (gymnasium, mujoco, stable-baselines3)
- [ ] GPU available and detected (`nvidia-smi` shows GPU)
- [ ] Sufficient disk space (>10GB for models and logs)
- [ ] WandB configured (optional but recommended)

### ☐ Clean Start
- [ ] Deleted old `models/vecnorm.pkl`
- [ ] Deleted old model files in `models/`
- [ ] Cleared old logs in `data/logs/` (optional)
- [ ] Verified all fixes are applied (check file modification dates)

### ☐ Configuration Verified
```bash
# Check entropy coefficient
grep "final_ent_coef" config/training_config.yaml
# Should show: final_ent_coef: 0.005 (NOT 0.01 or negative!)

# Check batch size
grep "batch_size" config/training_config.yaml
# Should show: batch_size: 512

# Check GAE lambda
grep "gae_lambda" config/training_config.yaml  
# Should show: gae_lambda: 0.90
```

---

## Training Progress Milestones

### Milestone 1: Initial Learning (0-500K steps)

**Expected behavior:**
- [ ] Curriculum at stage 0 or 1
- [ ] Mean height: 0.9-1.1m
- [ ] Episode length: 100-300 steps
- [ ] Entropy: 0.018-0.020
- [ ] No NaN values in logs

**Red flags:**
- ⚠️ Entropy becomes negative → STOP and check `train_standing.py`
- ⚠️ Mean reward is NaN → Reward scale still too large
- ⚠️ Agent immediately falls every episode → Check observation space

### Milestone 2: Curriculum Stage 2 (500K-1.5M steps)

**Expected behavior:**
- [ ] Curriculum at stage 2
- [ ] Mean height: 1.20-1.30m
- [ ] Episode length: 300-600 steps
- [ ] Entropy: 0.012-0.018
- [ ] Action magnitude: 140-180

**Red flags:**
- ⚠️ Stuck at stage 0/1 for >800K steps → Increase success rate
- ⚠️ Height decreasing → Reward shaping issue
- ⚠️ Episode length not increasing → Termination penalty not working

### Milestone 3: Curriculum Stage 3 (1.5M-2.5M steps)

**Expected behavior:**
- [ ] Curriculum at stage 3
- [ ] Mean height: 1.30-1.38m
- [ ] Episode length: 600-900 steps
- [ ] Entropy: 0.008-0.012
- [ ] Success rate at stage 3: >60%

**Red flags:**
- ⚠️ Can't advance past stage 2 → Tolerance too tight, or batch size too small
- ⚠️ Height variance >0.15m → Agent not stable
- ⚠️ Action magnitude <120 → Control cost too high

### Milestone 4: Final Stage 4 (2.5M-5M steps)

**Expected behavior:**
- [ ] Curriculum at stage 4 (target 1.40m)
- [ ] Mean height: 1.38-1.42m
- [ ] Episode length: 1000-1500+ steps
- [ ] Entropy: 0.005-0.010
- [ ] Height stability: <0.10m std
- [ ] Success rate at stage 4: >60%

**Red flags:**
- ⚠️ Can't reach stage 4 → Curriculum too difficult
- ⚠️ Height stuck at 1.26m → Reward peak not sharp enough
- ⚠️ Episodes still terminating <800 steps → Termination penalty too weak

---

## Real-Time Monitoring (Every 100K Steps)

### Check WandB Dashboard

#### Key Plots to Monitor

1. **train/mean_height**
   - Should steadily increase: 1.0 → 1.15 → 1.25 → 1.35 → 1.40
   - Should stabilize around 1.38-1.42 at stage 4

2. **train/episode_length**
   - Should increase over time
   - Should reach >1200 consistently at stage 4

3. **train/policy/policy_std_mean** (entropy proxy)
   - Should stay >0.05 throughout training
   - Never go to zero or negative!

4. **curriculum_stage**
   - Should progress: 0 → 1 → 2 → 3 → 4
   - Should reach stage 4 by ~2.5M steps

5. **train/action_magnitude_mean**
   - Should stabilize around 150-200
   - Should NOT continuously decrease

#### Reward Components Plot

Check `train/reward_components/*`:
- height_reward should be largest component (+30 to +50)
- termination_penalty should be 0 most of the time
- sustained_bonus should appear every 100 steps

---

## Testing During Training

### Quick Evaluation (Every 1M Steps)

```bash
# Record a test video
python scripts/record_video.py \
    --model models/best_standing_model \
    --vecnorm models/vecnorm.pkl \
    --episodes 3 \
    --max-steps 2000 \
    --output-dir data/test_videos
```

**Look for:**
- [ ] Agent stands for >500 steps
- [ ] Height visually around chest/head level
- [ ] No violent oscillations
- [ ] Recovers from small perturbations

### Detailed Analysis (Optional)

```python
from src.agents.standing_agent import StandingAgent

# Load config and model
agent = StandingAgent.load_model("models/best_standing_model")

# Run detailed analysis
results = agent.analyze_standing_performance(n_episodes=10)
print(results)
```

---

## Final Validation (After 5M Steps)

### ☐ Quantitative Metrics

```bash
python scripts/record_video.py \
    --model models/final_standing_model \
    --vecnorm models/vecnorm.pkl \
    --episodes 10 \
    --max-steps 5000
```

**Success criteria:**
- [ ] Mean episode length: >1200 steps (at least 6/10 episodes)
- [ ] Mean height: 1.38-1.42m
- [ ] Height std: <0.08m
- [ ] Early termination rate: <20%
- [ ] No episodes terminate in first 500 steps

### ☐ Qualitative Behavior

Watch the recorded videos:
- [ ] Agent maintains upright posture
- [ ] Height visually appears correct (~1.4m)
- [ ] Movements are smooth, not jerky
- [ ] Agent doesn't drift excessively in XY plane
- [ ] Recovers from small perturbations naturally

### ☐ Robustness Tests (Advanced)

1. **Long duration test**:
   ```bash
   python scripts/record_video.py --max-steps 10000
   # Should survive most/all of the 10K steps
   ```

2. **Multiple seeds test**:
   ```bash
   for seed in 1 2 3 4 5; do
       python scripts/record_video.py --seed $seed --episodes 5
   done
   # Should succeed across different seeds
   ```

3. **Domain randomization test**:
   - Manually increase mass/friction randomization in config
   - Agent should still maintain reasonable performance

---

## Troubleshooting During Training

### Problem: Training diverges (NaN values)

**Likely cause:** Gradient explosion

**Solutions:**
1. Reduce learning rate: `learning_rate: 0.0001`
2. Increase gradient clipping: `max_grad_norm: 0.3`
3. Reduce batch size back to 256
4. Check for reward spikes in logs

### Problem: Stuck at low height (<1.0m)

**Likely cause:** Agent learned to crouch

**Solutions:**
1. Increase penalty for height < 1.0m (change -50 to -100)
2. Reduce termination threshold from 0.75m to 0.85m
3. Initialize episodes with random height perturbations
4. Check if control cost is too high

### Problem: Can't advance past stage 2

**Likely cause:** Curriculum too strict

**Solutions:**
1. Reduce success rate: `curriculum_success_rate: 0.50`
2. Increase tolerance: Add 0.02m to all tolerances
3. Reduce min episode length by 100 steps at each stage
4. Check if episodes are terminating too early

### Problem: Height oscillates wildly

**Likely cause:** Insufficient stability reward

**Solutions:**
1. Increase stability reward weight (multiply by 1.5)
2. Increase action smoothing: `tau: 0.3`
3. Add joint velocity penalty
4. Reduce entropy coefficient to 0.003

### Problem: Episodes still terminate early at stage 4

**Likely cause:** Termination penalty too weak

**Solutions:**
1. Increase termination penalty to -200
2. Add death penalty: Additional -500 on termination
3. Reduce termination threshold (make more strict)
4. Increase episode length requirement

---

## Post-Training Deployment

### ☐ Model Packaging

- [ ] Final model saved: `models/final_standing_model.zip`
- [ ] VecNormalize saved: `models/vecnorm.pkl`
- [ ] Config saved: `config/training_config.yaml`
- [ ] Training logs archived
- [ ] WandB run saved/exported

### ☐ Documentation

- [ ] Document final training metrics
- [ ] Save example videos
- [ ] Note any remaining issues
- [ ] Record inference performance benchmarks

### ☐ Integration Testing

- [ ] Test in target deployment environment
- [ ] Verify real-time performance (if needed)
- [ ] Test with realistic perturbations
- [ ] Validate safety behaviors

---

## Quick Reference: Expected Training Timeline

| Timestep | Stage | Height | Episode Length | Key Event |
|----------|-------|--------|----------------|-----------|
| 0K | 0 | 0.9-1.1m | 100-200 | Initial learning |
| 500K | 1 | 1.1-1.2m | 200-400 | Stage 1 advancement |
| 1.0M | 2 | 1.2-1.3m | 400-600 | Stage 2 advancement |
| 2.0M | 3 | 1.3-1.38m | 600-900 | Stage 3 advancement |
| 3.0M | 4 | 1.38-1.42m | 900-1200 | Stage 4 reached! |
| 4.0M | 4 | 1.39-1.41m | 1200-1500 | Fine-tuning |
| 5.0M | 4 | 1.39-1.41m | 1500+ | Training complete |

**Total training time:** 12-18 hours on RTX 3090 / A100

---

## Success! What Next?

If your agent passes all validation criteria:

1. **Document your results** - Save videos, metrics, and observations
2. **Push to production** - Deploy with confidence
3. **Consider extensions**:
   - Add walking behavior on top of standing
   - Train for robustness to external forces
   - Add vision-based control
   - Transfer to different humanoid models

4. **Share your success** - Consider contributing improvements back!

---

**Remember:** If training isn't working as expected, refer to `docs/fixes_analysis.md` for detailed troubleshooting.

**Good luck with your training! 🎯**

