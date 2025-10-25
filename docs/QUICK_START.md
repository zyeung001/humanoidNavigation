# Quick Start Guide - Fixed Humanoid Standing

## 🚀 Quick Start (3 Steps)

### 1. Test the Reward Function
```bash
python scripts/test/test_reward_function.py
```
**Expected output:** All 6 tests should pass ✓

### 2. Start Training
```bash
python scripts/train_standing.py
```
**Monitor:** Check W&B dashboard (link printed at start)

### 3. Watch Progress
Key metrics to watch on W&B:
- `train/episode_reward` - should increase from ~5,000 to 50,000+
- `train/height_error` - should decrease to < 0.1m
- `train/episode_length` - should increase to 1000+ steps
- `video/training_progress` - visual confirmation

---

## 📊 What Changed?

### The Problem
Old reward was **negative** even for good standing → agent couldn't learn

### The Fix
New reward is **predominantly positive** (10-100 points/step):
- Base: +10 (always)
- Height: +0 to +50
- Upright: +0 to +20
- Stability: +0 to +10
- Smoothness: +0 to +5
- Control: -5 to 0
- **Total: 10-100/step** (typical: 50-85)

---

## 🎯 Expected Results

### Training Progress (2M timesteps, ~4-8 hours on GPU)

| Timesteps | Episode Reward | Height Error | Episode Length | Status |
|-----------|---------------|--------------|----------------|---------|
| 0 | ~5,000 | 0.3-0.5m | 100-200 | Random |
| 250k | ~15,000 | 0.2-0.3m | 300-500 | Learning |
| 500k | ~30,000 | 0.15-0.2m | 500-800 | Improving |
| 1M | ~45,000 | 0.1-0.15m | 800-1200 | Good |
| 1.5M | ~60,000 | 0.05-0.1m | 1200-2000 | Very Good |
| 2M | **70,000+** | **< 0.1m** | **2000+** | **Success!** |

---

## 📈 W&B Dashboard Guide

### Key Metrics to Monitor

**Primary Success Indicators:**
1. `train/episode_reward` → Target: > 50,000
2. `train/height_error` → Target: < 0.1m
3. `train/height_stability` → Target: < 0.1m std
4. `train/episode_length` → Target: > 1000 steps

**Reward Components (for debugging):**
- `train/reward_components/height_reward` → Should be 30-50
- `train/reward_components/upright_reward` → Should be 15-20
- `train/reward_components/stability_reward` → Should be 5-10
- `train/reward_components/control_cost` → Should be -5 to 0

**Policy Health:**
- `train/policy/learning_rate` → Should be ~0.0005
- `train/policy/policy_std_mean` → Should decrease over time (0.6 → 0.3)

**Videos:**
- `video/training_progress` → Check every 50k steps
- Should see humanoid standing more upright over time

---

## 🔧 Troubleshooting

### Agent Not Learning (reward not increasing)
**Check:**
1. Are rewards positive? (`train/episode_reward` > 0)
2. Is height_error decreasing?
3. Are episodes getting longer?

**Try:**
- Increase entropy: `ent_coef: 0.05` (was 0.03)
- Increase learning rate: `learning_rate: 0.001` (was 0.0005)
- Check GPU utilization

### Agent Learning But Unstable (high height_stability)
**Check:**
- `train/reward_components/stability_reward` - is it low?
- `train/action_magnitude_max` - are actions too large?

**Try:**
- Increase stability reward weight in `standing_env.py`
- Decrease control cost penalty
- Lower `log_std_init` in config (more deterministic policy)

### Training Too Slow
**Try:**
- Increase `n_envs: 16` (was 8)
- Increase `learning_rate: 0.001` (was 0.0005)
- Reduce `n_epochs: 5` (was 10)

---

## 🧪 Testing Commands

```bash
# Test reward function (before training)
python scripts/test/test_reward_function.py

# Test specific component
python scripts/test/test_reward_function.py --test standing
python scripts/test/test_reward_function.py --test scaling
python scripts/test/test_reward_function.py --test viz

# Test environment
python -c "from src.environments.standing_env import test_environment; test_environment()"

# Quick training test (1000 steps)
python scripts/train_standing.py  # Ctrl+C after 1000 steps
```

---

## 📁 Important Files

### Configuration
- `config/training_config.yaml` - All hyperparameters
- `config/environment_config.yaml` - Environment settings

### Core Code
- `src/environments/standing_env.py` - Reward function (REDESIGNED)
- `src/agents/standing_agent.py` - PPO agent + W&B logging
- `scripts/train_standing.py` - Training script

### Testing
- `scripts/test/test_reward_function.py` - Reward validation (NEW)
- `scripts/test/test_standing.py` - Policy testing

### Documentation
- `FIXES_SUMMARY.md` - Complete fix documentation (NEW)
- `EXPERT_PROMPT.txt` - Original problem analysis
- `ISSUE_SUMMARY.txt` - Detailed issues
- `README.md` - Project overview

---

## 🎓 Key Concepts

### Why the Old Reward Failed
1. **Negative baseline** → No incentive to stay upright
2. **Velocity penalty** → Punished necessary balancing movements
3. **Exponential cliff** → Small errors = massive reward drop
4. **Wrong target height** → 1.3m instead of 1.4m

### Why the New Reward Works
1. **Positive baseline (+10)** → Clear signal that upright is good
2. **Stability reward** → Encourages low angular momentum (not zero velocity)
3. **Gentle scaling** → 10cm error still gives ~60 points/step
4. **Correct target (1.4m)** → Matches Humanoid-v5 natural height

### Standing ≠ Locomotion
- Standing is **stabilization** (equilibrium), not trajectory following
- Requires **continuous small corrections** (not zero action)
- Reward should be **highest at equilibrium** (upright, correct height)

---

## 📞 Support

### If Something Breaks
1. Check `FIXES_SUMMARY.md` for detailed explanations
2. Run test suite: `python scripts/test/test_reward_function.py`
3. Check W&B logs for anomalies
4. Review `ISSUE_SUMMARY.txt` for original problems

### Common Issues
- **Import errors:** Install requirements: `pip install -r requirements.txt`
- **MuJoCo errors:** Set `export MUJOCO_GL=egl` (headless) or `glfw` (with display)
- **W&B errors:** Login: `wandb login` or disable: `use_wandb: false` in config
- **GPU errors:** Set `device: "cpu"` in config for CPU training

---

## 🎯 Success Checklist

Before considering training complete, verify:
- [ ] Episode reward > 50,000 (consistently)
- [ ] Height error < 0.1m (90%+ of time)
- [ ] Height stability < 0.1m std
- [ ] Episode length > 1000 steps
- [ ] Agent stands for 5000+ steps in evaluation
- [ ] Videos show stable, upright standing

---

## 🚀 Advanced Usage

### Resume Training
```bash
python scripts/train_standing.py --resume data/checkpoints/checkpoint_500000.zip
```

### Evaluate Trained Model
```bash
python scripts/train_standing.py --eval models/saved_models/best_standing_model.zip --render
```

### Custom Hyperparameters
Edit `config/training_config.yaml` and restart training.

### Curriculum Learning (Optional)
Modify `standing_env.py` to gradually increase difficulty:
- Phase 1: Reward for height > 1.0m (easy)
- Phase 2: Reward for height ≈ 1.4m (medium)
- Phase 3: Add stability requirements (hard)

---

## 📚 Further Reading

- **PPO Paper:** "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- **Humanoid Control:** "Emergence of Locomotion Behaviours in Rich Environments" (DeepMind, 2017)
- **Reward Shaping:** "Policy Invariance Under Reward Transformations" (Ng et al., 1999)

---

**Last Updated:** October 25, 2025  
**Branch:** `user/andrew-kent`  
**Status:** ✅ Ready for training

