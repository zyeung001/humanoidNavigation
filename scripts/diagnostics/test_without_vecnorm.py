"""
Test inference WITHOUT VecNormalize to see if corrupted statistics are the issue
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.environments.standing_curriculum import make_standing_curriculum_env

print("="*60)
print("TEST: Inference WITHOUT VecNormalize")
print("="*60)
print()

config = {
    'obs_history': 4,
    'obs_include_com': True,
    'obs_feature_norm': True,
    'action_smoothing': True,
    'action_smoothing_tau': 0.2,
    'curriculum_start_stage': 3,
    'curriculum_max_stage': 3,
}

# Create environment WITHOUT VecNormalize
env = DummyVecEnv([lambda: make_standing_curriculum_env(render_mode=None, config=config)])
print("✓ Created environment WITHOUT VecNormalize")

# Load model
model_path = "models/saved_models/best_standing_model.zip"
try:
    # Load model WITHOUT env binding (to avoid VecNormalize requirement)
    model = PPO.load(model_path, env=env)
    print(f"✓ Loaded model from {model_path}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

print("\nRunning 5 test episodes WITHOUT normalization...")
print()

results = []
for episode in range(5):
    obs = env.reset()
    steps = 0
    
    while steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        steps += 1
        
        if steps % 50 == 0:
            print(f"  Episode {episode+1}, Step {steps}: height={info[0]['height']:.3f}")
        
        if done[0]:
            print(f"  ❌ Episode {episode+1} TERMINATED at step {steps}, height={info[0]['height']:.3f}")
            break
    else:
        print(f"  ✓ Episode {episode+1} COMPLETED 1000 steps successfully!")
    
    results.append(steps)

print()
print("="*60)
print("RESULTS WITHOUT VECNORMALIZE:")
print("="*60)
print(f"  Mean survival: {np.mean(results):.1f} ± {np.std(results):.1f} steps")
print(f"  Range: {np.min(results)}-{np.max(results)} steps")
print(f"  Success rate: {np.sum(np.array(results) >= 1000) / len(results) * 100:.0f}%")
print()

if np.mean(results) > 300:
    print("✓ SIGNIFICANT IMPROVEMENT without VecNormalize!")
    print("  → Issue: VecNormalize statistics are corrupted")
    print("  → Solution: Retrain VecNormalize statistics OR disable normalization")
else:
    print("✗ No improvement - normalization is not the issue")

env.close()

