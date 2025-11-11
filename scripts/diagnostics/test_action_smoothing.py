"""
Diagnostic Experiment 1: Test Action Smoothing Train-Test Mismatch

This script tests whether resetting prev_action to zero causes inference failure.
"""

import os
import sys
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environments.standing_curriculum import make_standing_curriculum_env


def test_inference(model_path, vecnorm_path, test_name="Test"):
    """Run inference test and report results"""
    
    # Test config matching training
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.2,
        'curriculum_start_stage': 3,
        'curriculum_max_stage': 3,
    }
    
    # Create environment
    env = DummyVecEnv([lambda: make_standing_curriculum_env(render_mode=None, config=config)])
    
    # Load VecNormalize
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"✓ Loaded VecNormalize from {vecnorm_path}")
    else:
        print(f"✗ VecNormalize not found at {vecnorm_path}")
        return
    
    # Load model
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"✗ Model not found at {model_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"{test_name}")
    print(f"{'='*60}")
    
    # Run 5 episodes
    results = []
    heights_at_termination = []
    
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
                heights_at_termination.append(info[0]['height'])
                break
        else:
            print(f"  ✓ Episode {episode+1} COMPLETED {steps} steps successfully!")
        
        results.append(steps)
    
    print(f"\n{test_name} RESULTS:")
    print(f"  Mean survival: {np.mean(results):.1f} ± {np.std(results):.1f} steps")
    print(f"  Range: {np.min(results)}-{np.max(results)} steps")
    if heights_at_termination:
        print(f"  Mean termination height: {np.mean(heights_at_termination):.3f}m")
    print(f"  Success rate: {np.sum(np.array(results) >= 1000) / len(results) * 100:.0f}%")
    
    env.close()
    return results


def main():
    # Default paths
    model_path = "models/saved_models/best_standing_model.zip"
    vecnorm_path = "models/saved_models/vecnorm_standing.pkl"
    
    # Check if files exist with alternative paths
    if not os.path.exists(model_path):
        alt_model = "models/standing_model/best_standing_model.zip"
        if os.path.exists(alt_model):
            model_path = alt_model
    
    if not os.path.exists(vecnorm_path):
        alt_vecnorm = "vecnorm.pkl"
        if os.path.exists(alt_vecnorm):
            vecnorm_path = alt_vecnorm
    
    print("="*60)
    print("DIAGNOSTIC EXPERIMENT 1: Action Smoothing Train-Test Mismatch")
    print("="*60)
    print(f"\nModel: {model_path}")
    print(f"VecNormalize: {vecnorm_path}")
    print()
    
    # Test with current implementation
    print("\n" + "="*60)
    print("BASELINE: Current implementation (with prev_action reset)")
    print("="*60)
    results_baseline = test_inference(model_path, vecnorm_path, "BASELINE TEST")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Edit src/environments/standing_env.py line 110")
    print("   Comment out: self.prev_action[:] = 0.0")
    print("2. Re-run this script to compare results")
    print("3. If mean survival increases from ~140 to 500+, Issue #1 is confirmed!")
    print()
    
    if np.mean(results_baseline) < 200:
        print("⚠️  Agent is failing very quickly (< 200 steps)")
        print("    This strongly suggests action smoothing mismatch is the culprit.")
    elif np.mean(results_baseline) < 500:
        print("⚙️  Agent survives 200-500 steps")
        print("    Multiple issues may be contributing to failure.")
    else:
        print("✓ Agent is performing reasonably well (> 500 steps)")
        print("   Issue #1 may not be the primary problem.")


if __name__ == "__main__":
    main()

