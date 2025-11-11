"""
Diagnostic Experiment 3: Test Stochastic vs Deterministic Inference

This script tests whether the agent relies on action noise for stability.
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


def test_inference_mode(model, env, deterministic, n_episodes=10):
    """Run inference in deterministic or stochastic mode"""
    results = []
    heights_at_termination = []
    
    mode_name = "DETERMINISTIC" if deterministic else "STOCHASTIC"
    print(f"\n{mode_name} MODE:")
    print("-" * 40)
    
    for episode in range(n_episodes):
        obs = env.reset()
        steps = 0
        
        while steps < 1000:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            steps += 1
            
            if done[0]:
                print(f"  Episode {episode+1:2d}: {steps:4d} steps, height={info[0]['height']:.3f}m (terminated)")
                heights_at_termination.append(info[0]['height'])
                break
        else:
            print(f"  Episode {episode+1:2d}: {steps:4d} steps (completed successfully)")
        
        results.append(steps)
    
    print(f"\n{mode_name} RESULTS:")
    print(f"  Mean survival: {np.mean(results):6.1f} ± {np.std(results):5.1f} steps")
    print(f"  Median: {np.median(results):6.0f} steps")
    print(f"  Range: {np.min(results):4d} - {np.max(results):4d} steps")
    if heights_at_termination:
        print(f"  Mean termination height: {np.mean(heights_at_termination):.3f}m")
    print(f"  Success rate (≥1000 steps): {np.sum(np.array(results) >= 1000) / len(results) * 100:.0f}%")
    
    return results


def main():
    # Default paths
    model_path = "models/saved_models/best_standing_model.zip"
    vecnorm_path = "models/saved_models/vecnorm_standing.pkl"
    
    # Check alternative paths
    if not os.path.exists(model_path):
        alt_model = "models/standing_model/best_standing_model.zip"
        if os.path.exists(alt_model):
            model_path = alt_model
    
    if not os.path.exists(vecnorm_path):
        alt_vecnorm = "vecnorm.pkl"
        if os.path.exists(alt_vecnorm):
            vecnorm_path = alt_vecnorm
    
    print("="*60)
    print("DIAGNOSTIC EXPERIMENT 3: Stochastic vs Deterministic Inference")
    print("="*60)
    print(f"\nModel: {model_path}")
    print(f"VecNormalize: {vecnorm_path}")
    print()
    
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
    print("Setting up environment...")
    env = DummyVecEnv([lambda: make_standing_curriculum_env(render_mode=None, config=config)])
    
    # Load VecNormalize
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"✓ Loaded VecNormalize")
    else:
        print(f"✗ VecNormalize not found at {vecnorm_path}")
        return
    
    # Load model
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
        print(f"✓ Loaded model")
    else:
        print(f"✗ Model not found at {model_path}")
        return
    
    # Check policy entropy
    try:
        if hasattr(model.policy, 'log_std'):
            log_std = model.policy.log_std.detach().cpu().numpy()
            std = np.exp(log_std)
            print(f"\nPolicy stochasticity:")
            print(f"  Action std: mean={std.mean():.4f}, max={std.max():.4f}, min={std.min():.4f}")
            if std.mean() < 0.01:
                print(f"  ⚠️  Very low action noise (mean std < 0.01)")
                print(f"      Policy is nearly deterministic - this might be the issue!")
            elif std.mean() > 0.3:
                print(f"  ⚠️  High action noise (mean std > 0.3)")
                print(f"      Policy still exploring significantly")
            else:
                print(f"  ✓ Moderate action noise (0.01 < std < 0.3)")
    except Exception as e:
        print(f"Could not extract policy std: {e}")
    
    print("\n" + "="*60)
    print("Running experiments (10 episodes each mode)...")
    print("="*60)
    
    # Test deterministic mode
    results_deterministic = test_inference_mode(model, env, deterministic=True, n_episodes=10)
    
    # Test stochastic mode
    results_stochastic = test_inference_mode(model, env, deterministic=False, n_episodes=10)
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    mean_det = np.mean(results_deterministic)
    mean_sto = np.mean(results_stochastic)
    
    improvement = mean_sto - mean_det
    improvement_pct = (improvement / mean_det) * 100 if mean_det > 0 else 0
    
    print(f"Deterministic: {mean_det:6.1f} ± {np.std(results_deterministic):5.1f} steps")
    print(f"Stochastic:    {mean_sto:6.1f} ± {np.std(results_stochastic):5.1f} steps")
    print(f"Difference:    {improvement:+6.1f} steps ({improvement_pct:+.1f}%)")
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if improvement > 100:
        print("✓ SIGNIFICANT IMPROVEMENT with stochastic actions")
        print("  → Issue #4 CONFIRMED: Agent relies on action noise for stability")
        print()
        print("RECOMMENDED FIXES:")
        print("  1. Use deterministic=False during inference (quick fix)")
        print("  2. Retrain with higher final entropy coefficient (0.02 instead of 0.01)")
        print("  3. Add explicit robustness training (domain randomization)")
    elif improvement > 50:
        print("⚙️  MODERATE IMPROVEMENT with stochastic actions")
        print("  → Issue #4 partially contributing to failure")
        print()
        print("RECOMMENDED FIXES:")
        print("  1. Try stochastic inference for better performance")
        print("  2. Consider other issues (action smoothing, VecNormalize)")
    elif improvement < -50:
        print("⚠️  WORSE PERFORMANCE with stochastic actions")
        print("  → Issue #4 is NOT the problem")
        print("  → Agent actually benefits from deterministic execution")
        print("  → Focus on other issues (action smoothing, VecNormalize)")
    else:
        print("≈ NO SIGNIFICANT DIFFERENCE between modes")
        print("  → Issue #4 is NOT the problem")
        print("  → Stochasticity is not a factor in the failure")
        print()
        print("NEXT STEPS:")
        print("  - Focus on Issues #1 (action smoothing) and #3 (VecNormalize)")
    
    # Statistical significance test (simple t-test)
    from scipy import stats
    try:
        t_stat, p_value = stats.ttest_ind(results_deterministic, results_stochastic)
        print(f"\nStatistical significance: p={p_value:.4f}")
        if p_value < 0.05:
            print("  ✓ Difference is statistically significant (p < 0.05)")
        else:
            print("  ✗ Difference is NOT statistically significant (p ≥ 0.05)")
    except:
        # scipy might not be installed
        pass
    
    env.close()


if __name__ == "__main__":
    main()

