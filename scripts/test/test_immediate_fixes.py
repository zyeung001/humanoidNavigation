"""
Immediate fixes for humanoid standing inference issues - NO RETRAINING REQUIRED

Test the impact of:
1. Action smoothing warmup
2. Disabling VecNormalize
3. Height generalization

Run: python scripts/test/test_immediate_fixes.py
"""

import os
import sys
import numpy as np
import gymnasium as gym
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environments.standing_env import make_standing_env


# ============================================================================
# FIX #1: Action Smoothing Warmup Wrapper
# ============================================================================

class InferenceActionWarmup(gym.Wrapper):
    """
    Warm up action smoothing history during inference.
    
    Problem: Action smoothing uses prev_action which is reset to zeros.
    Solution: Take small stabilizing actions to build realistic action history.
    """
    def __init__(self, env, warmup_steps=10, warmup_noise=0.01):
        super().__init__(env)
        self.warmup_steps = warmup_steps
        self.warmup_noise = warmup_noise
        print(f"✓ InferenceActionWarmup enabled: {warmup_steps} steps")
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Warmup phase: take small random actions to build action history
        for i in range(self.warmup_steps):
            # Gradually increase noise (start from nearly zero)
            scale = (i + 1) / self.warmup_steps * self.warmup_noise
            action = self.env.action_space.sample() * scale
            obs, _, terminated, truncated, _ = self.env.step(action)
            
            # If warmup causes termination (shouldn't happen), restart
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        
        return obs, info


# ============================================================================
# FIX #2: VecNormalize Bypass
# ============================================================================

def create_env_without_vecnormalize(config):
    """Create environment without VecNormalize wrapper for testing"""
    env = make_standing_env(render_mode=None, config=config)
    vec_env = DummyVecEnv([lambda: env])
    print("✓ Environment created WITHOUT VecNormalize")
    return vec_env


def create_env_with_vecnormalize(vecnorm_path, config):
    """Create environment with VecNormalize (standard inference)"""
    env = make_standing_env(render_mode=None, config=config)
    vec_env = DummyVecEnv([lambda: env])
    
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"✓ Environment created WITH VecNormalize from {vecnorm_path}")
    else:
        print(f"✗ VecNormalize path not found: {vecnorm_path}")
        print("  Using raw environment instead")
    
    return vec_env


# ============================================================================
# FIX #3: Height Generalization Test
# ============================================================================

def test_height_generalization(model, config, heights=[0.6, 0.8, 1.0, 1.2, 1.35, 1.4]):
    """
    Test how well the model handles different initial heights.
    
    Expected: Model fails at h < 1.0 due to curriculum artifact.
    """
    print("\n" + "="*70)
    print("HEIGHT GENERALIZATION TEST")
    print("="*70)
    
    results = {}
    
    for target_height in heights:
        # Create fresh environment
        env = make_standing_env(render_mode=None, config=config)
        
        obs, info = env.reset()
        
        # MANUALLY set height (requires direct MuJoCo access)
        try:
            env.unwrapped.data.qpos[2] = target_height
            # Also need to reset velocity to be fair
            env.unwrapped.data.qvel[:] = 0.0
            
            # Get observation after manual height change
            obs = env._process_observation(env.unwrapped._get_obs())
        except Exception as e:
            print(f"  Warning: Could not set height directly: {e}")
            continue
        
        # Run episode
        steps_survived = 0
        total_reward = 0
        heights_during_episode = []
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            steps_survived += 1
            total_reward += reward
            heights_during_episode.append(info['height'])
            
            if terminated or truncated:
                break
        
        # Calculate success
        success = steps_survived >= 500 and not terminated
        avg_height = np.mean(heights_during_episode)
        
        results[target_height] = {
            'steps': steps_survived,
            'success': success,
            'reward': total_reward,
            'avg_height': avg_height
        }
        
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  Initial height {target_height:.2f}m: {status}")
        print(f"    Steps survived: {steps_survived}/500")
        print(f"    Avg height: {avg_height:.3f}m")
        print(f"    Total reward: {total_reward:.1f}")
        
        env.close()
    
    return results


# ============================================================================
# Main Test Functions
# ============================================================================

def run_episode(model, env, max_steps=500, verbose=False):
    """Run single episode and return detailed metrics"""
    obs = env.reset()
    
    episode_data = {
        'steps': 0,
        'total_reward': 0,
        'heights': [],
        'actions': [],
        'rewards': [],
        'terminated': False,
        'success': False
    }
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract info (handle vectorized env)
        if hasattr(terminated, '__len__'):
            terminated = terminated[0]
            truncated = truncated[0]
            reward = reward[0]
            info = info[0] if len(info) > 0 else {}
        
        episode_data['steps'] += 1
        episode_data['total_reward'] += reward
        episode_data['heights'].append(info.get('height', 0))
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        
        if terminated or truncated:
            episode_data['terminated'] = terminated
            break
    
    # Calculate success
    final_height = episode_data['heights'][-1] if episode_data['heights'] else 0
    episode_data['success'] = (episode_data['steps'] >= max_steps) and not episode_data['terminated']
    episode_data['final_height'] = final_height
    episode_data['avg_height'] = np.mean(episode_data['heights']) if episode_data['heights'] else 0
    episode_data['height_std'] = np.std(episode_data['heights']) if episode_data['heights'] else 0
    
    if verbose:
        status = "✓" if episode_data['success'] else "✗"
        print(f"  {status} Steps: {episode_data['steps']:3d}, "
              f"Height: {episode_data['avg_height']:.3f}±{episode_data['height_std']:.3f}m, "
              f"Reward: {episode_data['total_reward']:6.1f}")
    
    return episode_data


def test_configuration(name, model, env, n_episodes=10):
    """Test a specific configuration"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    results = []
    for ep in range(n_episodes):
        episode_data = run_episode(model, env, max_steps=500, verbose=True)
        results.append(episode_data)
    
    # Calculate aggregate metrics
    success_rate = sum(1 for r in results if r['success']) / len(results) * 100
    avg_steps = np.mean([r['steps'] for r in results])
    avg_height = np.mean([r['avg_height'] for r in results])
    avg_reward = np.mean([r['total_reward'] for r in results])
    
    # Count catastrophic failures (height < 0.85m)
    catastrophic = sum(1 for r in results if r['final_height'] < 0.85)
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {name}")
    print(f"{'='*70}")
    print(f"  Success Rate: {success_rate:.1f}% ({int(success_rate/10)}/{n_episodes} episodes)")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Avg Height: {avg_height:.3f}m")
    print(f"  Avg Reward: {avg_reward:.1f}")
    print(f"  Catastrophic Failures: {catastrophic}/{n_episodes}")
    print(f"{'='*70}\n")
    
    return {
        'name': name,
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_height': avg_height,
        'avg_reward': avg_reward,
        'catastrophic': catastrophic,
        'results': results
    }


# ============================================================================
# Main Test Suite
# ============================================================================

def main():
    print("\n" + "="*70)
    print("HUMANOID STANDING - IMMEDIATE FIX TESTS")
    print("="*70)
    
    # Paths (adjust as needed)
    model_path = "models/best_standing_model"  # Without .zip
    vecnorm_path = "vecnorm.pkl"
    
    # Check if files exist
    if not os.path.exists(f"{model_path}.zip"):
        print(f"✗ Model not found: {model_path}.zip")
        print(f"  Please provide the correct path to your trained model")
        return
    
    # Configuration (match training config)
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.2,
        'max_episode_steps': 5000,
    }
    
    # Load model (without env, we'll provide different envs)
    print(f"Loading model from: {model_path}.zip")
    
    all_results = {}
    
    # ========================================================================
    # TEST 1: BASELINE (Current Setup)
    # ========================================================================
    print("\n" + "#"*70)
    print("# TEST 1: BASELINE (Current Inference Setup)")
    print("#"*70)
    
    env_baseline = create_env_with_vecnormalize(vecnorm_path, config)
    model_baseline = PPO.load(f"{model_path}.zip", env=env_baseline)
    
    all_results['baseline'] = test_configuration(
        "BASELINE (VecNormalize + No Warmup)",
        model_baseline,
        env_baseline,
        n_episodes=10
    )
    
    env_baseline.close()
    
    # ========================================================================
    # TEST 2: WITH ACTION WARMUP
    # ========================================================================
    print("\n" + "#"*70)
    print("# TEST 2: ACTION SMOOTHING WARMUP FIX")
    print("#"*70)
    
    # Create env with warmup wrapper
    base_env = make_standing_env(render_mode=None, config=config)
    warmup_env = InferenceActionWarmup(base_env, warmup_steps=10)
    vec_warmup = DummyVecEnv([lambda: warmup_env])
    
    if os.path.exists(vecnorm_path):
        vec_warmup = VecNormalize.load(vecnorm_path, vec_warmup)
        vec_warmup.training = False
        vec_warmup.norm_reward = False
    
    model_warmup = PPO.load(f"{model_path}.zip", env=vec_warmup)
    
    all_results['warmup'] = test_configuration(
        "WITH ACTION WARMUP (10 steps)",
        model_warmup,
        vec_warmup,
        n_episodes=10
    )
    
    vec_warmup.close()
    
    # ========================================================================
    # TEST 3: WITHOUT VecNormalize
    # ========================================================================
    print("\n" + "#"*70)
    print("# TEST 3: WITHOUT VECNORMALIZE")
    print("#"*70)
    
    env_no_norm = create_env_without_vecnormalize(config)
    model_no_norm = PPO.load(f"{model_path}.zip", env=env_no_norm)
    
    all_results['no_vecnorm'] = test_configuration(
        "WITHOUT VecNormalize",
        model_no_norm,
        env_no_norm,
        n_episodes=10
    )
    
    env_no_norm.close()
    
    # ========================================================================
    # TEST 4: BOTH FIXES COMBINED
    # ========================================================================
    print("\n" + "#"*70)
    print("# TEST 4: BOTH FIXES COMBINED")
    print("#"*70)
    
    base_env_combined = make_standing_env(render_mode=None, config=config)
    warmup_env_combined = InferenceActionWarmup(base_env_combined, warmup_steps=10)
    vec_combined = DummyVecEnv([lambda: warmup_env_combined])
    # No VecNormalize
    
    model_combined = PPO.load(f"{model_path}.zip", env=vec_combined)
    
    all_results['combined'] = test_configuration(
        "WARMUP + NO VecNormalize",
        model_combined,
        vec_combined,
        n_episodes=10
    )
    
    vec_combined.close()
    
    # ========================================================================
    # TEST 5: HEIGHT GENERALIZATION
    # ========================================================================
    print("\n" + "#"*70)
    print("# TEST 5: HEIGHT GENERALIZATION (Curriculum Artifact Test)")
    print("#"*70)
    
    # Use baseline model
    env_height_test = create_env_with_vecnormalize(vecnorm_path, config)
    model_height_test = PPO.load(f"{model_path}.zip", env=env_height_test)
    
    # Note: This test might not work perfectly due to VecNormalize, but will show trend
    height_results = test_height_generalization(
        model_height_test,
        config,
        heights=[0.6, 0.8, 1.0, 1.2, 1.35, 1.4]
    )
    
    all_results['height_generalization'] = height_results
    env_height_test.close()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    comparison = []
    for key in ['baseline', 'warmup', 'no_vecnorm', 'combined']:
        if key in all_results:
            r = all_results[key]
            comparison.append({
                'Config': r['name'],
                'Success %': f"{r['success_rate']:.1f}%",
                'Avg Steps': f"{r['avg_steps']:.0f}",
                'Catastrophic': f"{r['catastrophic']}/10"
            })
    
    # Print comparison table
    print(f"{'Config':<35} {'Success %':<12} {'Avg Steps':<12} {'Catastrophic':<15}")
    print("-" * 70)
    for row in comparison:
        print(f"{row['Config']:<35} {row['Success %']:<12} {row['Avg Steps']:<12} {row['Catastrophic']:<15}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    baseline_success = all_results['baseline']['success_rate']
    warmup_success = all_results['warmup']['success_rate']
    no_norm_success = all_results['no_vecnorm']['success_rate']
    combined_success = all_results['combined']['success_rate']
    
    improvements = []
    
    if warmup_success > baseline_success + 10:
        improvements.append("✓ Action warmup HELPS (+{:.0f}% success)".format(warmup_success - baseline_success))
    
    if no_norm_success > baseline_success + 10:
        improvements.append("✓ Removing VecNormalize HELPS (+{:.0f}% success)".format(no_norm_success - baseline_success))
    
    if combined_success > baseline_success + 15:
        improvements.append("✓ Combined fixes give BEST results (+{:.0f}% success)".format(combined_success - baseline_success))
    
    if improvements:
        print("\nImmediate fixes that help:")
        for imp in improvements:
            print(f"  {imp}")
        print("\nRECOMMENDATION: Use the best-performing configuration for inference.")
    else:
        print("\nImmediate fixes show limited improvement.")
        print("RECOMMENDATION: Retrain with suggested fixes (see debugging_analysis.md)")
    
    # Curriculum artifact detection
    if 'height_generalization' in all_results:
        print("\nCurriculum Artifact Analysis:")
        hr = all_results['height_generalization']
        low_height_failures = sum(1 for h, r in hr.items() if h < 1.0 and not r['success'])
        high_height_success = sum(1 for h, r in hr.items() if h >= 1.2 and r['success'])
        
        if low_height_failures >= 2:
            print("  ✗ Model struggles at heights < 1.0m (curriculum artifact detected)")
            print("  → RETRAIN with random height initialization or recovery stage")
        
        if high_height_success >= 2:
            print("  ✓ Model performs well at trained heights (>1.2m)")
    
    print("\n" + "="*70)
    print("See debugging_analysis.md for detailed root cause analysis and training fixes")
    print("="*70)


if __name__ == "__main__":
    main()

