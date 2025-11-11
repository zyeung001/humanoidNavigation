"""
Diagnostic Experiment 2: Verify VecNormalize Integrity

This script checks if VecNormalize statistics match the actual observation dimension.
"""

import os
import sys
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environments.standing_curriculum import make_standing_curriculum_env


def test_vecnormalize_integrity(vecnorm_path):
    """Check VecNormalize dimension consistency"""
    
    print("="*60)
    print("DIAGNOSTIC EXPERIMENT 2: VecNormalize Integrity Check")
    print("="*60)
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
    
    print("Step 1: Create base environment")
    base_env = make_standing_curriculum_env(render_mode=None, config=config)
    obs_raw, _ = base_env.reset()
    print(f"✓ Raw observation dimension: {obs_raw.shape[0]}")
    print(f"  Environment observation space: {base_env.observation_space.shape}")
    base_env.close()
    
    print("\nStep 2: Wrap in VecEnv")
    vec_env = DummyVecEnv([lambda: make_standing_curriculum_env(render_mode=None, config=config)])
    obs_vec = vec_env.reset()
    print(f"✓ VecEnv observation shape: {obs_vec.shape}")
    print(f"  VecEnv observation space: {vec_env.observation_space.shape}")
    
    print("\nStep 3: Load VecNormalize statistics")
    if not os.path.exists(vecnorm_path):
        print(f"✗ VecNormalize file not found: {vecnorm_path}")
        vec_env.close()
        return
    
    try:
        vec_norm = VecNormalize.load(vecnorm_path, vec_env)
        print(f"✓ Loaded VecNormalize from {vecnorm_path}")
    except Exception as e:
        print(f"✗ Failed to load VecNormalize: {e}")
        vec_env.close()
        return
    
    print("\nStep 4: Check dimension consistency")
    print(f"VecNormalize obs_rms.mean shape: {vec_norm.obs_rms.mean.shape}")
    print(f"VecNormalize obs_rms.var shape: {vec_norm.obs_rms.var.shape}")
    print(f"VecEnv observation space shape: {vec_norm.observation_space.shape}")
    
    expected_dim = vec_norm.observation_space.shape[0]
    actual_dim = vec_norm.obs_rms.mean.shape[0]
    
    print("\n" + "="*60)
    if actual_dim != expected_dim:
        print("⚠️  CRITICAL: DIMENSION MISMATCH DETECTED!")
        print("="*60)
        print(f"  Expected dimension: {expected_dim}")
        print(f"  Actual VecNormalize buffer: {actual_dim}")
        print(f"  Difference: {actual_dim - expected_dim}")
        print()
        print("IMPLICATIONS:")
        print("  - Observations are being incorrectly normalized")
        print("  - This will corrupt the policy's inputs during inference")
        print("  - Explains why agent behavior differs from training")
        print()
        print("FIXES:")
        print("  Option A (Quick Fix for Inference):")
        print("    - Pad/truncate VecNormalize buffers to match expected dimension")
        print("    - See docs/expert_diagnosis.md Issue #3 for code")
        print("  Option B (Proper Fix):")
        print("    - Retrain with correct dimension from the start")
        print("    - This requires 1-2M timesteps")
        
        mismatch_detected = True
    else:
        print("✓ PASS: Dimensions match correctly")
        print("="*60)
        print(f"  Observation dimension: {expected_dim}")
        print(f"  VecNormalize buffer: {actual_dim}")
        print()
        print("VecNormalize integrity is good. Issue #3 is NOT the problem.")
        mismatch_detected = False
    
    print("\nStep 5: Test normalization range")
    obs_normalized = vec_norm.reset()
    print(f"Normalized observation stats:")
    print(f"  Shape: {obs_normalized.shape}")
    print(f"  Range: [{obs_normalized.min():.2f}, {obs_normalized.max():.2f}]")
    print(f"  Mean: {obs_normalized.mean():.2f}")
    print(f"  Std: {obs_normalized.std():.2f}")
    print(f"  Expected clip range: [-10.0, 10.0]")
    
    if obs_normalized.min() < -15.0 or obs_normalized.max() > 15.0:
        print("⚠️  Warning: Observations outside expected clip range!")
        print("   This suggests VecNormalize stats may be corrupted.")
    else:
        print("✓ Normalization range looks healthy")
    
    print("\nStep 6: Check for NaN or Inf values")
    if np.any(np.isnan(vec_norm.obs_rms.mean)) or np.any(np.isnan(vec_norm.obs_rms.var)):
        print("✗ CRITICAL: NaN values detected in VecNormalize statistics!")
    elif np.any(np.isinf(vec_norm.obs_rms.mean)) or np.any(np.isinf(vec_norm.obs_rms.var)):
        print("✗ CRITICAL: Inf values detected in VecNormalize statistics!")
    else:
        print("✓ No NaN/Inf values detected")
    
    print("\nStep 7: Analyze running statistics")
    print(f"First 5 observation means: {vec_norm.obs_rms.mean[:5]}")
    print(f"First 5 observation vars: {vec_norm.obs_rms.var[:5]}")
    print(f"Last 5 observation means: {vec_norm.obs_rms.mean[-5:]}")
    print(f"Last 5 observation vars: {vec_norm.obs_rms.var[-5:]}")
    
    # Check if variance is suspiciously low (indicates frozen statistics)
    if np.mean(vec_norm.obs_rms.var) < 0.1:
        print("⚠️  Warning: Very low variance detected")
        print("   VecNormalize statistics may not have been updated during training")
    elif np.mean(vec_norm.obs_rms.var) > 100:
        print("⚠️  Warning: Very high variance detected")
        print("   Observations may not have been normalized properly during training")
    else:
        print("✓ Variance levels look reasonable")
    
    vec_norm.close()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if mismatch_detected:
        print("❌ VecNormalize dimension mismatch detected (Issue #3 confirmed)")
        print("   Priority: HIGH - This is likely a major contributor to failure")
    else:
        print("✓ VecNormalize integrity looks good")
        print("  Issue #3 is NOT the root cause")
    print()


def main():
    # Default paths
    vecnorm_path = "models/saved_models/vecnorm_standing.pkl"
    
    # Check alternative paths
    if not os.path.exists(vecnorm_path):
        alt_paths = [
            "vecnorm.pkl",
            "models/saved_models/vecnorm.pkl",
            "data/vecnorm_standing.pkl"
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                vecnorm_path = alt
                break
    
    if not os.path.exists(vecnorm_path):
        print(f"✗ VecNormalize file not found at any of:")
        print(f"  - models/saved_models/vecnorm_standing.pkl")
        print(f"  - vecnorm.pkl")
        print(f"  - models/saved_models/vecnorm.pkl")
        print(f"  - data/vecnorm_standing.pkl")
        print()
        print("Please provide the correct path to your VecNormalize .pkl file")
        return
    
    print(f"Using VecNormalize file: {vecnorm_path}\n")
    test_vecnormalize_integrity(vecnorm_path)


if __name__ == "__main__":
    main()

