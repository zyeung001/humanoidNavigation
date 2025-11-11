"""
FIX #3: VecNormalize Dimension Mismatch (Quick Inference Fix)

This script provides a wrapper to fix VecNormalize dimension mismatches
without retraining. Use this for inference only.

For a proper fix, you need to retrain with correct dimensions.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


def fix_vecnormalize_dimension(vec_env):
    """
    Fix VecNormalize dimension mismatch by padding/truncating statistics.
    
    WARNING: This is a band-aid fix for inference only!
    For proper training, you need to retrain with correct dimensions.
    
    Args:
        vec_env: VecNormalize wrapped environment
    
    Returns:
        Fixed vec_env (modified in-place)
    """
    from stable_baselines3.common.vec_env import VecNormalize
    
    if not isinstance(vec_env, VecNormalize):
        print("Environment is not VecNormalize wrapped, no fix needed")
        return vec_env
    
    expected_dim = vec_env.observation_space.shape[0]
    actual_dim = vec_env.obs_rms.mean.shape[0]
    
    if expected_dim == actual_dim:
        print("✓ VecNormalize dimensions already match, no fix needed")
        return vec_env
    
    print(f"⚠️  Fixing VecNormalize dimension mismatch:")
    print(f"   Expected: {expected_dim} dims")
    print(f"   Actual buffer: {actual_dim} dims")
    
    if expected_dim > actual_dim:
        # Pad with neutral normalization values
        pad_size = expected_dim - actual_dim
        print(f"   Padding with {pad_size} neutral dimensions")
        
        vec_env.obs_rms.mean = np.concatenate([
            vec_env.obs_rms.mean,
            np.zeros(pad_size, dtype=np.float32)
        ])
        vec_env.obs_rms.var = np.concatenate([
            vec_env.obs_rms.var,
            np.ones(pad_size, dtype=np.float32)
        ])
    else:
        # Truncate to expected size
        print(f"   Truncating to {expected_dim} dimensions")
        vec_env.obs_rms.mean = vec_env.obs_rms.mean[:expected_dim]
        vec_env.obs_rms.var = vec_env.obs_rms.var[:expected_dim]
    
    print(f"✓ VecNormalize dimensions fixed")
    print(f"   New buffer shape: {vec_env.obs_rms.mean.shape}")
    
    return vec_env


def load_and_fix_vecnormalize(vecnorm_path, base_vec_env):
    """
    Load VecNormalize and automatically fix dimension mismatches.
    
    Args:
        vecnorm_path: Path to .pkl file
        base_vec_env: Base VecEnv to wrap
    
    Returns:
        Fixed VecNormalize environment
    """
    from stable_baselines3.common.vec_env import VecNormalize
    
    print(f"Loading VecNormalize from: {vecnorm_path}")
    
    try:
        vec_env = VecNormalize.load(vecnorm_path, base_vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print("✓ VecNormalize loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load VecNormalize: {e}")
        raise
    
    # Apply dimension fix
    vec_env = fix_vecnormalize_dimension(vec_env)
    
    return vec_env


def save_fixed_vecnormalize(vec_env, output_path):
    """
    Save the fixed VecNormalize for future use.
    
    Args:
        vec_env: Fixed VecNormalize environment
        output_path: Where to save the fixed .pkl file
    """
    from stable_baselines3.common.vec_env import VecNormalize
    
    if not isinstance(vec_env, VecNormalize):
        print("Environment is not VecNormalize, cannot save")
        return False
    
    try:
        vec_env.save(output_path)
        print(f"✓ Fixed VecNormalize saved to: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to save VecNormalize: {e}")
        return False


def main():
    """Interactive tool to fix VecNormalize files"""
    import argparse
    parser = argparse.ArgumentParser(description="Fix VecNormalize dimension mismatches")
    parser.add_argument('--vecnorm', type=str, required=True, help='Path to VecNormalize .pkl file')
    parser.add_argument('--output', type=str, help='Output path for fixed file (default: overwrites input)')
    parser.add_argument('--task', type=str, default='standing', choices=['standing'], help='Task type')
    args = parser.parse_args()
    
    print("="*60)
    print("VecNormalize Dimension Fix Tool")
    print("="*60)
    print()
    
    # Create environment to get expected dimensions
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.environments.standing_curriculum import make_standing_curriculum_env
    
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.2,
        'curriculum_start_stage': 3,
        'curriculum_max_stage': 3,
    }
    
    print("Creating reference environment...")
    base_env = DummyVecEnv([lambda: make_standing_curriculum_env(render_mode=None, config=config)])
    print(f"✓ Reference observation space: {base_env.observation_space.shape}")
    
    # Load and fix
    vec_env = load_and_fix_vecnormalize(args.vecnorm, base_env)
    
    # Save fixed version
    output_path = args.output or args.vecnorm
    if output_path == args.vecnorm:
        # Create backup
        import shutil
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = args.vecnorm.replace('.pkl', f'_backup_{timestamp}.pkl')
        shutil.copy2(args.vecnorm, backup_path)
        print(f"✓ Backup created: {backup_path}")
    
    save_fixed_vecnormalize(vec_env, output_path)
    
    print()
    print("="*60)
    print("Fix applied successfully!")
    print("="*60)
    print()
    print("NEXT STEPS:")
    print("  1. Use the fixed VecNormalize file for inference")
    print("  2. Test with: python scripts/diagnostics/test_vecnormalize_integrity.py")
    print("  3. If still having issues, you may need to retrain from scratch")
    print()
    print("IMPORTANT:")
    print("  This is a TEMPORARY FIX for inference only!")
    print("  For production, retrain with correct dimensions from the start.")
    
    vec_env.close()


if __name__ == "__main__":
    main()

