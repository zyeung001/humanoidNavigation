# diagnose_transfer.py
"""
Diagnostic script to verify standing → walking transfer learning.

Checks:
1. Whether standing weights are being used during inference
2. Whether command features are processed correctly
3. Whether action distribution changed after transfer vs fresh init
4. VecNormalize statistics alignment
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environments.walking_env import make_walking_env
from src.environments.standing_env import make_standing_env


def diagnose_observation_spaces():
    """Compare observation spaces between standing and walking."""
    print("\n" + "="*60)
    print("DIAGNOSIS: Observation Space Comparison")
    print("="*60)
    
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.2,
        'max_commanded_speed': 1.0,
        'include_yaw_rate': True,
    }
    
    standing_env = make_standing_env(config=config)
    walking_env = make_walking_env(config=config)
    
    print(f"\nStanding observation space: {standing_env.observation_space.shape}")
    print(f"Walking observation space: {walking_env.observation_space.shape}")
    print(f"Dimension difference: {walking_env.observation_space.shape[0] - standing_env.observation_space.shape[0]}")
    
    # Get sample observations
    standing_obs, _ = standing_env.reset()
    walking_obs, _ = walking_env.reset()
    
    print(f"\nSample observation shapes:")
    print(f"  Standing: {standing_obs.shape}")
    print(f"  Walking: {walking_obs.shape}")
    
    # Analyze observation structure
    per_frame_standing = 371  # 365 + 6
    per_frame_walking = 374   # 365 + 6 + 3
    
    print(f"\nPer-frame breakdown:")
    print(f"  Standing: {per_frame_standing} (base=365, COM=6)")
    print(f"  Walking: {per_frame_walking} (base=365, COM=6, cmd=3)")
    print(f"  × 4 frames = {per_frame_standing * 4} vs {per_frame_walking * 4}")
    
    # Show command feature values in walking observation
    print(f"\nCommand features in walking observation (first frame):")
    cmd_start = 371  # After base + COM
    cmd_end = 374
    print(f"  Indices {cmd_start}:{cmd_end}: {walking_obs[cmd_start:cmd_end]}")
    
    # Show what those values represent
    print(f"\nExpected command values:")
    print(f"  commanded_vx: {walking_env.commanded_vx_world:.3f}")
    print(f"  commanded_vy: {walking_env.commanded_vy_world:.3f}")
    print(f"  commanded_yaw: {walking_env.commanded_yaw_rate:.3f}")
    
    standing_env.close()
    walking_env.close()
    
    return per_frame_standing * 4, per_frame_walking * 4


def diagnose_policy_weights(
    standing_model_path: str,
    walking_model_path: str = None,
):
    """Compare policy weights between standing and walking models."""
    print("\n" + "="*60)
    print("DIAGNOSIS: Policy Weight Comparison")
    print("="*60)
    
    if not os.path.exists(standing_model_path):
        print(f"✗ Standing model not found: {standing_model_path}")
        return
    
    standing_model = PPO.load(standing_model_path, device='cpu')
    print(f"\n✓ Loaded standing model")
    
    if walking_model_path and os.path.exists(walking_model_path):
        walking_model = PPO.load(walking_model_path, device='cpu')
        print(f"✓ Loaded walking model")
    else:
        print(f"⚠ Walking model not provided - creating fresh for comparison")
        # Create fresh walking model for comparison
        config = {
            'obs_history': 4,
            'obs_include_com': True,
            'obs_feature_norm': True,
            'include_yaw_rate': True,
        }
        walking_env = make_walking_env(config=config)
        vec_env = DummyVecEnv([lambda: walking_env])
        wrapped_env = VecNormalize(vec_env)
        walking_model = PPO('MlpPolicy', wrapped_env, device='cpu')
    
    standing_state = standing_model.policy.state_dict()
    walking_state = walking_model.policy.state_dict()
    
    print(f"\nLayer comparison:")
    print("-" * 80)
    
    for key in walking_state.keys():
        walking_shape = walking_state[key].shape
        if key in standing_state:
            standing_shape = standing_state[key].shape
            
            if standing_shape == walking_shape:
                # Check if weights are similar (transfer worked)
                standing_w = standing_state[key].numpy()
                walking_w = walking_state[key].numpy()
                similarity = np.corrcoef(standing_w.flatten(), walking_w.flatten())[0, 1]
                
                status = "✓ MATCH" if np.isclose(similarity, 1.0) else f"~ corr={similarity:.3f}"
                print(f"  {key}: {standing_shape} → {walking_shape} {status}")
            else:
                print(f"  {key}: {standing_shape} → {walking_shape} [DIMENSION MISMATCH]")
                
                # Show input layer dimension details
                if "weight" in key and len(walking_shape) == 2:
                    in_diff = walking_shape[1] - standing_shape[1]
                    out_diff = walking_shape[0] - standing_shape[0]
                    print(f"       Input dims: +{in_diff}, Output dims: +{out_diff}")
                    
                    # Analyze the new weight initialization
                    if in_diff > 0 and walking_model_path:
                        new_weights = walking_state[key][:, -in_diff:].numpy()
                        print(f"       New weight stats: mean={new_weights.mean():.4f}, "
                              f"std={new_weights.std():.4f}, range=[{new_weights.min():.4f}, {new_weights.max():.4f}]")
                        
                        # Compare to standing weights scale
                        old_weights = standing_state[key].numpy()
                        print(f"       Standing weight stats: mean={old_weights.mean():.4f}, "
                              f"std={old_weights.std():.4f}")
        else:
            print(f"  {key}: NEW {walking_shape}")


def diagnose_action_distribution(
    standing_model_path: str,
    walking_model_path: str = None,
):
    """Compare action distributions between models."""
    print("\n" + "="*60)
    print("DIAGNOSIS: Action Distribution Comparison")
    print("="*60)
    
    if not os.path.exists(standing_model_path):
        print(f"✗ Standing model not found: {standing_model_path}")
        return
    
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'include_yaw_rate': True,
        'max_commanded_speed': 0.3,
    }
    
    # Create environments
    walking_env = make_walking_env(config=config)
    vec_env = DummyVecEnv([lambda: walking_env])
    walking_vecnorm = VecNormalize(vec_env)
    
    # Load models
    standing_model = PPO.load(standing_model_path, device='cpu')
    
    # Collect actions from standing model on walking observation
    # (This simulates what happens with dimension mismatch)
    print("\nTesting action prediction...")
    
    obs = walking_vecnorm.reset()
    print(f"Walking observation shape: {obs.shape}")
    print(f"Standing expects: {standing_model.observation_space.shape}")
    
    # Truncate walking obs to standing dims for comparison
    standing_dim = standing_model.observation_space.shape[0]
    walking_dim = obs.shape[1]
    
    if walking_dim > standing_dim:
        truncated_obs = obs[:, :standing_dim]
        print(f"Truncated obs shape: {truncated_obs.shape}")
        
        # Get action from standing model
        with torch.no_grad():
            standing_action, _ = standing_model.predict(truncated_obs, deterministic=True)
        print(f"Standing model action (on truncated obs): mean={standing_action.mean():.4f}, "
              f"std={standing_action.std():.4f}")
    
    # Get actions from walking model (if provided)
    if walking_model_path and os.path.exists(walking_model_path):
        walking_model = PPO.load(walking_model_path, device='cpu')
        
        # Test on same observation
        with torch.no_grad():
            walking_action, _ = walking_model.predict(obs, deterministic=True)
        print(f"Walking model action: mean={walking_action.mean():.4f}, "
              f"std={walking_action.std():.4f}")
        
        # Compare with different command values
        print("\nCommand sensitivity test (walking model):")
        for vx in [0.0, 0.5, 1.0, 2.0]:
            walking_env.fixed_command = (vx, 0.0, 0.0)
            obs, _ = walking_env.reset()
            obs = walking_vecnorm.normalize_obs(obs)
            
            with torch.no_grad():
                action, _ = walking_model.predict(obs.reshape(1, -1), deterministic=True)
            print(f"  vx={vx:.1f}: action_mean={action.mean():.4f}, action_std={action.std():.4f}")
    
    walking_env.close()


def diagnose_vecnormalize(
    standing_vecnorm_path: str,
    walking_vecnorm_path: str = None,
):
    """Analyze VecNormalize statistics."""
    print("\n" + "="*60)
    print("DIAGNOSIS: VecNormalize Statistics")
    print("="*60)
    
    import pickle
    
    if os.path.exists(standing_vecnorm_path):
        with open(standing_vecnorm_path, 'rb') as f:
            standing_data = pickle.load(f)
        
        standing_mean = standing_data['obs_rms']['mean']
        standing_var = standing_data['obs_rms']['var']
        standing_count = standing_data['obs_rms']['count']
        
        print(f"\nStanding VecNormalize:")
        print(f"  Dimension: {len(standing_mean)}")
        print(f"  Sample count: {standing_count:,.0f}")
        print(f"  Mean range: [{standing_mean.min():.3f}, {standing_mean.max():.3f}]")
        print(f"  Var range: [{standing_var.min():.3f}, {standing_var.max():.3f}]")
        
        # Show first few dims
        print(f"  First 10 means: {standing_mean[:10]}")
        print(f"  First 10 vars: {standing_var[:10]}")
    else:
        print(f"✗ Standing VecNormalize not found: {standing_vecnorm_path}")
    
    if walking_vecnorm_path and os.path.exists(walking_vecnorm_path):
        with open(walking_vecnorm_path, 'rb') as f:
            walking_data = pickle.load(f)
        
        walking_mean = walking_data['obs_rms']['mean']
        walking_var = walking_data['obs_rms']['var']
        walking_count = walking_data['obs_rms']['count']
        
        print(f"\nWalking VecNormalize:")
        print(f"  Dimension: {len(walking_mean)}")
        print(f"  Sample count: {walking_count:,.0f}")
        print(f"  Mean range: [{walking_mean.min():.3f}, {walking_mean.max():.3f}]")
        print(f"  Var range: [{walking_var.min():.3f}, {walking_var.max():.3f}]")
        
        # Show command feature statistics
        per_frame = 374
        for i in range(4):
            cmd_start = i * per_frame + 371
            cmd_end = i * per_frame + 374
            if cmd_end <= len(walking_mean):
                print(f"  Frame {i} command mean: {walking_mean[cmd_start:cmd_end]}")
                print(f"  Frame {i} command var: {walking_var[cmd_start:cmd_end]}")
    else:
        print(f"\n⚠ Walking VecNormalize not found: {walking_vecnorm_path}")


def diagnose_command_processing():
    """Verify command features are processed correctly in observations."""
    print("\n" + "="*60)
    print("DIAGNOSIS: Command Feature Processing")
    print("="*60)
    
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'include_yaw_rate': True,
        'max_commanded_speed': 3.0,
    }
    
    walking_env = make_walking_env(config=config)
    
    # Test different command values
    print("\nTesting command feature encoding:")
    
    test_commands = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 1.0, 0.5),
    ]
    
    per_frame = 374
    
    for vx, vy, yaw in test_commands:
        walking_env.fixed_command = (vx, vy, yaw)
        obs, info = walking_env.reset()
        
        print(f"\nCommand: vx={vx:.1f}, vy={vy:.1f}, yaw={yaw:.1f}")
        
        # Check each frame's command features
        for frame in range(4):
            cmd_start = frame * per_frame + 371
            cmd_end = frame * per_frame + 374
            
            if cmd_end <= len(obs):
                cmd_features = obs[cmd_start:cmd_end]
                print(f"  Frame {frame}: {cmd_features}")
        
        # Also check via info
        print(f"  Info: commanded_vx={info.get('commanded_vx', 'N/A'):.3f}, "
              f"commanded_vy={info.get('commanded_vy', 'N/A'):.3f}")
    
    walking_env.close()


def run_inference_comparison(
    standing_model_path: str,
    walking_model_path: str,
    n_steps: int = 200,
):
    """Run inference and compare behavior."""
    print("\n" + "="*60)
    print("DIAGNOSIS: Inference Behavior Comparison")
    print("="*60)
    
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'include_yaw_rate': True,
        'max_commanded_speed': 1.0,
        'fixed_command': (0.5, 0.0, 0.0),  # Fixed forward command
    }
    
    walking_env = make_walking_env(config=config)
    vec_env = DummyVecEnv([lambda: walking_env])
    
    if not os.path.exists(walking_model_path):
        print(f"✗ Walking model not found: {walking_model_path}")
        return
    
    # Create fresh VecNormalize
    vecnorm_path = walking_model_path.replace('model.zip', 'vecnorm.pkl').replace('.zip', '_vecnorm.pkl')
    if os.path.exists(vecnorm_path):
        wrapped_env = VecNormalize.load(vecnorm_path, vec_env)
        print(f"✓ Loaded VecNormalize from {vecnorm_path}")
    else:
        wrapped_env = VecNormalize(vec_env)
        print("⚠ VecNormalize not found, using fresh (may cause issues)")
    
    walking_model = PPO.load(walking_model_path, device='cpu')
    
    print(f"\nRunning {n_steps} steps with fixed forward command (vx=0.5):")
    
    obs = wrapped_env.reset()
    rewards = []
    velocities = []
    heights = []
    
    for step in range(n_steps):
        action, _ = walking_model.predict(obs, deterministic=True)
        obs, reward, done, info = wrapped_env.step(action)
        
        rewards.append(reward[0])
        velocities.append(info[0].get('x_velocity', 0.0))
        heights.append(info[0].get('height', 0.0))
        
        if step % 50 == 0:
            print(f"  Step {step}: h={heights[-1]:.3f}, vx={velocities[-1]:.3f}, r={rewards[-1]:.1f}")
        
        if done.any():
            print(f"  Episode terminated at step {step}")
            break
    
    print(f"\nSummary:")
    print(f"  Mean reward: {np.mean(rewards):.2f}")
    print(f"  Mean velocity: {np.mean(velocities):.3f} m/s")
    print(f"  Mean height: {np.mean(heights):.3f} m")
    print(f"  Velocity tracking: Target=0.5, Actual={np.mean(velocities[-50:]):.3f}")
    
    walking_env.close()


def main():
    parser = argparse.ArgumentParser(description="Diagnose transfer learning issues")
    parser.add_argument('--standing-model', type=str, default='models/best_standing_model.zip',
                        help='Path to standing model')
    parser.add_argument('--walking-model', type=str, default='models/final_walking_model.zip',
                        help='Path to walking model')
    parser.add_argument('--standing-vecnorm', type=str, default='models/vecnorm.pkl',
                        help='Path to standing VecNormalize')
    parser.add_argument('--walking-vecnorm', type=str, default='models/vecnorm_walking.pkl',
                        help='Path to walking VecNormalize')
    parser.add_argument('--all', action='store_true', help='Run all diagnostics')
    parser.add_argument('--obs', action='store_true', help='Diagnose observation spaces')
    parser.add_argument('--weights', action='store_true', help='Diagnose policy weights')
    parser.add_argument('--actions', action='store_true', help='Diagnose action distributions')
    parser.add_argument('--vecnorm', action='store_true', help='Diagnose VecNormalize')
    parser.add_argument('--commands', action='store_true', help='Diagnose command processing')
    parser.add_argument('--inference', action='store_true', help='Run inference comparison')
    
    args = parser.parse_args()
    
    # Default to all if nothing specified
    run_all = args.all or not any([args.obs, args.weights, args.actions, 
                                    args.vecnorm, args.commands, args.inference])
    
    print("\n" + "="*60)
    print("TRANSFER LEARNING DIAGNOSTICS")
    print("="*60)
    print(f"Standing model: {args.standing_model}")
    print(f"Walking model: {args.walking_model}")
    
    if run_all or args.obs:
        diagnose_observation_spaces()
    
    if run_all or args.weights:
        diagnose_policy_weights(args.standing_model, args.walking_model)
    
    if run_all or args.actions:
        diagnose_action_distribution(args.standing_model, args.walking_model)
    
    if run_all or args.vecnorm:
        diagnose_vecnormalize(args.standing_vecnorm, args.walking_vecnorm)
    
    if run_all or args.commands:
        diagnose_command_processing()
    
    if args.inference:
        run_inference_comparison(args.standing_model, args.walking_model)
    
    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
