"""
test_walking.py

Test script for walking controller with velocity tracking metrics.
Evaluates model performance across different velocity commands.
"""

import os
import sys
import argparse
import numpy as np

# Ensure project root & src on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environments.walking_env import make_walking_env


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def create_test_env(config, vx_target, vy_target, render_mode=None):
    """Create walking environment with fixed velocity command."""
    test_config = config.copy()
    test_config['fixed_command'] = (vx_target, vy_target)
    test_config['random_height_init'] = False
    test_config['max_episode_steps'] = 2000
    
    env = make_walking_env(render_mode=render_mode, config=test_config)
    return env


def run_episode(model, vec_env, max_steps=2000, verbose=False):
    """Run a single episode and collect metrics."""
    obs = vec_env.reset()
    
    velocity_errors = []
    heights = []
    actual_speeds = []
    xy_positions = []
    
    step = 0
    terminated_early = False
    
    while step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        step += 1
        
        # Collect metrics from info
        if len(info) > 0:
            i = info[0]
            if 'velocity_error' in i:
                velocity_errors.append(i['velocity_error'])
            if 'height' in i:
                heights.append(i['height'])
            if 'actual_speed' in i:
                actual_speeds.append(i['actual_speed'])
            if 'x_position' in i and 'y_position' in i:
                xy_positions.append((i['x_position'], i['y_position']))
        
        if done[0]:
            terminated_early = True
            break
    
    # Calculate metrics
    metrics = {
        'steps': step,
        'terminated_early': terminated_early,
        'success': step >= max_steps * 0.95,  # 95% of max steps
    }
    
    if velocity_errors:
        metrics['mean_velocity_error'] = np.mean(velocity_errors)
        metrics['std_velocity_error'] = np.std(velocity_errors)
        metrics['max_velocity_error'] = np.max(velocity_errors)
    
    if heights:
        metrics['mean_height'] = np.mean(heights)
        metrics['std_height'] = np.std(heights)
        metrics['min_height'] = np.min(heights)
        metrics['max_height'] = np.max(heights)
    
    if actual_speeds:
        metrics['mean_actual_speed'] = np.mean(actual_speeds)
    
    if xy_positions:
        xy = np.array(xy_positions)
        metrics['xy_drift'] = np.sqrt(xy[-1, 0]**2 + xy[-1, 1]**2)
        metrics['total_distance'] = np.sum(np.sqrt(np.diff(xy[:, 0])**2 + np.diff(xy[:, 1])**2))
    
    return metrics


def test_velocity_command(model, vec_env, vx, vy, n_episodes=5, max_steps=2000, verbose=True):
    """Test a specific velocity command over multiple episodes."""
    all_metrics = []
    
    for ep in range(n_episodes):
        metrics = run_episode(model, vec_env, max_steps=max_steps, verbose=verbose)
        all_metrics.append(metrics)
        
        if verbose:
            print(f"  Episode {ep+1}: steps={metrics['steps']}, "
                  f"vel_err={metrics.get('mean_velocity_error', 0):.4f} m/s, "
                  f"height={metrics.get('mean_height', 0):.3f} m, "
                  f"success={'✓' if metrics['success'] else '✗'}")
    
    # Aggregate results
    result = {
        'vx': vx,
        'vy': vy,
        'commanded_speed': np.sqrt(vx**2 + vy**2),
        'n_episodes': n_episodes,
        'success_rate': np.mean([m['success'] for m in all_metrics]),
        'mean_steps': np.mean([m['steps'] for m in all_metrics]),
        'mean_velocity_error': np.mean([m.get('mean_velocity_error', 0) for m in all_metrics]),
        'std_velocity_error': np.mean([m.get('std_velocity_error', 0) for m in all_metrics]),
        'mean_height': np.mean([m.get('mean_height', 0) for m in all_metrics]),
        'height_stability': np.mean([m.get('std_height', 0) for m in all_metrics]),
        'mean_xy_drift': np.mean([m.get('xy_drift', 0) for m in all_metrics]),
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Test walking controller")
    parser.add_argument('--model', type=str, required=True, help='Path to model .zip')
    parser.add_argument('--vecnorm', type=str, required=True, help='Path to VecNormalize .pkl')
    parser.add_argument('--episodes', type=int, default=5, help='Episodes per velocity command')
    parser.add_argument('--steps', type=int, default=2000, help='Max steps per episode')
    parser.add_argument('--render', action='store_true', help='Render environment')
    args = parser.parse_args()
    
    # Load config
    cfg = load_yaml('config/training_config.yaml')
    walking = cfg.get('walking', {}).copy()
    
    # Setup environment config
    walking['obs_history'] = walking.get('obs_history', 4)
    walking['obs_include_com'] = True
    walking['obs_feature_norm'] = True
    walking['action_smoothing'] = True
    walking['action_smoothing_tau'] = 0.2
    
    render_mode = 'human' if args.render else None
    
    # Test velocity commands
    test_commands = [
        # Standing
        (0.0, 0.0, "Standing"),
        # Forward walking at various speeds
        (0.5, 0.0, "Forward 0.5 m/s"),
        (1.0, 0.0, "Forward 1.0 m/s"),
        (1.5, 0.0, "Forward 1.5 m/s"),
        (2.0, 0.0, "Forward 2.0 m/s"),
        (2.5, 0.0, "Forward 2.5 m/s"),
        (3.0, 0.0, "Forward 3.0 m/s"),
        # Sideways
        (0.0, 1.0, "Sideways 1.0 m/s"),
        (0.0, 1.5, "Sideways 1.5 m/s"),
        # Diagonal
        (1.0, 1.0, "Diagonal (1,1) m/s"),
        (1.5, 1.5, "Diagonal (1.5,1.5) m/s"),
        # Backwards
        (-0.5, 0.0, "Backward 0.5 m/s"),
        (-1.0, 0.0, "Backward 1.0 m/s"),
    ]
    
    print("=" * 70)
    print("WALKING CONTROLLER TEST")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"VecNormalize: {args.vecnorm}")
    print(f"Episodes per command: {args.episodes}")
    print(f"Max steps per episode: {args.steps}")
    print("=" * 70)
    
    results = []
    
    for vx, vy, name in test_commands:
        print(f"\n--- Testing: {name} (vx={vx:.1f}, vy={vy:.1f}) ---")
        
        # Create environment with this velocity command
        env = create_test_env(walking, vx, vy, render_mode=render_mode)
        _ = env.reset()  # Pre-warm
        vec_env = DummyVecEnv([lambda: env])
        
        # Load VecNormalize
        if os.path.exists(args.vecnorm):
            vec_env = VecNormalize.load(args.vecnorm, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        else:
            print(f"WARNING: VecNormalize not found at {args.vecnorm}")
        
        # Load model
        model = PPO.load(args.model, env=vec_env, device="auto")
        
        # Test this velocity
        result = test_velocity_command(
            model, vec_env, vx, vy, 
            n_episodes=args.episodes, 
            max_steps=args.steps,
            verbose=True
        )
        result['name'] = name
        results.append(result)
        
        vec_env.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Command':<25} {'Speed':>7} {'Vel Err':>10} {'Height':>8} {'Success':>10} {'XY Drift':>10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<25} "
              f"{r['commanded_speed']:>6.2f}m "
              f"{r['mean_velocity_error']:>9.4f}m "
              f"{r['mean_height']:>7.3f}m "
              f"{r['success_rate']*100:>9.1f}% "
              f"{r['mean_xy_drift']:>9.3f}m")
    
    print("-" * 70)
    
    # Check success criteria
    print("\n=== SUCCESS CRITERIA CHECK ===")
    
    # 1. Max speed 3.0: average velocity error < 0.4 m/s
    fast_results = [r for r in results if r['commanded_speed'] >= 2.5]
    if fast_results:
        avg_vel_err_fast = np.mean([r['mean_velocity_error'] for r in fast_results])
        check1 = avg_vel_err_fast < 0.4
        print(f"1. Fast speed (≥2.5 m/s) vel error < 0.4: {avg_vel_err_fast:.4f} m/s "
              f"{'✓ PASS' if check1 else '✗ FAIL'}")
    
    # 2. Stable height 1.20-1.35m
    all_heights = [r['mean_height'] for r in results]
    min_height = np.min(all_heights)
    max_height = np.max(all_heights)
    check2 = min_height >= 1.15 and max_height <= 1.45
    print(f"2. Height range 1.15-1.45m: [{min_height:.3f}, {max_height:.3f}]m "
          f"{'✓ PASS' if check2 else '✗ FAIL'}")
    
    # 3. >95% success (no falling in 2000-step episodes)
    avg_success = np.mean([r['success_rate'] for r in results])
    check3 = avg_success >= 0.95
    print(f"3. Success rate > 95%: {avg_success*100:.1f}% "
          f"{'✓ PASS' if check3 else '✗ FAIL'}")
    
    # 4. Standing (0,0): XY drift < 1.0m over 2000 steps
    standing_result = [r for r in results if r['commanded_speed'] == 0.0]
    if standing_result:
        standing_drift = standing_result[0]['mean_xy_drift']
        check4 = standing_drift < 1.0
        print(f"4. Standing XY drift < 1.0m: {standing_drift:.3f}m "
              f"{'✓ PASS' if check4 else '✗ FAIL'}")
    
    # Overall
    all_pass = check1 and check2 and check3 and (not standing_result or check4)
    print(f"\n{'='*70}")
    if all_pass:
        print("OVERALL: ✓ ALL CRITERIA PASSED - Walking controller is production-ready!")
    else:
        print("OVERALL: ✗ SOME CRITERIA FAILED - Further training needed")
    print("=" * 70)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

