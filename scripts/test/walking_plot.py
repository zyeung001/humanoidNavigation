"""
walking_plot.py

Plotting utilities for walking controller analysis.
Visualizes velocity tracking, commanded vs achieved speed, direction error, etc.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Ensure project root & src on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environments.walking_env import make_walking_env


def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def collect_episode_data(model, vec_env, max_steps=2000):
    """Run an episode and collect detailed tracking data."""
    obs = vec_env.reset()
    
    data = {
        'commanded_vx': [],
        'commanded_vy': [],
        'actual_vx': [],
        'actual_vy': [],
        'velocity_error': [],
        'height': [],
        'x_position': [],
        'y_position': [],
        'step': [],
    }
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        if len(info) > 0:
            i = info[0]
            data['commanded_vx'].append(i.get('commanded_vx', 0))
            data['commanded_vy'].append(i.get('commanded_vy', 0))
            data['actual_vx'].append(i.get('x_velocity', 0))
            data['actual_vy'].append(i.get('y_velocity', 0))
            data['velocity_error'].append(i.get('velocity_error', 0))
            data['height'].append(i.get('height', 0))
            data['x_position'].append(i.get('x_position', 0))
            data['y_position'].append(i.get('y_position', 0))
            data['step'].append(step)
        
        if done[0]:
            break
    
    return {k: np.array(v) for k, v in data.items()}


def plot_velocity_tracking(data, title="Velocity Tracking", save_path=None):
    """Plot velocity tracking over time."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    steps = data['step']
    
    # 1. Commanded vs Actual Vx
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, data['commanded_vx'], 'b--', label='Commanded Vx', linewidth=2)
    ax1.plot(steps, data['actual_vx'], 'b-', label='Actual Vx', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('X-Velocity Tracking')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Commanded vs Actual Vy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, data['commanded_vy'], 'r--', label='Commanded Vy', linewidth=2)
    ax2.plot(steps, data['actual_vy'], 'r-', label='Actual Vy', alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Y-Velocity Tracking')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Velocity Error over time
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, data['velocity_error'], 'g-', linewidth=1)
    ax3.axhline(y=0.4, color='orange', linestyle='--', label='Target threshold (0.4 m/s)')
    ax3.fill_between(steps, data['velocity_error'], alpha=0.3, color='green')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Velocity Error (m/s)')
    ax3.set_title(f"Velocity Error (mean={np.mean(data['velocity_error']):.4f} m/s)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Height over time
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(steps, data['height'], 'purple', linewidth=1)
    ax4.axhline(y=1.40, color='green', linestyle='--', label='Target height (1.40m)')
    ax4.axhline(y=1.20, color='orange', linestyle=':', label='Min acceptable (1.20m)')
    ax4.axhline(y=1.35, color='orange', linestyle=':', label='Max acceptable (1.35m)')
    ax4.fill_between(steps, 1.20, 1.35, alpha=0.1, color='green')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Height (m)')
    ax4.set_title(f"Height (mean={np.mean(data['height']):.3f}m, std={np.std(data['height']):.4f}m)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. XY Trajectory
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(data['x_position'], data['y_position'], 'b-', linewidth=1, alpha=0.7)
    ax5.scatter([data['x_position'][0]], [data['y_position'][0]], 
                color='green', s=100, zorder=5, label='Start')
    ax5.scatter([data['x_position'][-1]], [data['y_position'][-1]], 
                color='red', s=100, zorder=5, label='End')
    
    # Draw commanded direction arrow
    cmd_vx = data['commanded_vx'][0]
    cmd_vy = data['commanded_vy'][0]
    cmd_speed = np.sqrt(cmd_vx**2 + cmd_vy**2)
    if cmd_speed > 0.1:
        scale = 2.0
        ax5.arrow(0, 0, cmd_vx * scale, cmd_vy * scale, 
                  head_width=0.3, head_length=0.2, fc='orange', ec='orange',
                  label=f'Command ({cmd_vx:.1f}, {cmd_vy:.1f})')
    
    ax5.set_xlabel('X Position (m)')
    ax5.set_ylabel('Y Position (m)')
    ax5.set_title('XY Trajectory')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # 6. Speed comparison
    ax6 = fig.add_subplot(gs[2, 1])
    commanded_speed = np.sqrt(data['commanded_vx']**2 + data['commanded_vy']**2)
    actual_speed = np.sqrt(data['actual_vx']**2 + data['actual_vy']**2)
    ax6.plot(steps, commanded_speed, 'b--', label='Commanded Speed', linewidth=2)
    ax6.plot(steps, actual_speed, 'b-', label='Actual Speed', alpha=0.7)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Speed (m/s)')
    ax6.set_title(f"Speed Tracking (commanded={np.mean(commanded_speed):.2f}, actual={np.mean(actual_speed):.2f} m/s)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_multi_velocity_comparison(model, vec_env_fn, velocities, max_steps=2000, save_path=None):
    """Plot comparison of velocity tracking across different commanded velocities."""
    fig, axes = plt.subplots(len(velocities), 3, figsize=(18, 4 * len(velocities)))
    
    for idx, (vx, vy, name) in enumerate(velocities):
        # Create environment with this velocity
        vec_env = vec_env_fn(vx, vy)
        
        # Collect data
        data = collect_episode_data(model, vec_env, max_steps)
        vec_env.close()
        
        steps = data['step']
        
        # Velocity error
        ax1 = axes[idx, 0] if len(velocities) > 1 else axes[0]
        ax1.plot(steps, data['velocity_error'], 'g-', linewidth=1)
        ax1.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5)
        ax1.set_title(f"{name}: Velocity Error (mean={np.mean(data['velocity_error']):.4f})")
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Error (m/s)')
        ax1.grid(True, alpha=0.3)
        
        # Height
        ax2 = axes[idx, 1] if len(velocities) > 1 else axes[1]
        ax2.plot(steps, data['height'], 'purple', linewidth=1)
        ax2.axhline(y=1.40, color='green', linestyle='--', alpha=0.5)
        ax2.set_title(f"{name}: Height (mean={np.mean(data['height']):.3f})")
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Height (m)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([1.0, 1.6])
        
        # XY trajectory
        ax3 = axes[idx, 2] if len(velocities) > 1 else axes[2]
        ax3.plot(data['x_position'], data['y_position'], 'b-', linewidth=1)
        ax3.scatter([0], [0], color='green', s=100, zorder=5, label='Start')
        ax3.scatter([data['x_position'][-1]], [data['y_position'][-1]], 
                    color='red', s=100, zorder=5, label='End')
        ax3.set_title(f"{name}: Trajectory")
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot walking controller performance")
    parser.add_argument('--model', type=str, required=True, help='Path to model .zip')
    parser.add_argument('--vecnorm', type=str, required=True, help='Path to VecNormalize .pkl')
    parser.add_argument('--vx', type=float, default=1.0, help='Target x-velocity')
    parser.add_argument('--vy', type=float, default=0.0, help='Target y-velocity')
    parser.add_argument('--steps', type=int, default=2000, help='Max steps per episode')
    parser.add_argument('--save', type=str, default=None, help='Save plot to path')
    parser.add_argument('--compare', action='store_true', help='Compare multiple velocities')
    args = parser.parse_args()
    
    # Load config
    cfg = load_yaml(os.path.join(PROJECT_ROOT, 'config/training_config.yaml'))
    walking = cfg.get('walking', {}).copy()
    walking['obs_history'] = walking.get('obs_history', 4)
    walking['obs_include_com'] = True
    walking['obs_feature_norm'] = True
    walking['action_smoothing'] = True
    walking['action_smoothing_tau'] = 0.2
    walking['random_height_init'] = False
    
    def create_vec_env(vx, vy):
        """Create a vec env with specific velocity command."""
        config = walking.copy()
        config['fixed_command'] = (vx, vy)
        env = make_walking_env(render_mode=None, config=config)
        _ = env.reset()
        vec_env = DummyVecEnv([lambda: env])
        
        if os.path.exists(args.vecnorm):
            vec_env = VecNormalize.load(args.vecnorm, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        
        return vec_env
    
    if args.compare:
        # Compare multiple velocities
        velocities = [
            (0.0, 0.0, "Standing"),
            (1.0, 0.0, "Forward 1.0"),
            (2.0, 0.0, "Forward 2.0"),
            (0.0, 1.0, "Sideways 1.0"),
            (1.0, 1.0, "Diagonal (1,1)"),
            (-1.0, 0.0, "Backward 1.0"),
        ]
        
        # Load model once
        vec_env = create_vec_env(0, 0)
        model = PPO.load(args.model, env=vec_env, device="auto")
        vec_env.close()
        
        plot_multi_velocity_comparison(
            model, create_vec_env, velocities, 
            max_steps=args.steps,
            save_path=args.save
        )
    else:
        # Single velocity analysis
        vec_env = create_vec_env(args.vx, args.vy)
        model = PPO.load(args.model, env=vec_env, device="auto")
        
        print(f"Collecting episode data for velocity ({args.vx}, {args.vy})...")
        data = collect_episode_data(model, vec_env, max_steps=args.steps)
        vec_env.close()
        
        plot_velocity_tracking(
            data, 
            title=f"Velocity Tracking: Command=({args.vx:.1f}, {args.vy:.1f}) m/s",
            save_path=args.save
        )


if __name__ == "__main__":
    main()

