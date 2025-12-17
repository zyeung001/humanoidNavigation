# plotting.py
"""
Plotting utilities for training visualization.

Contains functions for plotting:
- Velocity commands over time
- Reward components
- Episode metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def simulate_and_plot_commands(
    generator,
    duration: float = 60.0,
    dt: float = 0.01,
    save_path: Optional[str] = None,
    show: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate velocity command generation and create visualization.
    
    Args:
        generator: VelocityCommandGenerator instance
        duration: Total simulation time in seconds
        dt: Timestep in seconds
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        (time_array, vx_array, vy_array, speed_array)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed")
        return None, None, None, None
    
    n_steps = int(duration / dt)
    
    time_array = np.zeros(n_steps)
    vx_array = np.zeros(n_steps)
    vy_array = np.zeros(n_steps)
    speed_array = np.zeros(n_steps)
    
    print(f"Simulating {duration:.0f}s ({n_steps} steps)...")
    
    for i in range(n_steps):
        command = generator.get_command(dt)
        time_array[i] = i * dt
        vx_array[i] = command[0]
        vy_array[i] = command[1]
        speed_array[i] = np.sqrt(command[0]**2 + command[1]**2)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f'Velocity Commands - {duration}s simulation',
        fontsize=14, fontweight='bold'
    )
    
    colors = {'vx': '#2E86AB', 'vy': '#A23B72', 'speed': '#F18F01'}
    
    # Subplot 1: vx
    axes[0].plot(time_array, vx_array, color=colors['vx'], linewidth=1.5)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('vx (m/s)')
    axes[0].set_title('Forward Velocity')
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: vy
    axes[1].plot(time_array, vy_array, color=colors['vy'], linewidth=1.5)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('vy (m/s)')
    axes[1].set_title('Lateral Velocity')
    axes[1].grid(True, alpha=0.3)
    
    # Subplot 3: speed
    axes[2].plot(time_array, speed_array, color=colors['speed'], linewidth=1.5)
    axes[2].fill_between(time_array, 0, speed_array, alpha=0.3, color=colors['speed'])
    axes[2].set_ylabel('Speed (m/s)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Speed Magnitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    
    return time_array, vx_array, vy_array, speed_array


def plot_reward_components(
    reward_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot reward components over an episode.
    
    Args:
        reward_history: Dict mapping component name to list of values
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed")
        return
    
    n_components = len(reward_history)
    if n_components == 0:
        print("No reward components to plot")
        return
    
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 2.5 * n_components), sharex=True)
    
    if n_components == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_components))
    
    for ax, (name, values), color in zip(axes, reward_history.items(), colors):
        steps = np.arange(len(values))
        ax.plot(steps, values, color=color, linewidth=1.0, alpha=0.8)
        ax.fill_between(steps, 0, values, alpha=0.3, color=color)
        ax.set_ylabel(name)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add stats
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color=color, linestyle=':', alpha=0.7)
        ax.text(len(values) * 0.02, mean_val, f'μ={mean_val:.2f}', 
                fontsize=9, color=color)
    
    axes[-1].set_xlabel('Step')
    fig.suptitle('Reward Components', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()


def plot_episode_metrics(
    episodes: List[Dict],
    metrics: List[str] = ['reward', 'length', 'velocity_error'],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training metrics across episodes.
    
    Args:
        episodes: List of episode info dicts
        metrics: Which metrics to plot
        save_path: Optional path to save figure
        show: Whether to display
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed")
        return
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    episode_nums = np.arange(len(episodes))
    
    for ax, metric in zip(axes, metrics):
        values = [ep.get(metric, 0) for ep in episodes]
        ax.plot(episode_nums, values, 'b-', alpha=0.5, linewidth=0.5)
        
        # Rolling average
        window = min(50, len(values) // 4)
        if window > 1:
            rolling_avg = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(values)), rolling_avg, 
                   'r-', linewidth=2, label=f'{window}-ep avg')
            ax.legend()
        
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Episode')
    fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()


def plot_curriculum_progression(
    stage_history: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot curriculum stage progression over training.
    
    Args:
        stage_history: List of {'timestep': int, 'stage': int} dicts
        save_path: Optional save path
        show: Whether to display
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed")
        return
    
    timesteps = [h['timestep'] for h in stage_history]
    stages = [h['stage'] for h in stage_history]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.step(timesteps, stages, where='post', color='#2E86AB', linewidth=2)
    ax.scatter(timesteps, stages, color='#E63946', s=50, zorder=5)
    
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Curriculum Stage')
    ax.set_title('Curriculum Progression', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, max(stages) + 0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()

