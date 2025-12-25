# plot_velocity_commands.py
"""
Visualization script for VelocityCommandGenerator.

Simulates and plots the generated target commands over a 60-second period,
demonstrating the step-like changes and zero-velocity braking periods
characteristic of Uniform Command Sampling.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import from core module (Prompt 1 implementation)
try:
    from src.core.command_generator import VelocityCommandGenerator
except ImportError:
    # Fallback for running from src/utils directory
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.command_generator import VelocityCommandGenerator


def simulate_and_plot(
    duration: float = 60.0,
    dt: float = 0.01,
    seed: int = 42,
    save_path: str = None
):
    """
    Simulate velocity command generation and create visualization.
    
    Args:
        duration: Total simulation time in seconds
        dt: Timestep in seconds
        seed: Random seed for reproducibility
        save_path: Optional path to save the figure
    """
    # Initialize generator
    generator = VelocityCommandGenerator(
        vx_range=(-0.5, 1.5),
        vy_range=(-0.5, 0.5),
        yaw_rate_range=(-1.0, 1.0),
        switch_interval_range=(2.0, 5.0),
        stop_probability=0.15,
        seed=seed
    )
    
    # Calculate number of steps
    n_steps = int(duration / dt)
    
    # Storage arrays
    time_array = np.zeros(n_steps)
    vx_array = np.zeros(n_steps)
    vy_array = np.zeros(n_steps)
    yaw_rate_array = np.zeros(n_steps)
    speed_array = np.zeros(n_steps)
    
    # Run simulation
    print(f"Simulating {duration:.0f} seconds ({n_steps} steps) with dt={dt}s...")
    
    for i in range(n_steps):
        # Get current command
        command = generator.get_command(dt)
        
        # Store values
        time_array[i] = i * dt
        vx_array[i] = command[0]
        vy_array[i] = command[1]
        yaw_rate_array[i] = command[2]
        speed_array[i] = np.sqrt(command[0]**2 + command[1]**2)
    
    # Get generator statistics
    stats = generator.get_statistics()
    print(f"\nGenerator Statistics:")
    print(f"  Total commands sampled: {stats['total_commands']}")
    print(f"  Stop commands: {stats['stop_commands']}")
    print(f"  Stop ratio: {stats['stop_ratio']:.1%}")
    
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        'Velocity Command Generator - Uniform Sampling Visualization\n'
        f'Duration: {duration}s | dt: {dt}s | Stop Probability: 15%',
        fontsize=14, fontweight='bold'
    )
    
    # Color scheme
    colors = {
        'vx': '#2E86AB',      # Steel blue
        'vy': '#A23B72',      # Raspberry
        'speed': '#F18F01',   # Orange
        'zero': '#E63946',    # Red
        'grid': '#E0E0E0'
    }
    
    # Subplot 1: Forward velocity (vx)
    ax1 = axes[0]
    ax1.plot(time_array, vx_array, color=colors['vx'], linewidth=1.5, label='vx (forward)')
    ax1.axhline(y=0, color=colors['zero'], linestyle='--', alpha=0.5, linewidth=1)
    ax1.fill_between(time_array, 0, vx_array, alpha=0.3, color=colors['vx'])
    ax1.set_ylabel('vx (m/s)', fontsize=12)
    ax1.set_ylim(-1.0, 2.0)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    ax1.set_title('Forward Velocity Command', fontsize=11)
    
    # Highlight zero regions (braking periods)
    _highlight_zero_regions(ax1, time_array, vx_array, vy_array, colors['zero'])
    
    # Subplot 2: Lateral velocity (vy)
    ax2 = axes[1]
    ax2.plot(time_array, vy_array, color=colors['vy'], linewidth=1.5, label='vy (lateral)')
    ax2.axhline(y=0, color=colors['zero'], linestyle='--', alpha=0.5, linewidth=1)
    ax2.fill_between(time_array, 0, vy_array, alpha=0.3, color=colors['vy'])
    ax2.set_ylabel('vy (m/s)', fontsize=12)
    ax2.set_ylim(-1.0, 1.0)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, color=colors['grid'])
    ax2.set_title('Lateral Velocity Command', fontsize=11)
    
    # Highlight zero regions
    _highlight_zero_regions(ax2, time_array, vx_array, vy_array, colors['zero'])
    
    # Subplot 3: Speed magnitude
    ax3 = axes[2]
    ax3.plot(time_array, speed_array, color=colors['speed'], linewidth=1.5, label='speed')
    ax3.fill_between(time_array, 0, speed_array, alpha=0.3, color=colors['speed'])
    ax3.axhline(y=0, color=colors['zero'], linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_ylabel('Speed (m/s)', fontsize=12)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylim(0, 2.0)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, color=colors['grid'])
    ax3.set_title(r'Speed Magnitude: $\sqrt{vx^2 + vy^2}$', fontsize=11)
    
    # Highlight zero regions
    _highlight_zero_regions(ax3, time_array, vx_array, vy_array, colors['zero'])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    return time_array, vx_array, vy_array, speed_array


def _highlight_zero_regions(ax, time_array, vx_array, vy_array, color):
    """Highlight regions where both vx and vy are zero (braking periods)."""
    # Find zero regions (both vx and vy are 0)
    is_zero = (np.abs(vx_array) < 1e-6) & (np.abs(vy_array) < 1e-6)
    
    # Find transitions
    in_zero_region = False
    start_idx = 0
    
    for i in range(len(is_zero)):
        if is_zero[i] and not in_zero_region:
            # Start of zero region
            start_idx = i
            in_zero_region = True
        elif not is_zero[i] and in_zero_region:
            # End of zero region
            ax.axvspan(
                time_array[start_idx], time_array[i],
                alpha=0.15, color=color, label='_nolegend_'
            )
            in_zero_region = False
    
    # Handle case where simulation ends in zero region
    if in_zero_region:
        ax.axvspan(
            time_array[start_idx], time_array[-1],
            alpha=0.15, color=color, label='_nolegend_'
        )


def plot_yaw_rate(
    duration: float = 60.0,
    dt: float = 0.01,
    seed: int = 42,
    save_path: str = None
):
    """
    Additional plot showing yaw rate commands.
    
    Args:
        duration: Total simulation time in seconds
        dt: Timestep in seconds
        seed: Random seed for reproducibility
        save_path: Optional path to save the figure
    """
    # Initialize generator
    generator = VelocityCommandGenerator(
        vx_range=(-0.5, 1.5),
        vy_range=(-0.5, 0.5),
        yaw_rate_range=(-1.0, 1.0),
        switch_interval_range=(2.0, 5.0),
        stop_probability=0.15,
        seed=seed
    )
    
    n_steps = int(duration / dt)
    time_array = np.zeros(n_steps)
    yaw_rate_array = np.zeros(n_steps)
    
    for i in range(n_steps):
        command = generator.get_command(dt)
        time_array[i] = i * dt
        yaw_rate_array[i] = command[2]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))
    
    ax.plot(time_array, yaw_rate_array, color='#6B4C9A', linewidth=1.5, label='yaw_rate')
    ax.axhline(y=0, color='#E63946', linestyle='--', alpha=0.5, linewidth=1)
    ax.fill_between(time_array, 0, yaw_rate_array, alpha=0.3, color='#6B4C9A')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Yaw Rate (rad/s)', fontsize=12)
    ax.set_title('Yaw Rate Command - Uniform Sampling', fontsize=14, fontweight='bold')
    ax.set_ylim(-1.5, 1.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Yaw rate figure saved to: {save_path}")
    
    plt.show()


def plot_command_distribution(n_samples: int = 10000, seed: int = 42):
    """
    Plot the distribution of sampled commands to verify uniform sampling.
    
    Args:
        n_samples: Number of command samples to generate
        seed: Random seed
    """
    generator = VelocityCommandGenerator(seed=seed)
    
    # Force sample many commands
    commands = []
    for _ in range(n_samples):
        generator._sample_new_command()
        commands.append(generator.get_current_command())
    
    commands = np.array(commands)
    
    # Filter out stop commands for distribution plot
    non_zero_mask = np.any(commands != 0, axis=1)
    active_commands = commands[non_zero_mask]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    labels = ['vx (m/s)', 'vy (m/s)', 'yaw_rate (rad/s)']
    ranges = [(-0.5, 1.5), (-0.5, 0.5), (-1.0, 1.0)]
    colors = ['#2E86AB', '#A23B72', '#6B4C9A']
    
    for i, (ax, label, rng, color) in enumerate(zip(axes, labels, ranges, colors)):
        ax.hist(active_commands[:, i], bins=50, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(x=rng[0], color='red', linestyle='--', alpha=0.7, label='Range bounds')
        ax.axvline(x=rng[1], color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{label} Distribution', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    stats = generator.get_statistics()
    fig.suptitle(
        f'Command Distribution ({n_samples} samples, '
        f'{stats["stop_commands"]} stop commands = {stats["stop_ratio"]:.1%})',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Velocity Command Generator Visualization")
    print("=" * 60)
    
    # Main visualization (vx, vy, speed)
    simulate_and_plot(
        duration=60.0,
        dt=0.01,
        seed=42,
        save_path=None  # Set to 'velocity_commands.png' to save
    )
    
    # Yaw rate visualization
    plot_yaw_rate(
        duration=60.0,
        dt=0.01,
        seed=42,
        save_path=None
    )
    
    # Distribution plot
    plot_command_distribution(n_samples=10000, seed=42)

