# velocity_command_generator.py
"""
Velocity Command Generator for Humanoid RL Training.

Implements Uniform Command Sampling at randomized intervals with
configurable curriculum ranges and stop command probability.
"""

import numpy as np
from typing import List, Tuple, Optional


class VelocityCommandGenerator:
    """
    Generates target velocity commands [vx, vy, yaw_rate] for RL training.
    
    Features:
    - Uniform command sampling at randomized intervals (2-5 seconds)
    - 15% probability of "stop" command [0, 0, 0] for braking practice
    - Curriculum-based range updates for progressive difficulty
    
    Example usage:
        generator = VelocityCommandGenerator()
        command = generator.get_command(dt=0.01)  # Call every simulation step
    """
    
    def __init__(
        self,
        vx_range: Tuple[float, float] = (-0.5, 1.5),
        vy_range: Tuple[float, float] = (-0.5, 0.5),
        yaw_rate_range: Tuple[float, float] = (-1.0, 1.0),
        switch_interval_range: Tuple[float, float] = (2.0, 5.0),
        stop_probability: float = 0.15,
        seed: Optional[int] = None
    ):
        """
        Initialize the velocity command generator.
        
        Args:
            vx_range: (min, max) range for forward velocity in m/s
            vy_range: (min, max) range for lateral velocity in m/s
            yaw_rate_range: (min, max) range for yaw rate in rad/s
            switch_interval_range: (min, max) seconds between command changes
            stop_probability: Probability of generating a stop command [0,0,0]
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Command ranges
        self.vx_range = list(vx_range)
        self.vy_range = list(vy_range)
        self.yaw_rate_range = list(yaw_rate_range)
        
        # Timing parameters
        self.switch_interval_range = list(switch_interval_range)
        
        # Stop command probability
        self.stop_probability = stop_probability
        
        # Current command state
        self._current_command = np.zeros(3, dtype=np.float32)
        
        # Time tracking
        self._time_since_switch = 0.0
        self._next_switch_time = self._sample_switch_interval()
        
        # Statistics tracking
        self._total_commands_generated = 0
        self._stop_commands_generated = 0
        
        # Generate initial command
        self._sample_new_command()
    
    def _sample_switch_interval(self) -> float:
        """Sample the next command switch interval uniformly."""
        return np.random.uniform(
            self.switch_interval_range[0],
            self.switch_interval_range[1]
        )
    
    def _sample_new_command(self) -> None:
        """Sample a new command using uniform sampling with stop probability."""
        self._total_commands_generated += 1
        
        # Check for stop command (braking practice)
        if np.random.random() < self.stop_probability:
            self._current_command = np.zeros(3, dtype=np.float32)
            self._stop_commands_generated += 1
        else:
            # Sample each component uniformly from its range
            vx = np.random.uniform(self.vx_range[0], self.vx_range[1])
            vy = np.random.uniform(self.vy_range[0], self.vy_range[1])
            yaw_rate = np.random.uniform(self.yaw_rate_range[0], self.yaw_rate_range[1])
            
            self._current_command = np.array([vx, vy, yaw_rate], dtype=np.float32)
    
    def get_command(self, dt: float) -> np.ndarray:
        """
        Get the current velocity command, updating if interval has elapsed.
        
        This method should be called every simulation step. It will
        automatically handle timing and command switching.
        
        Args:
            dt: Simulation timestep in seconds
            
        Returns:
            np.ndarray: 3-element array [vx, vy, yaw_rate]
        """
        # Accumulate time
        self._time_since_switch += dt
        
        # Check if it's time to switch commands
        if self._time_since_switch >= self._next_switch_time:
            self._sample_new_command()
            self._time_since_switch = 0.0
            self._next_switch_time = self._sample_switch_interval()
        
        return self._current_command.copy()
    
    def update_curriculum_ranges(self, new_ranges: List[Tuple[float, float]]) -> None:
        """
        Update command ranges for curriculum learning.
        
        Allows progressive increase of difficulty by expanding velocity ranges.
        
        Args:
            new_ranges: List of 3 tuples [(vx_min, vx_max), (vy_min, vy_max), (yaw_min, yaw_max)]
            
        Example:
            # Start with easy ranges
            generator.update_curriculum_ranges([(-0.3, 0.5), (-0.2, 0.2), (-0.5, 0.5)])
            
            # Later, expand to harder ranges
            generator.update_curriculum_ranges([(-1.0, 2.0), (-1.0, 1.0), (-2.0, 2.0)])
        """
        if len(new_ranges) != 3:
            raise ValueError(f"Expected 3 range tuples, got {len(new_ranges)}")
        
        self.vx_range = list(new_ranges[0])
        self.vy_range = list(new_ranges[1])
        self.yaw_rate_range = list(new_ranges[2])
        
        print(f"Curriculum updated - vx: {self.vx_range}, vy: {self.vy_range}, yaw: {self.yaw_rate_range}")
    
    def force_new_command(self) -> np.ndarray:
        """
        Force generation of a new command immediately.
        
        Useful for episode resets.
        
        Returns:
            np.ndarray: The newly sampled command
        """
        self._sample_new_command()
        self._time_since_switch = 0.0
        self._next_switch_time = self._sample_switch_interval()
        return self._current_command.copy()
    
    def set_stop_probability(self, probability: float) -> None:
        """
        Update the stop command probability.
        
        Args:
            probability: New probability in range [0.0, 1.0]
        """
        self.stop_probability = np.clip(probability, 0.0, 1.0)
    
    def get_current_command(self) -> np.ndarray:
        """Get the current command without updating time."""
        return self._current_command.copy()
    
    def get_time_until_switch(self) -> float:
        """Get remaining time until next command switch."""
        return max(0.0, self._next_switch_time - self._time_since_switch)
    
    def get_statistics(self) -> dict:
        """
        Get generator statistics.
        
        Returns:
            dict: Statistics about command generation
        """
        return {
            'total_commands': self._total_commands_generated,
            'stop_commands': self._stop_commands_generated,
            'stop_ratio': (
                self._stop_commands_generated / max(1, self._total_commands_generated)
            ),
            'current_ranges': {
                'vx': self.vx_range,
                'vy': self.vy_range,
                'yaw_rate': self.yaw_rate_range
            }
        }
    
    def reset(self, sample_new: bool = True) -> np.ndarray:
        """
        Reset the generator state.
        
        Args:
            sample_new: If True, sample a new command. If False, set to zero.
            
        Returns:
            np.ndarray: The current command after reset
        """
        self._time_since_switch = 0.0
        self._next_switch_time = self._sample_switch_interval()
        
        if sample_new:
            self._sample_new_command()
        else:
            self._current_command = np.zeros(3, dtype=np.float32)
        
        return self._current_command.copy()


class VelocityCommandGeneratorWithSmoothing(VelocityCommandGenerator):
    """
    Extended generator with optional command smoothing/interpolation.
    
    Provides smoother transitions between commands instead of step changes,
    which can help with certain training scenarios.
    """
    
    def __init__(
        self,
        smoothing_tau: float = 0.1,
        **kwargs
    ):
        """
        Args:
            smoothing_tau: Exponential smoothing factor (0 = no smoothing, 1 = instant)
            **kwargs: Arguments passed to VelocityCommandGenerator
        """
        super().__init__(**kwargs)
        self.smoothing_tau = smoothing_tau
        self._smoothed_command = self._current_command.copy()
    
    def get_command(self, dt: float) -> np.ndarray:
        """Get smoothed velocity command."""
        # Get the target command from parent
        target = super().get_command(dt)
        
        # Apply exponential smoothing
        alpha = 1.0 - np.exp(-dt / max(self.smoothing_tau, 1e-6))
        self._smoothed_command = (
            (1 - alpha) * self._smoothed_command + alpha * target
        )
        
        return self._smoothed_command.copy()
    
    def reset(self, sample_new: bool = True) -> np.ndarray:
        """Reset with smoothed command sync."""
        command = super().reset(sample_new)
        self._smoothed_command = command.copy()
        return command


if __name__ == "__main__":
    # Quick test
    print("Testing VelocityCommandGenerator")
    print("=" * 50)
    
    generator = VelocityCommandGenerator(seed=42)
    
    dt = 0.01
    total_time = 10.0
    steps = int(total_time / dt)
    
    commands = []
    for _ in range(steps):
        cmd = generator.get_command(dt)
        commands.append(cmd.copy())
    
    commands = np.array(commands)
    
    print(f"\nSimulated {total_time:.1f} seconds ({steps} steps)")
    print(f"Command statistics:")
    print(f"  vx range: [{commands[:, 0].min():.2f}, {commands[:, 0].max():.2f}]")
    print(f"  vy range: [{commands[:, 1].min():.2f}, {commands[:, 1].max():.2f}]")
    print(f"  yaw range: [{commands[:, 2].min():.2f}, {commands[:, 2].max():.2f}]")
    
    stats = generator.get_statistics()
    print(f"\nGenerator stats: {stats}")

