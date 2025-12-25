# reward_calculator.py
"""
Modular reward calculator for humanoid velocity tracking.

Implements the refined reward function with:
- Gaussian kernel velocity tracking
- Direction bonus
- Stability rewards (height + upright)
- Action smoothness (jerk penalty)
- Alive bonus
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    """Configurable weights for each reward component."""
    tracking: float = 10.0          # Gaussian kernel for velocity matching
    direction_bonus: float = 5.0    # Bonus for correct direction
    height: float = 5.0             # Height maintenance
    upright: float = 3.0            # Orientation reward
    alive: float = 1.0              # Per-step survival bonus
    action_penalty: float = 0.005   # Effort penalty coefficient
    jerk_penalty: float = 0.01      # Action smoothness penalty


@dataclass
class RewardMetrics:
    """Container for individual reward components (for logging)."""
    tracking: float = 0.0
    direction_bonus: float = 0.0
    height: float = 0.0
    upright: float = 0.0
    alive: float = 0.0
    action_penalty: float = 0.0
    jerk_penalty: float = 0.0
    total: float = 0.0
    
    # Tracking metrics
    velocity_error: float = 0.0
    velocity_error_x: float = 0.0
    velocity_error_y: float = 0.0
    direction_error: float = 0.0
    jerk_magnitude: float = 0.0
    action_magnitude: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'reward/tracking': self.tracking,
            'reward/direction_bonus': self.direction_bonus,
            'reward/height': self.height,
            'reward/upright': self.upright,
            'reward/alive': self.alive,
            'reward/action_penalty': self.action_penalty,
            'reward/jerk_penalty': self.jerk_penalty,
            'reward/total': self.total,
            'metrics/velocity_error': self.velocity_error,
            'metrics/velocity_error_x': self.velocity_error_x,
            'metrics/velocity_error_y': self.velocity_error_y,
            'metrics/direction_error': self.direction_error,
            'metrics/jerk_magnitude': self.jerk_magnitude,
            'metrics/action_magnitude': self.action_magnitude,
        }


class RewardCalculator:
    """
    Calculates rewards for humanoid velocity tracking task.
    
    Reward Function:
        R_total = R_tracking + R_direction + R_height + R_upright + R_alive 
                  + R_action_penalty + R_jerk_penalty
    
    Example:
        calc = RewardCalculator()
        metrics = calc.compute(
            v_target=[1.0, 0.0],
            v_actual=[0.8, 0.1],
            height=1.4,
            quaternion=[0.98, 0.0, 0.1, 0.0],
            action=np.zeros(17),
            prev_action=np.zeros(17)
        )
        print(f"Total reward: {metrics.total}")
    """
    
    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        target_height: float = 1.40,
        height_bandwidth: float = 10.0,
        tracking_bandwidth: float = 4.0,
    ):
        """
        Initialize reward calculator.
        
        Args:
            weights: Custom reward weights (uses defaults if None)
            target_height: Target standing/walking height in meters
            height_bandwidth: Gaussian bandwidth for height reward
            tracking_bandwidth: Gaussian bandwidth for velocity tracking
        """
        self.weights = weights or RewardWeights()
        self.target_height = target_height
        self.height_bandwidth = height_bandwidth
        self.tracking_bandwidth = tracking_bandwidth
        
        # History for smoothing metrics
        self._prev_action = None
        
    def compute(
        self,
        v_target: np.ndarray,
        v_actual: np.ndarray,
        height: float,
        quaternion: np.ndarray,
        action: np.ndarray,
        prev_action: Optional[np.ndarray] = None,
    ) -> RewardMetrics:
        """
        Compute all reward components.
        
        Args:
            v_target: Target velocity [vx, vy] in m/s
            v_actual: Actual velocity [vx, vy] in m/s
            height: Current COM height in meters
            quaternion: Orientation quaternion [w, x, y, z]
            action: Current action array
            prev_action: Previous action (for jerk calculation)
            
        Returns:
            RewardMetrics containing all reward components
        """
        v_target = np.asarray(v_target)
        v_actual = np.asarray(v_actual)
        action = np.asarray(action)
        
        # Use stored prev_action if not provided
        if prev_action is None:
            prev_action = self._prev_action if self._prev_action is not None else action
        prev_action = np.asarray(prev_action)
        
        metrics = RewardMetrics()
        
        # ========== VELOCITY TRACKING (Gaussian Kernel) ==========
        vel_error_vec = v_target - v_actual
        velocity_error = np.linalg.norm(vel_error_vec)
        
        metrics.velocity_error = velocity_error
        metrics.velocity_error_x = abs(vel_error_vec[0]) if len(vel_error_vec) > 0 else 0.0
        metrics.velocity_error_y = abs(vel_error_vec[1]) if len(vel_error_vec) > 1 else 0.0
        
        # Gaussian kernel: exp(-β * ||error||²)
        R_tracking = self.weights.tracking * np.exp(
            -self.tracking_bandwidth * velocity_error**2
        )
        metrics.tracking = R_tracking
        
        # ========== DIRECTION BONUS (when speed commanded) ==========
        speed_cmd = np.linalg.norm(v_target)
        R_direction = 0.0
        
        if speed_cmd > 0.1:
            # Normalize target to get direction
            direction_cmd = v_target / speed_cmd
            
            # Project actual velocity onto commanded direction
            projected_speed = np.dot(v_actual, direction_cmd)
            
            # Reward for moving in correct direction
            direction_ratio = np.clip(projected_speed / speed_cmd, 0.0, 1.0)
            R_direction = self.weights.direction_bonus * direction_ratio
            
            # Calculate angular error
            speed_actual = np.linalg.norm(v_actual)
            if speed_actual > 0.1:
                direction_actual = v_actual / speed_actual
                cos_angle = np.clip(np.dot(direction_cmd, direction_actual), -1.0, 1.0)
                metrics.direction_error = np.arccos(cos_angle)
            else:
                metrics.direction_error = np.pi  # Max error if not moving
        
        metrics.direction_bonus = R_direction
        
        # ========== HEIGHT REWARD ==========
        height_error = abs(height - self.target_height)
        R_height = self.weights.height * np.exp(
            -self.height_bandwidth * height_error**2
        )
        metrics.height = R_height
        
        # ========== UPRIGHT REWARD ==========
        quat_w = abs(quaternion[0])
        if quat_w > 0.85:
            R_upright = self.weights.upright
        else:
            R_upright = self.weights.upright * (quat_w / 0.85)
        metrics.upright = R_upright
        
        # ========== ALIVE BONUS ==========
        R_alive = self.weights.alive
        metrics.alive = R_alive
        
        # ========== ACTION PENALTY (Effort) ==========
        action_magnitude = np.sum(action**2)
        R_action_penalty = -self.weights.action_penalty * action_magnitude
        metrics.action_penalty = R_action_penalty
        metrics.action_magnitude = np.sqrt(action_magnitude)
        
        # ========== JERK PENALTY (Smoothness) ==========
        action_delta = action - prev_action
        jerk_magnitude = np.sum(action_delta**2)
        R_jerk_penalty = -self.weights.jerk_penalty * jerk_magnitude
        metrics.jerk_penalty = R_jerk_penalty
        metrics.jerk_magnitude = np.sqrt(jerk_magnitude)
        
        # ========== TOTAL REWARD ==========
        metrics.total = (
            R_tracking +
            R_direction +
            R_height +
            R_upright +
            R_alive +
            R_action_penalty +
            R_jerk_penalty
        )
        
        # Store action for next call
        self._prev_action = action.copy()
        
        return metrics
    
    def reset(self):
        """Reset calculator state (call at episode start)."""
        self._prev_action = None
    
    def get_expected_reward_range(self) -> Tuple[float, float]:
        """
        Get expected reward range for normalization reference.
        
        Returns:
            (min_expected, max_expected) reward per step
        """
        # Maximum reward (perfect tracking)
        max_reward = (
            self.weights.tracking +      # Full tracking
            self.weights.direction_bonus + # Full direction
            self.weights.height +         # Perfect height
            self.weights.upright +        # Perfect upright
            self.weights.alive            # Alive bonus
        )
        
        # Minimum reward (poor performance but not terminated)
        min_reward = (
            0.0 +                         # No tracking
            0.0 +                         # No direction
            0.0 +                         # Poor height
            0.0 +                         # Poor upright
            self.weights.alive +          # Still alive
            -5.0 +                        # Moderate action penalty
            -2.0                          # Moderate jerk penalty
        )
        
        return min_reward, max_reward


class AdaptiveRewardCalculator(RewardCalculator):
    """
    Reward calculator with curriculum-based weight adaptation.
    
    Adjusts reward weights based on training stage to focus on
    different aspects at different points in training.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stage = 0
        
    def set_stage(self, stage: int):
        """
        Update reward weights for curriculum stage.
        
        Stage progression:
        - Early stages: Focus on stability and survival
        - Mid stages: Balance tracking and stability  
        - Late stages: Focus on precise tracking
        """
        self.stage = stage
        
        if stage <= 1:
            # Early: Prioritize staying alive and stable
            self.weights.tracking = 8.0
            self.weights.height = 6.0
            self.weights.upright = 4.0
            self.weights.alive = 2.0
            self.weights.jerk_penalty = 0.015  # Higher smoothness requirement
        elif stage <= 3:
            # Mid: Balance
            self.weights.tracking = 10.0
            self.weights.height = 5.0
            self.weights.upright = 3.0
            self.weights.alive = 1.0
            self.weights.jerk_penalty = 0.01
        else:
            # Late: Precise tracking
            self.weights.tracking = 12.0
            self.weights.direction_bonus = 6.0
            self.weights.height = 4.0
            self.weights.upright = 2.0
            self.weights.alive = 0.5
            self.weights.jerk_penalty = 0.008  # Lower penalty for agility


if __name__ == "__main__":
    # Test the reward calculator
    print("Testing RewardCalculator")
    print("=" * 50)
    
    calc = RewardCalculator()
    
    # Test case 1: Perfect tracking
    metrics = calc.compute(
        v_target=np.array([1.0, 0.0]),
        v_actual=np.array([1.0, 0.0]),
        height=1.40,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        action=np.zeros(17),
        prev_action=np.zeros(17)
    )
    print(f"\nPerfect tracking:")
    print(f"  Total reward: {metrics.total:.2f}")
    print(f"  Velocity error: {metrics.velocity_error:.4f}")
    
    # Test case 2: Poor tracking
    metrics = calc.compute(
        v_target=np.array([1.0, 0.0]),
        v_actual=np.array([0.0, 0.5]),
        height=1.20,
        quaternion=np.array([0.8, 0.2, 0.0, 0.0]),
        action=np.ones(17) * 0.5,
        prev_action=np.zeros(17)
    )
    print(f"\nPoor tracking:")
    print(f"  Total reward: {metrics.total:.2f}")
    print(f"  Velocity error: {metrics.velocity_error:.4f}")
    print(f"  Direction error: {np.degrees(metrics.direction_error):.1f}°")
    
    # Test expected range
    min_r, max_r = calc.get_expected_reward_range()
    print(f"\nExpected reward range: [{min_r:.1f}, {max_r:.1f}]")

