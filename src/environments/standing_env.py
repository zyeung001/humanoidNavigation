"""
Standing environment wrapper for MuJoCo Humanoid-v5
OPTIMIZED reward for indefinite standing balance

FIXES:
- Simplified reward function (height first, then stability)
- More lenient termination conditions
- Better reward shaping with clearer objectives
- Enhanced debugging output

standing_env.py
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from mujoco import mj_name2id, mjtObj

class StandingEnv(gym.Wrapper):
    """Wrapper for MuJoCo Humanoid environment optimized for standing task

    Enhancements (config-togglable, defaults preserve existing behavior):
    - Refactored reward via compute_reward() with components: height, upright, stability, smoothness,
      control cost, sustained bonus, velocity penalties, foot-contact shaping, recovery handling.
    - Action preprocessing: optional smoothing, clipping to limits, symmetry constraints, PD assist.
    - Observation processing: optional feature normalization, COM features, history stacking.
    """
    
    def __init__(self, render_mode: Optional[str] = None, config=None):
        env_id = "Humanoid-v5"
        print(f"Using {env_id} for standing task")
        
        # FIXED: Include position in observations so agent can correct for drift
        # This fixes the MDP observability violation where we penalize position
        # but the agent couldn't observe it
        env = gym.make(
            env_id, 
            render_mode=render_mode,
            exclude_current_positions_from_observation=False  # CRITICAL FIX
        )
        super().__init__(env)
        
        
        # FIXED: Correct target height for Humanoid-v5 natural standing pose
        self.base_target_height = 1.4
        
        self.max_episode_steps = config.get('max_episode_steps', 5000) if config else 5000
        self.current_step = 0
        
        self.domain_rand = config.get('domain_rand', False) if config else False
        self.rand_mass_range = config.get('rand_mass_range', [0.95, 1.05]) if config else [0.95, 1.05]
        self.rand_friction_range = config.get('rand_friction_range', [0.95, 1.05]) if config else [0.95, 1.05]
        
        self.reward_history = {
            'height': [], 'upright': [], 'velocity': [], 
            'angular': [], 'position': [], 'control': []
        }

        # ======== Enhanced controls (all optional via config) ========
        self.cfg = config or {}
        # Action preprocessing
        self.enable_action_smoothing = bool(self.cfg.get('action_smoothing', False))
        self.action_smoothing_tau = float(self.cfg.get('action_smoothing_tau', 0.15))  # 0..1 low-pass
        self.enable_action_symmetry = bool(self.cfg.get('action_symmetry', False))
        self.enable_pd_assist = bool(self.cfg.get('pd_assist', False))
        self.pd_kp = float(self.cfg.get('pd_kp', 0.0))
        self.pd_kd = float(self.cfg.get('pd_kd', 0.0))
        self.prev_action = np.zeros(self.env.action_space.shape, dtype=np.float32)

        # Observation processing
        self.enable_history = int(self.cfg.get('obs_history', 0)) > 0
        self.history_len = int(self.cfg.get('obs_history', 0))
        self.obs_history = []
        self.include_com = bool(self.cfg.get('obs_include_com', False))
        self.feature_norm = bool(self.cfg.get('obs_feature_norm', False))
        self._proc_obs_dim: Optional[int] = None  # will be frozen after first processed obs
        
        obs_space = env.observation_space
        print(f"Observation space shape for standing: {obs_space.shape}")

        # Adjust observation_space if enhanced observation processing is enabled
        try:
            base_dim = int(obs_space.shape[0])
            extra_dim = 6 if self.include_com else 0  # COM pos (3) + COM vel (3)
            stack = self.history_len if self.enable_history else 1
            new_dim = (base_dim + extra_dim) * stack
            if new_dim != base_dim:
                from gymnasium.spaces import Box
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32)
                print(f"Adjusted observation space for processing: base={base_dim}, extra={extra_dim}, stack={stack} -> {new_dim}")
        except Exception:
            # Fallback: keep original observation_space
            pass
    
    def reset(self, seed: Optional[int] = None): 
        observation, info = self.env.reset(seed=seed)
        
        default_height = self.env.unwrapped.data.qpos[2]
        
        self.current_step = 0
        self.prev_height = default_height
        self.target_height = self.base_target_height
        self.prev_action[:] = 0.0
        self.obs_history = []
        
        # Clear reward history
        for key in self.reward_history:
            self.reward_history[key] = []
        
        if self.domain_rand:
            # Randomize body masses
            original_masses = self.env.unwrapped.model.body_mass.copy()
            self.env.unwrapped.model.body_mass *= np.random.uniform(
                self.rand_mass_range[0], self.rand_mass_range[1],
                size=self.env.unwrapped.model.body_mass.shape
            )
            
            # Randomize geom friction (for feet/contact surfaces)
            original_friction = self.env.unwrapped.model.geom_friction.copy()
            self.env.unwrapped.model.geom_friction[:, 0] *= np.random.uniform(
                self.rand_friction_range[0], self.rand_friction_range[1],
                size=self.env.unwrapped.model.geom_friction.shape[0]
            )
        
        # Process initial observation if using enhanced obs and freeze processed dim
        if self.enable_history or self.include_com or self.feature_norm:
            observation = self._process_observation(observation)
            # Freeze processed observation dimension and update observation_space accordingly
            if self._proc_obs_dim is None:
                self._proc_obs_dim = int(np.asarray(observation, dtype=np.float32).shape[0])
                try:
                    from gymnasium.spaces import Box
                    self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self._proc_obs_dim,), dtype=np.float32)
                    print(f"Frozen processed observation dim: {self._proc_obs_dim}")
                except Exception:
                    pass

        return observation, info
    
    def step(self, action):
        # Action preprocessing
        proc_action = self._process_action(np.asarray(action, dtype=np.float32))

        observation, base_reward, terminated, truncated, info = self.env.step(proc_action)
        self.current_step += 1
        
        # Modify reward for standing
        reward, terminated = self._compute_task_reward(observation, base_reward, info, proc_action)
        
        # Override termination for standing to allow indefinite episodes
        truncated = self.current_step >= self.max_episode_steps
        
        # Add task info 
        info.update(self._get_task_info())
        
        # Process observation if enabled
        if self.enable_history or self.include_com or self.feature_norm:
            observation = self._process_observation(observation)

        return observation, reward, terminated, truncated, info
        
    def _compute_task_reward(self, obs, base_reward, info, action):
        """
        OPTIMIZED reward function to solve the "crouching local optimum" problem
        
        CRITICAL FIX: Strong height penalty to escape low-height local minima
        
        Key improvements:
        1. ASYMMETRIC height reward: HUGE penalty for low heights, lenient near target
        2. Progressive height bonus: increasing rewards from 1.0m to 1.4m
        3. Height gradient: agent MUST learn to stand taller to get positive reward
        4. Stability comes AFTER height: don't reward stability at wrong height
        
        Reward strategy:
        - Below 1.0m: LARGE negative penalty (-100 points)
        - 1.0-1.2m: Neutral/slightly negative (-10 to +20)
        - 1.2-1.35m: Positive but not maximal (+20 to +60)
        - 1.35-1.45m: Near-maximal (+60 to +100)
        - Above 1.5m: Small penalty for overextension
        """
        
        # ========== STATE EXTRACTION ==========
        height = self.env.unwrapped.data.qpos[2]
        root_x, root_y = self.env.unwrapped.data.qpos[0:2]
        quat = self.env.unwrapped.data.qpos[3:7]  # [w, x, y, z] quaternion
        
        linear_vel = self.env.unwrapped.data.qvel[0:3]
        angular_vel = self.env.unwrapped.data.qvel[3:6]
        joint_vel = self.env.unwrapped.data.qvel[6:]  # Joint velocities
        
        # Target height
        target_height = self.base_target_height
        height_error = abs(height - target_height)
        
        # ========== CORE REWARD: ASYMMETRIC HEIGHT REWARD ==========
        # This is THE KEY FIX - make low heights extremely unappealing
        
        # CRITICAL: Penalize low heights HARD to escape crouching local optimum
        if height < 1.0:
            # EXTREME penalty for crouching: -100 to -20 points
            height_reward = -100.0 + 80.0 * np.clip(height / 1.0, 0.0, 1.0)
        elif height < 1.2:
            # Still penalize for being too low: -10 to +10 points
            # Create gradient forcing agent upward
            height_reward = -10.0 + 20.0 * (height - 1.0) / 0.2
        elif height < 1.35:
            # Moderate reward for decent height: +10 to +50 points
            height_reward = 10.0 + 40.0 * (height - 1.2) / 0.15
        elif height < 1.45:
            # EXCELLENT reward at target height: +50 to +100 points
            height_reward = 50.0 + 50.0 * (height - 1.35) / 0.1
        elif height < 1.6:
            # Small penalty for overextending: +100 to +80 points
            height_reward = 100.0 - 20.0 * (height - 1.45) / 0.15
        else:
            # Very tall = bad: +80 to +50 points
            height_reward = 80.0 - 30.0 * np.clip((height - 1.6) / 0.2, 0.0, 1.0)
        
        # ========== UPRIGHT ORIENTATION REWARD ==========
        # Only reward upright IF already at decent height
        upright_error = 1.0 - abs(quat[0])
        if height >= 1.2:
            # Full reward when tall enough
            upright_reward = 25.0 * np.exp(-8.0 * upright_error**2)
        elif height >= 1.0:
            # Reduced reward when somewhat low
            upright_reward = 15.0 * np.exp(-8.0 * upright_error**2)
        else:
            # No upright reward when crouching
            upright_reward = 0.0
        
        # ========== STABILITY REWARD ==========
        # Only reward stability when at correct height
        angular_momentum = np.sum(np.square(angular_vel))
        if height >= 1.3:
            stability_reward = 15.0 * np.exp(-2.0 * angular_momentum)
        elif height >= 1.2:
            stability_reward = 10.0 * np.exp(-2.0 * angular_momentum)
        elif height >= 1.0:
            stability_reward = 5.0 * np.exp(-2.0 * angular_momentum)
        else:
            stability_reward = 0.0
        
        # ========== SMOOTHNESS REWARD ==========
        joint_velocity_magnitude = np.sum(np.square(joint_vel))
        smoothness_reward = 5.0 * np.exp(-0.1 * joint_velocity_magnitude)
        
        # ========== CONTROL COST ==========
        # Reduced penalty to not discourage exploration
        control_cost = -0.2 * np.sum(np.square(action))
        
        # ========== VELOCITY PENALTY ==========
        speed = np.linalg.norm(linear_vel)
        velocity_penalty = -0.5 * np.clip(speed - 1.0, 0.0, 2.0)
        
        # ========== SPARSE BONUS ==========
        # Huge bonus ONLY for sustained good standing
        sustained_bonus = 0.0
        if self.current_step > 0 and self.current_step % 100 == 0:
            if height_error < 0.10 and upright_error < 0.08 and height >= 1.3:
                sustained_bonus = 200.0  # ENORMOUS bonus for sustained target performance
        
        # ========== TOTAL REWARD ==========
        total_reward = (
            height_reward +          # THE DOMINANT term (-100 to +100)
            upright_reward +         # +0 to +25 (conditional)
            stability_reward +       # +0 to +15 (conditional)
            smoothness_reward +      # +0 to +5
            control_cost +           # -4 to 0 (small)
            velocity_penalty +       # 0 to -1 (small)
            sustained_bonus          # 0 or +200 (rare)
        )
        
        # ========== TERMINATION CONDITIONS ==========
        terminate = (
            height < 0.75 or         # Below 0.75m = fallen
            height > 2.0 or          # Unrealistic height
            abs(quat[0]) < 0.6       # Torso > 53 degrees
        )
        
        # ========== TRACK REWARD COMPONENTS ==========
        self.reward_history['height'].append(height_reward)
        self.reward_history['upright'].append(upright_reward)
        self.reward_history['velocity'].append(stability_reward)
        self.reward_history['control'].append(control_cost)
        
        # ========== DEBUG LOGGING ==========
        if self.current_step % 100 == 0:
            print(f"Step {self.current_step:4d}: "
                f"h={height:.3f} (err={height_error:.3f}), "
                f"quat_w={quat[0]:.3f}, "
                f"r={total_reward:6.1f} "
                f"[h={height_reward:6.1f}, u={upright_reward:4.1f}, "
                f"stab={stability_reward:4.1f}, bonus={sustained_bonus:.0f}]")
        
        return total_reward, terminate

    # ======== Action and Observation Processing ========

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        # Symmetry constraint (optional): average symmetric joints if configured
        if self.enable_action_symmetry:
            # Generic symmetry: pair even/odd indices
            half = action.shape[-1] // 2
            if half > 0:
                left = action[:half]
                right = action[-half:]
                mean_lr = 0.5 * (left + right)
                action[:half] = mean_lr
                action[-half:] = mean_lr

        # PD assist (optional): small stabilizing term towards zero position
        if self.enable_pd_assist and (self.pd_kp > 0.0 or self.pd_kd > 0.0):
            try:
                qpos = self.env.unwrapped.data.qpos[7:7+action.shape[-1]]  # skip root
                qvel = self.env.unwrapped.data.qvel[6:6+action.shape[-1]]
                pd = (-self.pd_kp * qpos) + (-self.pd_kd * qvel)
                action = np.clip(action + pd, -1.0, 1.0)
            except Exception:
                pass

        # Low-pass smoothing (optional)
        if self.enable_action_smoothing:
            tau = np.clip(self.action_smoothing_tau, 0.0, 1.0)
            action = (1.0 - tau) * self.prev_action + tau * action

        # Clip to action space
        low, high = self.env.action_space.low, self.env.action_space.high
        action = np.clip(action, low, high)
        self.prev_action = action.copy()
        return action

    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        features = [obs]

        # Add COM features if enabled
        if self.include_com:
            try:
                com_pos = self.env.unwrapped.data.subtree_com[0]  # root subtree COM
                com_vel = self.env.unwrapped.data.cdof_dot[:3] if hasattr(self.env.unwrapped.data, 'cdof_dot') else self.env.unwrapped.data.qvel[:3]
                features.append(np.asarray(com_pos, dtype=np.float32))
                features.append(np.asarray(com_vel, dtype=np.float32))
            except Exception:
                pass

        feat_vec = np.concatenate([np.atleast_1d(f).ravel() for f in features]).astype(np.float32)

        # Feature normalization (simple running scale-free tanh)
        if self.feature_norm:
            feat_vec = np.tanh(feat_vec * 0.1)

        # History stacking (k previous obs)
        if self.enable_history:
            self.obs_history.append(feat_vec)
            self.obs_history = self.obs_history[-self.history_len:]
            if len(self.obs_history) < self.history_len:
                # pad with zeros
                pad = [np.zeros_like(feat_vec) for _ in range(self.history_len - len(self.obs_history))]
                stacked = pad + self.obs_history
            else:
                stacked = self.obs_history
            feat_vec = np.concatenate(stacked, axis=0)

        # Ensure fixed processed dimension by padding/truncating to frozen size
        if self._proc_obs_dim is not None:
            if feat_vec.shape[0] < self._proc_obs_dim:
                pad = np.zeros((self._proc_obs_dim - feat_vec.shape[0],), dtype=np.float32)
                feat_vec = np.concatenate([feat_vec, pad], axis=0)
            elif feat_vec.shape[0] > self._proc_obs_dim:
                feat_vec = feat_vec[:self._proc_obs_dim]

        return feat_vec
        
    def _get_task_info(self):
        """Get task-specific information"""
        root_x, root_y = self.env.unwrapped.data.qpos[0:2]
        dist = np.sqrt(root_x**2 + root_y**2)
        return {
            'height': self.env.unwrapped.data.qpos[2],
            'distance_from_origin': dist,
            'x_position': root_x,
            'y_position': root_y,
            'x_velocity': self.env.unwrapped.data.qvel[0],
            'y_velocity': self.env.unwrapped.data.qvel[1],
            'z_velocity': self.env.unwrapped.data.qvel[2],
            'quaternion_w': self.env.unwrapped.data.qpos[3],
        }
    
    def get_observation_info(self):
        """Helper method to understand observation space"""
        obs, _ = self.env.reset()
        print("\nObservation Analysis:")
        print(f"Total observation size: {len(obs)}")
        print(f"Actual height (qpos[2]): {self.env.unwrapped.data.qpos[2]}")
        print(f"Root quaternion w (qpos[3]): {self.env.unwrapped.data.qpos[3]}")
        print(f"Linear vel x,y,z (qvel[0:3]): {self.env.unwrapped.data.qvel[0:3]}")
        return obs
    
    def get_reward_analysis(self):
        """Analyze reward components over episode"""
        if not any(self.reward_history.values()):
            return None
            
        analysis = {}
        for component, values in self.reward_history.items():
            if values:
                analysis[component] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'total': np.sum(values)
                }
        return analysis


def make_standing_env(render_mode=None, config=None):
    """Factory function to create standing environment"""
    return StandingEnv(render_mode=render_mode, config=config)


def test_environment():
    """Test the fixed environment"""
    print("Testing OPTIMIZED Humanoid Standing Environment")
    print("=" * 60)
    
    # Test with random policy
    env = make_standing_env(render_mode=None, config=None)
    
    # Show observation info
    env.get_observation_info()
    
    obs, info = env.reset()
    total_reward = 0
    
    print("\nRunning 200 steps with random policy...")
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 50 == 0:
            print(f"\nStep {step}:")
            print(f"  Height: {info['height']:.3f} (target: 1.4)")
            print(f"  Quaternion w: {info['quaternion_w']:.3f}")
            print(f"  Reward: {reward:.3f}")
            print(f"  Total reward: {total_reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step} ({'terminated' if terminated else 'truncated'})")
            break
    
    # Show reward analysis
    print("\n" + "=" * 60)
    print("Reward Component Analysis:")
    print("=" * 60)
    analysis = env.get_reward_analysis()
    if analysis:
        for component, stats in analysis.items():
            print(f"\n{component.upper()}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Total contribution: {stats['total']:.2f}")
            print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    
    env.close()
    print(f"\nFinal total reward: {total_reward:.2f}")


if __name__ == "__main__":
    test_environment()