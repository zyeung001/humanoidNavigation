# standing_env.py

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium.spaces import Box


class StandingEnv(gym.Wrapper):
    """Wrapper for MuJoCo Humanoid environment optimized for standing task

    Humanoid-v5 with exclude_current_positions_from_observation=False
    adds 2 extra dimensions (x,y position) that aren't reflected in observation_space
    until after the first reset. We account for this manually.
    """
    
    def __init__(self, render_mode: Optional[str] = None, config=None):
        env_id = "Humanoid-v5"
        print(f"Using {env_id} for standing task")
        
        # Create base environment
        env = gym.make(
            env_id, 
            render_mode=render_mode,
            exclude_current_positions_from_observation=False  # Adds +2 dims (x,y)
        )
        super().__init__(env)
        
        # Configuration
        self.cfg = config or {}
        self.base_target_height = 1.4
        self.max_episode_steps = self.cfg.get('max_episode_steps', 5000)
        self.current_step = 0
        
        self.domain_rand = self.cfg.get('domain_rand', False)
        self.rand_mass_range = self.cfg.get('rand_mass_range', [0.95, 1.05])
        self.rand_friction_range = self.cfg.get('rand_friction_range', [0.95, 1.05])
        
        self.reward_history = {
            'height': [], 'upright': [], 'velocity': [], 
            'angular': [], 'position': [], 'control': []
        }

        # ======== Enhanced controls (all optional via config) ========
        # Action preprocessing
        self.enable_action_smoothing = bool(self.cfg.get('action_smoothing', False))
        self.action_smoothing_tau = float(self.cfg.get('action_smoothing_tau', 0.15))
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
        
        # Humanoid-v5 base observation_space.shape[0] = 350
        base_obs_from_space = int(env.observation_space.shape[0])  # 350
        
        # Actual observations have 15 MORE dimensions than observation_space reports
        # This is due to exclude_current_positions_from_observation=False quirk in Gymnasium
        base_obs_dim = base_obs_from_space + 15  # Actual observations are 365 dims
        
        # Calculate dimension after all processing steps
        extra_dim = 6 if self.include_com else 0  # COM pos (3) + COM vel (3)
        feature_dim = base_obs_dim + extra_dim
        
        if self.enable_history:
            # History stacking multiplies dimension
            total_dim = feature_dim * self.history_len
        else:
            total_dim = feature_dim
        
        # FREEZE observation dimension now (before VecNormalize initialization)
        self.frozen_obs_dim = total_dim
        
        # Declare observation space with frozen dimension
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.frozen_obs_dim,), 
            dtype=np.float32
        )
        
        print(f"Observation space configuration:")
        print(f"  Base from env.observation_space: {base_obs_from_space}")
        print(f"  + Position inclusion adjustment: +21 → {base_obs_dim}")
        print(f"  + COM features: {extra_dim} → {feature_dim}")
        print(f"  × History stack: {self.history_len if self.enable_history else 1}")
        print(f"  = FROZEN dimension: {self.frozen_obs_dim}")
        print(f"  Feature normalization: {self.feature_norm}")
    
    def reset(self, seed: Optional[int] = None): 
        observation, info = self.env.reset(seed=seed)
        
        default_height = self.env.unwrapped.data.qpos[2]
        
        self.current_step = 0
        self.prev_height = default_height
        self.target_height = self.base_target_height
        self.prev_action[:] = 0.0
        self.obs_history = []  # Clear history
        
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
            
            # Randomize geom friction
            original_friction = self.env.unwrapped.model.geom_friction.copy()
            self.env.unwrapped.model.geom_friction[:, 0] *= np.random.uniform(
                self.rand_friction_range[0], self.rand_friction_range[1],
                size=self.env.unwrapped.model.geom_friction.shape[0]
            )
        
        # Process observation with guaranteed dimension
        if self.enable_history or self.include_com or self.feature_norm:
            observation = self._process_observation(observation)

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
        
        # Process observation with dimension verification
        if self.enable_history or self.include_com or self.feature_norm:
            observation = self._process_observation(observation)

        return observation, reward, terminated, truncated, info
        
    def _compute_task_reward(self, obs, base_reward, info, action):
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
        if height < 1.0:
            height_reward = -50.0 + 40.0 * np.clip(height / 1.0, 0.0, 1.0)  
        elif height < 1.2:
            height_reward = -10.0 + 15.0 * (height - 1.0) / 0.2  
        elif height < 1.35:
            height_reward = 5.0 + 20.0 * (height - 1.2) / 0.15 
        elif height < 1.45:  # Peak reward zone around 1.40m
            height_reward = 25.0 + 25.0 * (1.0 - abs(height - 1.40) / 0.05)  
        elif height < 1.6:
            height_reward = 40.0 - 15.0 * (height - 1.45) / 0.15  
        else:
            height_reward = 25.0 - 20.0 * np.clip((height - 1.6) / 0.2, 0.0, 1.0)  
        
        # ========== UPRIGHT ORIENTATION REWARD  ==========
        upright_error = 1.0 - abs(quat[0])
        if height >= 1.2:
            upright_reward = 12.0 * np.exp(-8.0 * upright_error**2)  
        elif height >= 1.0:
            upright_reward = 7.0 * np.exp(-8.0 * upright_error**2)  
        else:
            upright_reward = 0.0
        
        # ========== STABILITY REWARD (RESCALED) ==========
        angular_momentum = np.sum(np.square(angular_vel))
        if height >= 1.3:
            stability_reward = 7.0 * np.exp(-2.0 * angular_momentum)
        elif height >= 1.2:
            stability_reward = 5.0 * np.exp(-2.0 * angular_momentum) 
        elif height >= 1.0:
            stability_reward = 2.5 * np.exp(-2.0 * angular_momentum)
        else:
            stability_reward = 0.0
        
        # ========== SMOOTHNESS REWARD ==========
        joint_velocity_magnitude = np.sum(np.square(joint_vel))
        smoothness_reward = 2.5 * np.exp(-0.1 * joint_velocity_magnitude) 
        
        # ========== CONTROL COST ========== 
        control_cost = -0.005 * np.sum(np.square(action))
        
        # ========== HEIGHT MAINTENANCE REWARD  ==========
        # Reward for maintaining height, penalty for losing it
        height_velocity = height - self.prev_height if hasattr(self, 'prev_height') else 0.0
        if height_velocity < -0.003:  # Losing height
            height_maintenance = -150.0 * abs(height_velocity)  
        elif abs(height_velocity) < 0.003:  
            height_maintenance = 5.0  
        else:  # Gaining height (not penalized)
            height_maintenance = 0.0
        
        # ========== VELOCITY PENALTY  ==========
        speed = np.linalg.norm(linear_vel)
        velocity_penalty = -1.0 * np.clip(speed - 1.0, 0.0, 2.0)  
        
        # ========== SPARSE BONUS (RESCALED) ==========
        sustained_bonus = 0.0
        if self.current_step > 0 and self.current_step % 100 == 0:
           
            if height_error < 0.08 and upright_error < 0.08 and height >= 1.32:
                sustained_bonus = 100.0
        
        # ========== TERMINATION PENALTY ==========
        termination_penalty = 0.0
        
        # ========== TERMINATION CONDITIONS (check before penalty) ==========
        terminate = (
            height < 0.75 or
            height > 2.0 or
            abs(quat[0]) < 0.6
        )
        
        # Apply termination penalty if about to terminate
        if terminate:
            # Penalty proportional to how far from end of episode
            steps_remaining = self.max_episode_steps - self.current_step
            termination_penalty = -100.0 * (steps_remaining / self.max_episode_steps)  # Up to -100
        
        # ========== TOTAL REWARD ==========
        total_reward = (
            height_reward +
            upright_reward +
            stability_reward +
            smoothness_reward +
            control_cost +
            height_maintenance +
            velocity_penalty +
            sustained_bonus +
            termination_penalty
        )
        
        # Update previous height for next step
        self.prev_height = height
        
        # ========== TRACK REWARD COMPONENTS ==========
        self.reward_history['height'].append(height_reward)
        self.reward_history['upright'].append(upright_reward)
        self.reward_history['velocity'].append(stability_reward)
        self.reward_history['control'].append(control_cost)
        
        # ========== DEBUG LOGGING  ==========
        if self.current_step % 500 == 0:
            print(f"Step {self.current_step:4d}: "
                f"h={height:.3f} (err={height_error:.3f}), "
                f"quat_w={quat[0]:.3f}, "
                f"r={total_reward:6.1f} "
                f"[h={height_reward:6.1f}, u={upright_reward:4.1f}, "
                f"stab={stability_reward:4.1f}, bonus={sustained_bonus:.0f}]")
        
        return total_reward, terminate

    # ======== Action and Observation Processing ========

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Process actions with optional smoothing, symmetry, and PD control."""
        if self.enable_action_symmetry:
            half = action.shape[-1] // 2
            if half > 0:
                left = action[:half]
                right = action[-half:]
                mean_lr = 0.5 * (left + right)
                action[:half] = mean_lr
                action[-half:] = mean_lr

        if self.enable_pd_assist and (self.pd_kp > 0.0 or self.pd_kd > 0.0):
            try:
                qpos = self.env.unwrapped.data.qpos[7:7+action.shape[-1]]
                qvel = self.env.unwrapped.data.qvel[6:6+action.shape[-1]]
                pd = (-self.pd_kp * qpos) + (-self.pd_kd * qvel)
                action = np.clip(action + pd, -1.0, 1.0)
            except Exception:
                pass

        if self.enable_action_smoothing:
            tau = np.clip(self.action_smoothing_tau, 0.0, 1.0)
            action = (1.0 - tau) * self.prev_action + tau * action

        low, high = self.env.action_space.low, self.env.action_space.high
        action = np.clip(action, low, high)
        self.prev_action = action.copy()
        return action

    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process observation with correct dimension handling."""
        features = [obs]

        # Add COM features if enabled
        if self.include_com:
            try:
                com_pos = self.env.unwrapped.data.subtree_com[0]
                com_vel = self.env.unwrapped.data.cdof_dot[:3] if hasattr(self.env.unwrapped.data, 'cdof_dot') else self.env.unwrapped.data.qvel[:3]
                features.append(np.asarray(com_pos, dtype=np.float32))
                features.append(np.asarray(com_vel, dtype=np.float32))
            except Exception as e:
                print(f"Warning: Failed to add COM features: {e}")

        # Concatenate base + COM features
        feat_vec = np.concatenate([np.atleast_1d(f).ravel() for f in features]).astype(np.float32)

        # Feature normalization
        if self.feature_norm:
            feat_vec = np.tanh(feat_vec * 0.1)

        # History stacking
        if self.enable_history:
            self.obs_history.append(feat_vec)
            
            if len(self.obs_history) > self.history_len:
                self.obs_history = self.obs_history[-self.history_len:]
            
            if len(self.obs_history) < self.history_len:
                pad_count = self.history_len - len(self.obs_history)
                padded = [np.zeros_like(feat_vec) for _ in range(pad_count)] + self.obs_history
            else:
                padded = self.obs_history
            
            feat_vec = np.concatenate(padded, axis=0)

        # Handle dimension mismatch gracefully
        current_dim = feat_vec.shape[0]
        if current_dim != self.frozen_obs_dim:
            if current_dim > self.frozen_obs_dim:
                # Truncate if larger
                feat_vec = feat_vec[:self.frozen_obs_dim]
            else:
                # Pad if smaller
                pad = np.zeros((self.frozen_obs_dim - current_dim,), dtype=np.float32)
                feat_vec = np.concatenate([feat_vec, pad], axis=0)

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
        print(f"Raw observation size: {len(obs)}")
        print(f"Frozen dimension: {self.frozen_obs_dim}")
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
    return StandingEnv(render_mode=render_mode, config=config)


def test_environment():
    print("Testing Humanoid Standing Environment")
    print("=" * 60)
    
    config = {
        'obs_history': 4,
        'obs_include_com': True,
        'obs_feature_norm': True,
        'action_smoothing': True,
        'action_smoothing_tau': 0.2,
    }
    
    env = make_standing_env(render_mode=None, config=config)
    
    obs = env.get_observation_info()
    
    obs, info = env.reset()
    print(f"\n✓ Reset observation shape: {obs.shape}")
    print(f"✓ Expected frozen dimension: {env.frozen_obs_dim}")
    
    print("\nRunning 200 steps...")
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step == 0:
            print(f"First step - obs shape: {obs.shape}, expected: {env.frozen_obs_dim}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break
    
    env.close()
    print(f"\n✓ Test completed!")


if __name__ == "__main__":
    test_environment()