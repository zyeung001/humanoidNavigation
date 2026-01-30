# transfer_utils.py
"""
Transfer learning utilities for standing → walking policy transfer.

Addresses the key issues in transfer learning:
1. VecNormalize dimension mismatch (1484 → 1496 dims)
2. Command feature weight initialization
3. Normalization statistics extension
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


class VecNormalizeExtender:
    """
    Extends VecNormalize statistics from standing (1484-dim) to walking (1496-dim).
    
    The issue: Standing VecNormalize trained on 1484-dim observations cannot be 
    directly loaded for walking (1496-dim). When load fails, SB3 creates fresh 
    VecNormalize with random statistics, which corrupts transferred weights.
    
    Solution: Manually extend the running mean/var statistics to the new dimension,
    using sensible defaults for the new command feature dimensions.
    
    Observation Layout (per frame, before history stacking):
    - Base obs: 365 dims
    - COM pos:  3 dims  
    - COM vel:  3 dims
    - Commands: 3 dims (vx, vy, yaw_rate) - NEW for walking
    Total per frame: 374 dims (standing: 371)
    With 4-frame history: 1496 dims (standing: 1484)
    """
    
    def __init__(
        self,
        standing_vecnorm_path: str,
        walking_env,
        command_mean: float = 0.0,
        command_var: float = 1.0,
    ):
        """
        Initialize the VecNormalize extender.
        
        Args:
            standing_vecnorm_path: Path to standing VecNormalize pickle
            walking_env: The walking VecEnv (for dimension validation)
            command_mean: Default mean for command features (0.0 is sensible)
            command_var: Default variance for command features 
                        (1.0 means commands are roughly in [-1, 1] range)
        """
        self.standing_path = Path(standing_vecnorm_path)
        self.walking_env = walking_env
        self.command_mean = command_mean
        self.command_var = command_var

        # ========== NEW DIMENSION STRUCTURE ==========
        # Walking now has: stacked body (1484) + command block ONCE (9) = 1493
        # (Previously was: (body + cmd) * 4 = 1496)
        self.standing_per_frame = 371  # base(365) + com(6)
        self.history_len = 4
        self.standing_total = self.standing_per_frame * self.history_len  # 1484

        # Walking: stacked body + command block (appended once, not per frame)
        # Command block: [vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual,
        #                 err_vx, err_vy, err_speed, err_angle] = 9 dims
        self.command_block_dim = 9
        self.walking_total = self.standing_total + self.command_block_dim  # 1493
        
    def extend(self) -> VecNormalize:
        """
        Create walking VecNormalize with extended statistics from standing.
        
        Returns:
            VecNormalize wrapper for walking environment with extended stats
        """
        if not self.standing_path.exists():
            print(f"⚠ Standing VecNormalize not found at {self.standing_path}")
            print("  Creating fresh VecNormalize for walking (no transfer)")
            return self._create_fresh_vecnorm()
        
        # Load standing VecNormalize data
        with open(self.standing_path, 'rb') as f:
            standing_data = pickle.load(f)
        
        # Handle both VecNormalize object and dictionary formats
        if isinstance(standing_data, dict):
            # Dictionary format (older save format)
            standing_mean = standing_data['obs_rms']['mean']
            standing_var = standing_data['obs_rms']['var']
            standing_count = standing_data['obs_rms']['count']
            gamma = standing_data.get('gamma', 0.995)
            ret_rms_data = standing_data.get('ret_rms', None)
        elif hasattr(standing_data, 'obs_rms'):
            # VecNormalize object (current SB3 format)
            standing_mean = standing_data.obs_rms.mean.copy()
            standing_var = standing_data.obs_rms.var.copy()
            standing_count = standing_data.obs_rms.count
            gamma = getattr(standing_data, 'gamma', 0.995)
            ret_rms_data = standing_data.ret_rms if hasattr(standing_data, 'ret_rms') else None
        else:
            print(f"⚠ Unknown VecNormalize format: {type(standing_data)}")
            print("  Creating fresh VecNormalize for walking")
            return self._create_fresh_vecnorm()
        
        print(f"✓ Loaded standing VecNormalize statistics")
        print(f"  Standing dimension: {len(standing_mean)}")
        print(f"  Sample count: {standing_count:,.0f}")
        
        # Validate dimensions
        expected_walking_dim = self.walking_env.observation_space.shape[0]
        if len(standing_mean) != self.standing_total:
            print(f"⚠ Unexpected standing dimension: {len(standing_mean)} (expected {self.standing_total})")
            # Adjust expected dimensions based on actual
            actual_standing_per_frame = len(standing_mean) // self.history_len
            print(f"  Actual standing per-frame: {actual_standing_per_frame}")
        
        # Extend statistics to walking dimension
        walking_mean, walking_var = self._extend_statistics(
            standing_mean, standing_var
        )
        
        print(f"  Extended to walking dimension: {len(walking_mean)}")
        
        # Create walking VecNormalize and inject extended statistics
        walking_vecnorm = VecNormalize(
            self.walking_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
        )
        
        # Inject extended statistics
        walking_vecnorm.obs_rms.mean = walking_mean
        walking_vecnorm.obs_rms.var = walking_var
        walking_vecnorm.obs_rms.count = standing_count
        
        # Copy reward normalization statistics
        if ret_rms_data is not None:
            if isinstance(ret_rms_data, dict):
                walking_vecnorm.ret_rms.mean = ret_rms_data['mean']
                walking_vecnorm.ret_rms.var = ret_rms_data['var']
                walking_vecnorm.ret_rms.count = ret_rms_data['count']
            elif hasattr(ret_rms_data, 'mean'):
                # RunningMeanStd object
                walking_vecnorm.ret_rms.mean = ret_rms_data.mean
                walking_vecnorm.ret_rms.var = ret_rms_data.var
                walking_vecnorm.ret_rms.count = ret_rms_data.count
        
        # Mark as not needing initial update (already has good statistics)
        walking_vecnorm.training = True
        
        print(f"✓ Extended VecNormalize created successfully")
        self._verify_statistics(walking_vecnorm)
        
        return walking_vecnorm
    
    def _extend_statistics(
        self,
        standing_mean: np.ndarray,
        standing_var: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extend standing statistics to walking dimensions.

        NEW STRUCTURE (command block appended once, not per-frame):
        - Stacked body: [body_t-3, body_t-2, body_t-1, body_t] = 1484 dims
        - Command block: [vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual,
                         err_vx, err_vy, err_speed, err_angle] = 9 dims
        - Total: 1493 dims

        Since standing uses the same body structure (1484 dims), we can
        copy standing stats directly to the body portion and add fresh
        command block stats at the end.
        """
        # Initialize walking arrays
        walking_mean = np.zeros(self.walking_total, dtype=np.float64)
        walking_var = np.ones(self.walking_total, dtype=np.float64)

        # Copy standing stats directly to body portion (first 1484 dims)
        # Standing and walking share the same body feature structure
        min_len = min(len(standing_mean), self.standing_total)
        walking_mean[:min_len] = standing_mean[:min_len]
        walking_var[:min_len] = standing_var[:min_len]

        # Add command block statistics (last 9 dims)
        # Command block is normalized to [-1, 1], so mean=0, var=1 is appropriate
        cmd_start = self.standing_total
        walking_mean[cmd_start:] = self.command_mean  # 0.0
        walking_var[cmd_start:] = self.command_var    # 1.0

        return walking_mean, walking_var
    
    def _create_fresh_vecnorm(self) -> VecNormalize:
        """Create fresh VecNormalize without transfer."""
        return VecNormalize(
            self.walking_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.995,
        )
    
    def _verify_statistics(self, vecnorm: VecNormalize):
        """Print verification info about the VecNormalize statistics."""
        mean = vecnorm.obs_rms.mean
        var = vecnorm.obs_rms.var

        # Check first frame's base obs
        print(f"  Base obs mean range: [{mean[:20].min():.3f}, {mean[:20].max():.3f}]")
        print(f"  Base obs var range: [{var[:20].min():.3f}, {var[:20].max():.3f}]")

        # Check command block (last 9 dims, appended once)
        cmd_start = self.standing_total  # 1484
        print(f"  Command block mean: {mean[cmd_start:]}")
        print(f"  Command block var: {var[cmd_start:]}")


class PolicyTransfer:
    """
    Transfers policy weights from standing to walking model with proper initialization.
    
    Key fix: The command feature weights (12 new dims = 3 per frame × 4 frames)
    were initialized with randn() * 0.01, which is essentially noise.
    
    Better approaches:
    1. Zero initialization: Policy ignores commands initially, learns to use them
    2. Xavier/Kaiming initialization: Proper scale for gradient flow
    3. Copy from similar features: Use velocity feature weights as template
    """
    
    # Command weight initialization strategies
    INIT_ZERO = "zero"           # Zero out - policy ignores commands initially
    INIT_XAVIER = "xavier"       # Xavier uniform - good gradient flow
    INIT_KAIMING = "kaiming"     # Kaiming/He - best for ReLU-like activations  
    INIT_SMALL_NOISE = "small"   # Small random (0.1 scale, not 0.01)
    INIT_FROM_VELOCITY = "velocity"  # Copy from velocity observation weights
    
    def __init__(
        self,
        standing_model: PPO,
        walking_model: PPO,
        init_strategy: str = "xavier",
        device: str = "cuda",
        command_weight_scale: float = 5.0,
    ):
        """
        Initialize policy transfer.

        Args:
            standing_model: Pre-trained standing PPO model
            walking_model: Fresh walking PPO model to transfer into
            init_strategy: How to initialize new command feature weights
                - "zero": Zero initialization (conservative)
                - "xavier": Xavier uniform (recommended)
                - "kaiming": He/Kaiming (for SiLU activation)
                - "small": Small noise (0.1 scale)
                - "velocity": Copy from velocity features
            device: PyTorch device
            command_weight_scale: Multiplier for new command weights to make
                them visible against 1484 trained standing weights
        """
        self.standing_model = standing_model
        self.walking_model = walking_model
        self.init_strategy = init_strategy
        self.device = device
        self.command_weight_scale = command_weight_scale

        # Dimension info (NEW structure: body stacked + command block once)
        self.standing_dim = 1484  # Body features × 4 frames
        self.walking_dim = 1493   # Body (1484) + command block (9)
        self.new_dims = 9         # Command block: [vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual, err_vx, err_vy, err_speed, err_angle]
        
    def transfer(self) -> Dict[str, Any]:
        """
        Transfer weights from standing to walking policy.
        
        Returns:
            Dictionary with transfer statistics
        """
        standing_state = self.standing_model.policy.state_dict()
        walking_state = self.walking_model.policy.state_dict()
        
        stats = {
            'transferred': 0,
            'partial_transfer': 0,
            'skipped': 0,
            'new_layers': 0,
            'first_layer_init': None,
        }
        
        for key in walking_state.keys():
            if key not in standing_state:
                stats['new_layers'] += 1
                continue
            
            standing_tensor = standing_state[key]
            walking_tensor = walking_state[key]
            
            if standing_tensor.shape == walking_tensor.shape:
                # Exact match - direct copy
                walking_state[key] = standing_tensor.clone()
                stats['transferred'] += 1
                
            elif len(standing_tensor.shape) == 2 and len(walking_tensor.shape) == 2:
                # 2D weight matrix - handle dimension mismatch
                walking_state[key] = self._transfer_2d_weights(
                    key, standing_tensor, walking_tensor, stats
                )
                stats['partial_transfer'] += 1
                
            elif len(standing_tensor.shape) == 1 and len(walking_tensor.shape) == 1:
                # 1D bias vector
                walking_state[key] = self._transfer_1d_weights(
                    standing_tensor, walking_tensor
                )
                stats['partial_transfer'] += 1
            else:
                stats['skipped'] += 1
        
        # Load transferred state
        self.walking_model.policy.load_state_dict(walking_state)

        # Re-initialize value function for new task
        # Standing value function predicts standing-task returns, causing
        # value estimation errors that destabilize early walking training
        import torch.nn as nn
        reinit_count = 0
        with torch.no_grad():
            for name, param in self.walking_model.policy.named_parameters():
                if 'value' in name.lower() or 'vf' in name.lower():
                    if 'bias' in name.lower():
                        nn.init.constant_(param, 0.0)
                        reinit_count += 1
                    elif param.dim() >= 2:
                        nn.init.orthogonal_(param, gain=1.0)
                        reinit_count += 1

        stats['value_fn_reinit'] = reinit_count
        print(f"  Value function re-initialized: {reinit_count} parameters")

        return stats
    
    def _transfer_2d_weights(
        self,
        key: str,
        standing: torch.Tensor,
        walking: torch.Tensor,
        stats: Dict,
    ) -> torch.Tensor:
        """
        Transfer 2D weight matrix with proper initialization for new dimensions.
        
        For the first layer (input layer), we need to handle the dimension mismatch
        where walking has 12 extra input features (command velocities × 4 frames).
        """
        out_dim = walking.shape[0]
        in_dim_standing = standing.shape[1]
        in_dim_walking = walking.shape[1]
        
        # Create new weight tensor
        new_weights = walking.clone()
        
        # Copy overlapping dimensions
        min_out = min(standing.shape[0], walking.shape[0])
        min_in = min(in_dim_standing, in_dim_walking)
        new_weights[:min_out, :min_in] = standing[:min_out, :min_in]
        
        # Initialize new input dimensions (command features)
        if in_dim_walking > in_dim_standing:
            new_in_dims = in_dim_walking - in_dim_standing
            new_weights_section = new_weights[:, -new_in_dims:]
            
            if self.init_strategy == self.INIT_ZERO:
                # Zero initialization - policy ignores commands initially
                new_weights[:, -new_in_dims:] = 0.0
                init_scale = 0.0
                
            elif self.init_strategy == self.INIT_XAVIER:
                # Xavier uniform initialization
                fan_in = in_dim_walking
                fan_out = out_dim
                std = np.sqrt(2.0 / (fan_in + fan_out))
                new_weights[:, -new_in_dims:] = torch.randn_like(
                    new_weights_section, device=self.device
                ) * std
                init_scale = std
                
            elif self.init_strategy == self.INIT_KAIMING:
                # Kaiming/He initialization (for SiLU activation)
                std = np.sqrt(2.0 / in_dim_walking)
                new_weights[:, -new_in_dims:] = torch.randn_like(
                    new_weights_section, device=self.device
                ) * std
                init_scale = std
                
            elif self.init_strategy == self.INIT_SMALL_NOISE:
                # Small random initialization (10x larger than original 0.01)
                new_weights[:, -new_in_dims:] = torch.randn_like(
                    new_weights_section, device=self.device
                ) * 0.1
                init_scale = 0.1
                
            elif self.init_strategy == self.INIT_FROM_VELOCITY:
                # Copy weights from velocity features (similar semantic meaning)
                # Velocity features are at indices 0:3 in base obs
                velocity_weights = standing[:, :3]
                # Tile to cover all new dims
                n_copies = new_in_dims // 3 + 1
                tiled = velocity_weights.repeat(1, n_copies)[:, :new_in_dims]
                # Add small noise to break symmetry
                new_weights[:, -new_in_dims:] = tiled + torch.randn_like(
                    tiled, device=self.device
                ) * 0.01
                init_scale = float(velocity_weights.std().item())
            else:
                raise ValueError(f"Unknown init strategy: {self.init_strategy}")

            # Boost command weights to be visible against 1484 trained standing weights
            new_weights[:, -new_in_dims:] *= self.command_weight_scale

            # Check if this is the first layer
            if "mlp_extractor" in key and ("0" in key or "policy_net" in key):
                stats['first_layer_init'] = {
                    'strategy': self.init_strategy,
                    'scale': init_scale * self.command_weight_scale,
                    'new_dims': new_in_dims,
                }
                print(f"  First layer command weights initialized with {self.init_strategy}")
                print(f"    Base scale: {init_scale:.4f}, boosted by {self.command_weight_scale}x")
                print(f"    Effective scale: {init_scale * self.command_weight_scale:.4f}")
                print(f"    New input dims: {new_in_dims}")
        
        return new_weights
    
    def _transfer_1d_weights(
        self,
        standing: torch.Tensor,
        walking: torch.Tensor,
    ) -> torch.Tensor:
        """Transfer 1D bias vectors."""
        new_bias = walking.clone()
        min_size = min(len(standing), len(walking))
        new_bias[:min_size] = standing[:min_size]
        return new_bias


class WarmupCollector:
    """
    Collects observation statistics before starting actual training.
    
    Issue: When VecNormalize is fresh or partially initialized, the first
    few thousand steps have incorrect normalization, causing the policy
    to receive garbage observations.
    
    Solution: Run a warmup period with random actions to collect statistics
    before loading transferred weights.
    """
    
    def __init__(
        self,
        env: VecNormalize,
        warmup_steps: int = 10000,
        verbose: bool = True,
    ):
        """
        Initialize warmup collector.
        
        Args:
            env: VecNormalize environment
            warmup_steps: Number of steps to collect
            verbose: Print progress
        """
        self.env = env
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        
    def collect(self):
        """
        Run warmup data collection with random actions.
        
        After this, VecNormalize will have stable statistics for the
        walking observation space, including command features.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"WARMUP: Collecting {self.warmup_steps:,} observation samples")
            print(f"{'='*60}")
        
        obs = self.env.reset()
        collected = 0
        episodes = 0
        n_envs = self.env.num_envs

        while collected < self.warmup_steps:
            # Random actions for ALL parallel environments (FIX: was only generating 1 action)
            actions = np.array([self.env.action_space.sample() for _ in range(n_envs)])
            obs, reward, done, info = self.env.step(actions)
            collected += n_envs  # FIX: Count observations from all envs, not just 1

            if done.any():
                episodes += done.sum()  # Count all completed episodes
                # VecEnv auto-resets, no need to call reset()

            if self.verbose and collected % 2000 < n_envs:  # Trigger roughly every 2000
                mean_range = (self.env.obs_rms.mean.min(), self.env.obs_rms.mean.max())
                var_range = (self.env.obs_rms.var.min(), self.env.obs_rms.var.max())
                print(f"  Step {collected:,}: mean=[{mean_range[0]:.3f}, {mean_range[1]:.3f}], "
                      f"var=[{var_range[0]:.3f}, {var_range[1]:.3f}]")
        
        # Prevent variance collapse for command dimensions
        # During Stage 0 warmup, near-constant fixed commands cause variance
        # to collapse to ~0.001, making commands invisible to the policy
        # Commands are pre-normalized to [-1, 1], so identity stats (mean=0, var=1) are correct
        body_dim = 1484
        self.env.obs_rms.var[body_dim:] = 1.0
        self.env.obs_rms.mean[body_dim:] = 0.0

        if self.verbose:
            print(f"\n✓ Warmup complete: {collected:,} steps, {episodes} episodes")
            print(f"  Final count: {self.env.obs_rms.count:.0f}")

            # Verify command block statistics (last 9 dims, pinned to identity)
            cmd_mean = self.env.obs_rms.mean[body_dim:]
            cmd_var = self.env.obs_rms.var[body_dim:]
            print(f"  Command block mean (pinned): {cmd_mean}")
            print(f"  Command block var (pinned): {cmd_var}")

        return self.env


def transfer_standing_to_walking(
    standing_model_path: str,
    standing_vecnorm_path: str,
    walking_env,
    walking_model_kwargs: Dict[str, Any],
    device: str = "cuda",
    init_strategy: str = "velocity",  # FIX: Use velocity weights as template (not xavier)
    warmup_steps: int = 10000,
) -> Tuple[PPO, VecNormalize]:
    """
    Complete transfer from standing to walking model.
    
    This function handles:
    1. VecNormalize extension
    2. Warmup collection
    3. Policy weight transfer with proper initialization
    
    Args:
        standing_model_path: Path to standing model .zip
        standing_vecnorm_path: Path to standing VecNormalize .pkl
        walking_env: VecEnv for walking (not wrapped in VecNormalize yet)
        walking_model_kwargs: Kwargs for PPO initialization
        device: PyTorch device
        init_strategy: How to initialize command feature weights
        warmup_steps: Steps to collect for normalization (0 to skip)
        
    Returns:
        (walking_model, walking_vecnorm) ready for training
    """
    print(f"\n{'='*60}")
    print("TRANSFER LEARNING: Standing → Walking")
    print(f"{'='*60}")
    
    # Step 1: Extend VecNormalize
    print("\n[1/4] Extending VecNormalize statistics...")
    extender = VecNormalizeExtender(
        standing_vecnorm_path=standing_vecnorm_path,
        walking_env=walking_env,
        command_mean=0.0,
        command_var=1.0,  # Commands are in ~[-3, 3] range
    )
    walking_vecnorm = extender.extend()
    
    # Step 2: Warmup collection (optional but recommended)
    if warmup_steps > 0:
        print(f"\n[2/4] Collecting warmup samples...")
        collector = WarmupCollector(
            env=walking_vecnorm,
            warmup_steps=warmup_steps,
            verbose=True,
        )
        walking_vecnorm = collector.collect()
    else:
        print(f"\n[2/4] Skipping warmup (warmup_steps=0)")
    
    # Step 3: Load standing model and create walking model
    print(f"\n[3/4] Loading standing model and creating walking model...")
    standing_model = PPO.load(standing_model_path, device=device)
    print(f"  Standing obs space: {standing_model.observation_space.shape}")
    print(f"  Walking obs space: {walking_vecnorm.observation_space.shape}")
    
    # Create walking model
    walking_model = PPO(
        env=walking_vecnorm,
        device=device,
        **walking_model_kwargs
    )
    
    # Step 4: Transfer weights
    print(f"\n[4/4] Transferring policy weights...")
    transfer = PolicyTransfer(
        standing_model=standing_model,
        walking_model=walking_model,
        init_strategy=init_strategy,
        device=device,
        command_weight_scale=5.0,
    )
    stats = transfer.transfer()
    
    print(f"\n✓ Transfer complete:")
    print(f"  Exact match: {stats['transferred']} layers")
    print(f"  Partial transfer: {stats['partial_transfer']} layers")
    print(f"  Skipped: {stats['skipped']} layers")
    print(f"  New: {stats['new_layers']} layers")
    if stats['first_layer_init']:
        init_info = stats['first_layer_init']
        print(f"  Command weight init: {init_info['strategy']} (scale={init_info['scale']:.4f})")
    print(f"{'='*60}\n")
    
    return walking_model, walking_vecnorm


if __name__ == "__main__":
    print("Transfer utils module - run tests")

    # Test VecNormalizeExtender dimensions (NEW structure)
    print("\nDimension calculations (NEW structure):")
    print(f"  Body per-frame: 371 (365 base + 6 COM)")
    print(f"  Standing total: 371 × 4 = {371 * 4} (body only)")
    print(f"  Walking body: 371 × 4 = {371 * 4} (stacked)")
    print(f"  Command block: 9 (vx_cmd, vy_cmd, yaw_cmd, vx_actual, vy_actual, err_vx, err_vy, err_speed, err_angle)")
    print(f"  Walking total: {371 * 4} + 9 = {371 * 4 + 9}")
