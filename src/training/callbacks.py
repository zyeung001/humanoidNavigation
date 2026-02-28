# wandb_callbacks.py
"""
WandB logging callbacks for humanoid velocity tracking training.

Provides comprehensive logging of:
- Velocity tracking metrics
- Action smoothness (jerk)
- Curriculum progression
- Episode statistics
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging disabled.")

from stable_baselines3.common.callbacks import BaseCallback


class VelocityTrackingWandBCallback(BaseCallback):
    """
    WandB callback for velocity tracking training metrics.
    
    Logs:
    - Velocity error (total, x, y components)
    - Action smoothness (jerk penalty)
    - Curriculum stage progression
    - Episode rewards and lengths
    - Action magnitudes
    
    Example:
        callback = VelocityTrackingWandBCallback(
            log_freq=1000,
            eval_freq=100000
        )
        model.learn(total_timesteps=1000000, callback=callback)
    """
    
    def __init__(
        self,
        log_freq: int = 1000,
        eval_freq: int = 100000,
        buffer_size: int = 1000,
        project_name: str = "humanoid_walking",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        verbose: int = 0
    ):
        """
        Initialize WandB callback.
        
        Args:
            log_freq: Steps between logging aggregated metrics
            eval_freq: Steps between evaluation logging
            buffer_size: Size of rolling buffer for metric aggregation
            project_name: WandB project name
            run_name: Optional run name (auto-generated if None)
            config: Training config to log
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.buffer_size = buffer_size
        
        # Metric buffers (rolling windows)
        self.velocity_errors = deque(maxlen=buffer_size)
        self.velocity_errors_x = deque(maxlen=buffer_size)
        self.velocity_errors_y = deque(maxlen=buffer_size)
        self.jerk_penalties = deque(maxlen=buffer_size)
        self.action_magnitudes = deque(maxlen=buffer_size)
        self.heights = deque(maxlen=buffer_size)
        
        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.curriculum_stages = deque(maxlen=100)
        
        # Curriculum tracking
        self.current_stage = 0
        self.stage_history = []
        
        # WandB initialization
        self._wandb_initialized = False
        self.project_name = project_name
        self.run_name = run_name
        self.config = config
    
    def _init_wandb(self):
        """Initialize WandB if not already done."""
        if not WANDB_AVAILABLE:
            return
        
        if not self._wandb_initialized and wandb.run is None:
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                reinit=True
            )
            self._wandb_initialized = True
    
    def _on_training_start(self):
        """Called when training starts."""
        self._init_wandb()
    
    def _on_step(self) -> bool:
        """
        Called after each environment step.
        
        Collects metrics from infos and logs aggregated statistics.
        """
        # Collect metrics from step infos
        infos = self.locals.get("infos", [])
        
        for info in infos:
            # Velocity tracking metrics
            if 'velocity_error' in info:
                self.velocity_errors.append(info['velocity_error'])
            
            if 'velocity_error_x' in info:
                self.velocity_errors_x.append(info['velocity_error_x'])
            elif 'commanded_vx' in info and 'x_velocity' in info:
                self.velocity_errors_x.append(
                    abs(info['commanded_vx'] - info['x_velocity'])
                )
            
            if 'velocity_error_y' in info:
                self.velocity_errors_y.append(info['velocity_error_y'])
            elif 'commanded_vy' in info and 'y_velocity' in info:
                self.velocity_errors_y.append(
                    abs(info['commanded_vy'] - info['y_velocity'])
                )
            
            # Action smoothness
            if 'jerk_penalty' in info:
                self.jerk_penalties.append(info['jerk_penalty'])
            
            if 'action_magnitude' in info:
                self.action_magnitudes.append(info['action_magnitude'])
            
            # Height tracking
            if 'height' in info:
                self.heights.append(info['height'])
            
            # Episode completion
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
                # Curriculum info
                if 'curriculum_stage' in info:
                    stage = info['curriculum_stage']
                    self.curriculum_stages.append(stage)
                    
                    if stage != self.current_stage:
                        self.current_stage = stage
                        self.stage_history.append({
                            'timestep': self.num_timesteps,
                            'stage': stage
                        })
        
        # Log aggregated metrics at intervals
        if self.num_timesteps % self.log_freq == 0:
            self._log_metrics()
        
        return True
    
    def _log_metrics(self):
        """Log aggregated metrics to WandB."""
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        
        log_dict = {
            "timesteps": self.num_timesteps,
        }
        
        # Velocity errors
        if self.velocity_errors:
            log_dict["train/velocity_error"] = np.mean(self.velocity_errors)
            log_dict["train/velocity_error_std"] = np.std(self.velocity_errors)
        
        if self.velocity_errors_x:
            log_dict["train/velocity_error_x"] = np.mean(self.velocity_errors_x)
        
        if self.velocity_errors_y:
            log_dict["train/velocity_error_y"] = np.mean(self.velocity_errors_y)
        
        # Action smoothness
        if self.jerk_penalties:
            log_dict["train/jerk_penalty"] = np.mean(self.jerk_penalties)
            log_dict["train/jerk_penalty_max"] = np.max(self.jerk_penalties)
        
        if self.action_magnitudes:
            log_dict["train/action_magnitude"] = np.mean(self.action_magnitudes)
        
        # Height
        if self.heights:
            log_dict["train/height_mean"] = np.mean(self.heights)
            log_dict["train/height_std"] = np.std(self.heights)
        
        # Episode stats
        if self.episode_rewards:
            log_dict["episode/reward_mean"] = np.mean(self.episode_rewards)
            log_dict["episode/reward_std"] = np.std(self.episode_rewards)
        
        if self.episode_lengths:
            log_dict["episode/length_mean"] = np.mean(self.episode_lengths)
            log_dict["episode/length_std"] = np.std(self.episode_lengths)
        
        # Curriculum
        if self.curriculum_stages:
            log_dict["curriculum/stage"] = self.current_stage
            log_dict["curriculum/avg_stage"] = np.mean(self.curriculum_stages)
        
        wandb.log(log_dict)
    
    def _on_training_end(self):
        """Called when training ends."""
        if WANDB_AVAILABLE and wandb.run is not None:
            # Log final summary
            summary = {
                "final/velocity_error": np.mean(self.velocity_errors) if self.velocity_errors else 0,
                "final/episode_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
                "final/curriculum_stage": self.current_stage,
                "final/total_timesteps": self.num_timesteps,
            }
            wandb.log(summary)
            
            # Log curriculum progression
            if self.stage_history:
                table = wandb.Table(columns=["timestep", "stage"])
                for entry in self.stage_history:
                    table.add_data(entry['timestep'], entry['stage'])
                wandb.log({"curriculum_progression": table})


class CurriculumWandBCallback(BaseCallback):
    """
    Focused callback for curriculum progression logging.
    
    Tracks stage advancement and metrics per stage.
    """
    
    def __init__(
        self,
        log_freq: int = 5000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        
        self.current_stage = 0
        self.stage_starts = {0: 0}
        self.stage_metrics = {}
    
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if 'curriculum_stage' in info:
                stage = info['curriculum_stage']
                
                if stage != self.current_stage:
                    # Record stage transition
                    self.stage_starts[stage] = self.num_timesteps
                    self.current_stage = stage
                    
                    if WANDB_AVAILABLE and wandb.run is not None:
                        wandb.log({
                            "curriculum/stage_advanced": stage,
                            "curriculum/advancement_timestep": self.num_timesteps,
                        })
                        
                        print(f"📈 WandB: Logged curriculum advancement to stage {stage}")
                
                # Track per-stage metrics
                if stage not in self.stage_metrics:
                    self.stage_metrics[stage] = {
                        'velocity_errors': [],
                        'episode_lengths': []
                    }
                
                if 'velocity_error' in info:
                    self.stage_metrics[stage]['velocity_errors'].append(
                        info['velocity_error']
                    )
                
                if 'episode' in info:
                    self.stage_metrics[stage]['episode_lengths'].append(
                        info['episode']['l']
                    )
        
        # Log stage metrics periodically
        if self.num_timesteps % self.log_freq == 0:
            self._log_stage_metrics()
        
        return True
    
    def _log_stage_metrics(self):
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        
        if self.current_stage in self.stage_metrics:
            metrics = self.stage_metrics[self.current_stage]
            
            log_dict = {}
            if metrics['velocity_errors']:
                recent = metrics['velocity_errors'][-100:]
                log_dict[f"stage_{self.current_stage}/velocity_error"] = np.mean(recent)
            
            if metrics['episode_lengths']:
                recent = metrics['episode_lengths'][-20:]
                log_dict[f"stage_{self.current_stage}/episode_length"] = np.mean(recent)
            
            if log_dict:
                wandb.log(log_dict)


class VideoRecordingCallback(BaseCallback):
    """
    Callback for recording evaluation videos to WandB.
    """
    
    def __init__(
        self,
        eval_env,
        video_freq: int = 500000,
        video_length: int = 500,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_freq = video_freq
        self.video_length = video_length
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.video_freq == 0 and self.num_timesteps > 0:
            self._record_video()
        return True
    
    def _record_video(self):
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        
        try:
            frames = []
            obs = self.eval_env.reset()[0]
            
            for _ in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, _ = self.eval_env.step(action)
                
                frame = self.eval_env.render()
                if frame is not None:
                    frames.append(frame)
                
                if done:
                    obs = self.eval_env.reset()[0]
            
            if frames:
                # Log video to wandb
                video = np.array(frames).transpose(0, 3, 1, 2)  # THWC -> TCHW
                wandb.log({
                    "video/evaluation": wandb.Video(
                        video, 
                        fps=30, 
                        format="mp4"
                    ),
                    "video/timestep": self.num_timesteps
                })
                print(f"📹 Video recorded at timestep {self.num_timesteps}")
                
        except Exception as e:
            print(f"Video recording failed: {e}")


class RewardBreakdownWandBCallback(BaseCallback):
    """
    WandB callback for comprehensive reward breakdown logging.

    Logs:
    - Individual reward components (mean/std)
    - Termination cause distribution
    - Behavior ratios (standing vs walking)
    - Standing exploit ratio (critical diagnostic)
    - Command effectiveness (achieved/commanded speed ratio)
    """

    def __init__(
        self,
        log_freq: int = 5000,
        buffer_size: int = 1000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.buffer_size = buffer_size

        # Reward component buffers
        self.reward_components: Dict[str, deque] = {}

        # Termination tracking
        self.termination_causes: deque = deque(maxlen=buffer_size)

        # Behavior tracking
        self.is_standing_buffer: deque = deque(maxlen=buffer_size)
        self.standing_penalty_buffer: deque = deque(maxlen=buffer_size)
        self.speed_ratio_buffer: deque = deque(maxlen=buffer_size)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            # Collect reward components
            for key, value in info.items():
                if key.startswith('reward/'):
                    component_name = key.replace('reward/', '')
                    if component_name not in self.reward_components:
                        self.reward_components[component_name] = deque(maxlen=self.buffer_size)
                    self.reward_components[component_name].append(value)

            # Collect termination causes
            if 'termination_cause' in info:
                self.termination_causes.append(info['termination_cause'])

            # Collect behavior metrics
            if 'behavior/is_standing' in info:
                self.is_standing_buffer.append(1 if info['behavior/is_standing'] else 0)
            if 'behavior/standing_penalty_applied' in info:
                self.standing_penalty_buffer.append(1 if info['behavior/standing_penalty_applied'] else 0)
            if 'behavior/speed_ratio' in info:
                self.speed_ratio_buffer.append(info['behavior/speed_ratio'])

        # Log at intervals
        if self.num_timesteps % self.log_freq == 0:
            self._log_metrics()

        return True

    def _log_metrics(self):
        if not WANDB_AVAILABLE or wandb.run is None:
            return

        log_dict = {"timesteps": self.num_timesteps}

        # Log reward components
        for component_name, values in self.reward_components.items():
            if values:
                log_dict[f"reward/{component_name}_mean"] = np.mean(values)
                log_dict[f"reward/{component_name}_std"] = np.std(values)

        # Log termination cause distribution
        if self.termination_causes:
            causes = list(self.termination_causes)
            cause_counts = {}
            for cause in causes:
                cause_counts[cause] = cause_counts.get(cause, 0) + 1
            total = len(causes)
            for cause, count in cause_counts.items():
                log_dict[f"termination/{cause}_ratio"] = count / total

        # Log behavior metrics
        if self.is_standing_buffer:
            standing_ratio = np.mean(self.is_standing_buffer)
            log_dict["behavior/standing_ratio"] = standing_ratio
            log_dict["behavior/standing_exploit_ratio"] = standing_ratio  # Critical diagnostic

        if self.standing_penalty_buffer:
            log_dict["behavior/standing_penalty_ratio"] = np.mean(self.standing_penalty_buffer)

        if self.speed_ratio_buffer:
            log_dict["behavior/speed_ratio_mean"] = np.mean(self.speed_ratio_buffer)
            log_dict["behavior/command_effectiveness"] = np.clip(np.mean(self.speed_ratio_buffer), 0, 1)

        wandb.log(log_dict)


def init_wandb_run(
    project: str = "humanoid_walking",
    name: Optional[str] = None,
    config: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
) -> bool:
    """
    Initialize a WandB run with common settings.
    
    Args:
        project: Project name
        name: Run name (auto-generated if None)
        config: Configuration to log
        tags: Tags for the run
        
    Returns:
        True if initialization succeeded
    """
    if not WANDB_AVAILABLE:
        print("WandB not available")
        return False
    
    try:
        wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            reinit=True
        )
        print(f"✓ WandB initialized: {wandb.run.name}")
        return True
    except Exception as e:
        print(f"WandB initialization failed: {e}")
        return False


def finish_wandb_run():
    """Finish and upload the current WandB run."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        print("✓ WandB run finished")

