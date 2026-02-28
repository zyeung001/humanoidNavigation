# model_manager.py
"""
Model and checkpoint management for humanoid training.

Features:
- Organized directory structure
- Best model tracking
- Checkpoint cleanup
- Config archiving
"""

import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import yaml


class ModelManager:
    """
    Manages model checkpoints and weights storage.
    
    Directory Structure:
        models/
        ├── {task}/
        │   ├── latest/
        │   │   ├── model.zip
        │   │   └── vecnorm.pkl
        │   ├── best/
        │   │   ├── model.zip
        │   │   └── vecnorm.pkl
        │   ├── checkpoints/
        │   │   └── stage_{n}/
        │   │       └── model_{timesteps}.zip
        │   └── final/
        │       └── model.zip
        └── configs/
            └── run_{timestamp}.yaml
    
    Example:
        manager = ModelManager("walking")
        manager.save_checkpoint(model, env, timesteps=1000000, stage=2)
        manager.save_best(model, env, velocity_error=0.15)
    """
    
    def __init__(
        self,
        task: str,
        base_dir: str = "models",
        max_checkpoints_per_stage: int = 5,
    ):
        """
        Initialize model manager.
        
        Args:
            task: Task name (e.g., 'walking', 'standing')
            base_dir: Base directory for all models
            max_checkpoints_per_stage: Max checkpoints to keep per curriculum stage
        """
        self.task = task
        self.base_dir = Path(base_dir)
        self.max_checkpoints_per_stage = max_checkpoints_per_stage
        
        # Directory paths
        self.task_dir = self.base_dir / task
        self.latest_dir = self.task_dir / "latest"
        self.best_dir = self.task_dir / "best"
        self.checkpoint_dir = self.task_dir / "checkpoints"
        self.final_dir = self.task_dir / "final"
        self.config_dir = self.base_dir / "configs"
        
        # Create directories
        for d in [self.latest_dir, self.best_dir, self.checkpoint_dir, 
                  self.final_dir, self.config_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Best model tracking
        self._best_metric = float('inf')
        self._best_timesteps = 0
        
        # Load existing best info if available
        self._load_best_info()
    
    def _load_best_info(self):
        """Load best model info from metadata file."""
        info_path = self.best_dir / "info.json"
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    self._best_metric = info.get('metric', float('inf'))
                    self._best_timesteps = info.get('timesteps', 0)
            except Exception:
                pass
    
    def _save_best_info(self, metric: float, timesteps: int, extra: Dict = None):
        """Save best model metadata."""
        info = {
            'metric': metric,
            'timesteps': timesteps,
            'timestamp': datetime.now().isoformat(),
            **(extra or {})
        }
        info_path = self.best_dir / "info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def save_latest(self, model, env, timesteps: int = None):
        """
        Save model and vecnorm to latest directory.
        
        Args:
            model: SB3 model to save
            env: VecNormalize environment
            timesteps: Current timesteps (for metadata)
        """
        model_path = self.latest_dir / "model"
        vecnorm_path = self.latest_dir / "vecnorm.pkl"
        
        model.save(str(model_path))
        try:
            env.save(str(vecnorm_path))
        except Exception as e:
            print(f"Warning: Could not save vecnorm: {e}")
        
        # Save metadata
        info = {
            'timesteps': timesteps or model.num_timesteps,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.latest_dir / "info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ Latest saved: {model_path}.zip")
    
    def save_checkpoint(
        self, 
        model, 
        env, 
        timesteps: int,
        stage: int = 0,
        velocity_error: float = None,
        extra_suffix: str = ""
    ):
        """
        Save a checkpoint organized by curriculum stage.
        
        Args:
            model: SB3 model
            env: VecNormalize environment
            timesteps: Current timesteps
            stage: Curriculum stage
            velocity_error: Optional velocity error for filename
            extra_suffix: Additional suffix for filename
        """
        stage_dir = self.checkpoint_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Build filename
        ts_str = f"{timesteps // 1000}k" if timesteps >= 1000 else str(timesteps)
        if velocity_error is not None:
            filename = f"model_{ts_str}_err{velocity_error:.3f}{extra_suffix}"
        else:
            filename = f"model_{ts_str}{extra_suffix}"
        
        model_path = stage_dir / filename
        model.save(str(model_path))
        
        # Save vecnorm alongside
        vecnorm_path = stage_dir / f"vecnorm_{ts_str}.pkl"
        try:
            env.save(str(vecnorm_path))
        except Exception:
            pass
        
        print(f"✓ Checkpoint saved: {model_path}.zip")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints(stage_dir)
    
    def save_best(
        self, 
        model, 
        env, 
        metric: float,
        timesteps: int = None,
        metric_name: str = "velocity_error"
    ) -> bool:
        """
        Save model if it's the best so far.
        
        Args:
            model: SB3 model
            env: VecNormalize environment
            metric: Metric to compare (lower is better)
            timesteps: Current timesteps
            metric_name: Name of metric for logging
            
        Returns:
            True if this was a new best, False otherwise
        """
        timesteps = timesteps or model.num_timesteps
        
        if metric < self._best_metric:
            old_best = self._best_metric
            self._best_metric = metric
            self._best_timesteps = timesteps
            
            model_path = self.best_dir / "model"
            vecnorm_path = self.best_dir / "vecnorm.pkl"
            
            model.save(str(model_path))
            try:
                env.save(str(vecnorm_path))
            except Exception:
                pass
            
            self._save_best_info(metric, timesteps, {
                'metric_name': metric_name,
                'previous_best': old_best
            })
            
            print(f"🏆 New best! {metric_name}: {metric:.4f} (was {old_best:.4f})")
            return True
        
        return False
    
    def save_final(self, model, env):
        """Save final production model."""
        model_path = self.final_dir / "model"
        vecnorm_path = self.final_dir / "vecnorm.pkl"
        
        model.save(str(model_path))
        try:
            env.save(str(vecnorm_path))
        except Exception:
            pass
        
        info = {
            'timesteps': model.num_timesteps,
            'timestamp': datetime.now().isoformat(),
            'best_metric': self._best_metric
        }
        with open(self.final_dir / "info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ Final model saved: {model_path}.zip")
    
    def archive_config(self, config: Dict[str, Any], run_name: str = None):
        """
        Archive training configuration for this run.
        
        Args:
            config: Configuration dictionary
            run_name: Optional run name (uses timestamp if not provided)
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        config_path = self.config_dir / f"{run_name}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Config archived: {config_path}")
    
    def _cleanup_checkpoints(self, stage_dir: Path):
        """Remove old checkpoints, keeping only the most recent N."""
        checkpoints = sorted(
            stage_dir.glob("model_*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_ckpt in checkpoints[self.max_checkpoints_per_stage:]:
            try:
                old_ckpt.unlink()
                # Also remove corresponding vecnorm if exists
                ts_match = str(old_ckpt.stem).split('_')[1]
                for vecnorm in stage_dir.glob(f"vecnorm_{ts_match}*.pkl"):
                    vecnorm.unlink()
            except Exception:
                pass
    
    def load_latest(self, model_class, env):
        """
        Load the latest model.
        
        Args:
            model_class: SB3 model class (e.g., PPO)
            env: Environment to attach
            
        Returns:
            Loaded model or None if not found
        """
        model_path = self.latest_dir / "model.zip"
        if model_path.exists():
            return model_class.load(str(model_path), env=env)
        return None
    
    def load_best(self, model_class, env):
        """Load the best performing model."""
        model_path = self.best_dir / "model.zip"
        if model_path.exists():
            return model_class.load(str(model_path), env=env)
        return None
    
    def get_latest_path(self) -> Optional[Path]:
        """Get path to latest model if exists."""
        path = self.latest_dir / "model.zip"
        return path if path.exists() else None
    
    def get_best_path(self) -> Optional[Path]:
        """Get path to best model if exists."""
        path = self.best_dir / "model.zip"
        return path if path.exists() else None
    
    def get_vecnorm_path(self, which: str = "latest") -> Optional[Path]:
        """Get path to vecnorm file."""
        if which == "latest":
            path = self.latest_dir / "vecnorm.pkl"
        elif which == "best":
            path = self.best_dir / "vecnorm.pkl"
        else:
            path = self.final_dir / "vecnorm.pkl"
        return path if path.exists() else None
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get information about best model."""
        return {
            'metric': self._best_metric,
            'timesteps': self._best_timesteps,
            'path': str(self.get_best_path())
        }
    
    def list_checkpoints(self, stage: int = None) -> List[Path]:
        """List all checkpoints, optionally filtered by stage."""
        if stage is not None:
            stage_dir = self.checkpoint_dir / f"stage_{stage}"
            if stage_dir.exists():
                return sorted(stage_dir.glob("model_*.zip"))
            return []
        
        all_checkpoints = []
        for stage_dir in self.checkpoint_dir.iterdir():
            if stage_dir.is_dir():
                all_checkpoints.extend(stage_dir.glob("model_*.zip"))
        return sorted(all_checkpoints)


if __name__ == "__main__":
    # Test the model manager
    print("Testing ModelManager")
    print("=" * 50)
    
    manager = ModelManager("test_task", base_dir="test_models")
    
    print(f"\nTask directory: {manager.task_dir}")
    print(f"Latest directory: {manager.latest_dir}")
    print(f"Best directory: {manager.best_dir}")
    print(f"Checkpoint directory: {manager.checkpoint_dir}")
    
    # Test config archiving
    test_config = {
        'learning_rate': 0.0003,
        'n_envs': 32,
        'batch_size': 512
    }
    manager.archive_config(test_config)
    
    print("\n✓ ModelManager test complete")
    
    # Cleanup test directory
    shutil.rmtree("test_models", ignore_errors=True)

