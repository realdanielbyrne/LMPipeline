"""
Training state persistence system for LMPipeline.

This module provides comprehensive training state management including:
- Local JSON status file tracking
- Weights & Biases integration
- Recovery mechanism for interrupted training sessions
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class StageProgress:
    """Progress tracking for a single pipeline stage."""

    stage_name: str
    status: str  # "not_started", "in_progress", "completed", "failed"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    last_checkpoint_path: Optional[str] = None
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageProgress":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingState:
    """Complete training state for a pipeline run."""

    pipeline_id: str
    config_hash: str
    current_stage: str
    stages: Dict[str, StageProgress] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    output_dir: str = ""
    model_name_or_path: str = ""
    dataset_paths: List[str] = field(default_factory=list)
    final_model_path: Optional[str] = None
    pipeline_completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert StageProgress objects to dicts
        data["stages"] = {
            k: v.to_dict() if isinstance(v, StageProgress) else v
            for k, v in self.stages.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        """Create from dictionary."""
        # Convert stage dicts back to StageProgress objects
        stages = {
            k: StageProgress.from_dict(v) if isinstance(v, dict) else v
            for k, v in data.get("stages", {}).items()
        }
        data["stages"] = stages
        return cls(**data)


class TrainingStateManager:
    """Manages training state persistence and recovery."""

    def __init__(
        self,
        output_dir: str,
        pipeline_config: Dict[str, Any],
        enable_persistence: bool = True,
    ):
        """
        Initialize the training state manager.

        Args:
            output_dir: Directory where state file will be saved
            pipeline_config: Pipeline configuration for hash generation
            enable_persistence: Whether to enable state persistence
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_dir / "training_status.json"
        self.enable_persistence = enable_persistence

        # Generate configuration hash for change detection
        self.config_hash = self._generate_config_hash(pipeline_config)

        # Initialize or load state
        self.state: Optional[TrainingState] = None
        if self.enable_persistence:
            self._load_or_create_state(pipeline_config)

    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate a hash of the pipeline configuration."""
        # Create a copy and remove non-deterministic fields
        config_copy = config.copy()

        # Remove fields that shouldn't affect resumption
        exclude_fields = {
            "output_dir",
            "wandb_run_name",
            "resume_from_checkpoint",
            "log_level",
            "save_intermediate",
            "cleanup_intermediate",
        }

        for field in exclude_fields:
            config_copy.pop(field, None)

        # Sort keys for consistent hashing
        config_str = json.dumps(config_copy, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _load_or_create_state(self, pipeline_config: Dict[str, Any]) -> None:
        """Load existing state or create new one."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state_data = json.load(f)

                self.state = TrainingState.from_dict(state_data)

                # Check if configuration has changed
                if self.state.config_hash != self.config_hash:
                    logger.warning(
                        "Pipeline configuration has changed since last run. "
                        "Starting fresh training."
                    )
                    self._create_new_state(pipeline_config)
                else:
                    logger.info(
                        f"Loaded existing training state from {self.state_file}"
                    )

            except Exception as e:
                logger.warning(f"Failed to load state file: {e}. Starting fresh.")
                self._create_new_state(pipeline_config)
        else:
            self._create_new_state(pipeline_config)

    def _create_new_state(self, pipeline_config: Dict[str, Any]) -> None:
        """Create a new training state."""
        pipeline_id = f"pipeline_{int(time.time())}"

        self.state = TrainingState(
            pipeline_id=pipeline_id,
            config_hash=self.config_hash,
            current_stage="",
            output_dir=str(self.output_dir),
            model_name_or_path=pipeline_config.get("model_name_or_path", ""),
            dataset_paths=self._extract_dataset_paths(pipeline_config),
        )

        # Initialize stage progress for all configured stages
        stages = pipeline_config.get("stages", [])
        for stage_name in stages:
            self.state.stages[stage_name] = StageProgress(
                stage_name=stage_name, status="not_started"
            )

        logger.info(f"Created new training state with ID: {pipeline_id}")

    def _extract_dataset_paths(self, config: Dict[str, Any]) -> List[str]:
        """Extract dataset paths from configuration."""
        paths = []

        # Check stage configs for dataset paths
        stage_configs = config.get("stage_configs", {})
        for stage_config in stage_configs.values():
            if isinstance(stage_config, dict):
                dataset_path = stage_config.get("dataset_name_or_path")
                if dataset_path and os.path.exists(dataset_path):
                    paths.append(dataset_path)

        return paths

    def save_state(self) -> None:
        """Save current state to file."""
        if not self.enable_persistence or not self.state:
            return

        try:
            self.state.last_update = time.time()
            with open(self.state_file, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
            logger.debug(f"Saved training state to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save training state: {e}")

    def start_stage(
        self, stage_name: str, total_epochs: int = 0, total_steps: int = 0
    ) -> None:
        """Mark a stage as started."""
        if not self.state:
            return

        if stage_name not in self.state.stages:
            self.state.stages[stage_name] = StageProgress(
                stage_name=stage_name, status="not_started"
            )

        progress = self.state.stages[stage_name]
        progress.status = "in_progress"
        progress.start_time = time.time()
        progress.total_epochs = total_epochs
        progress.total_steps = total_steps
        progress.current_epoch = 0
        progress.current_step = 0

        self.state.current_stage = stage_name
        self.save_state()

        logger.info(f"Started stage: {stage_name}")

    def update_stage_progress(
        self,
        stage_name: str,
        current_epoch: int = None,
        current_step: int = None,
        metrics: Dict[str, Any] = None,
        checkpoint_path: str = None,
    ) -> None:
        """Update progress for a stage."""
        if not self.state or stage_name not in self.state.stages:
            return

        progress = self.state.stages[stage_name]

        if current_epoch is not None:
            progress.current_epoch = current_epoch
        if current_step is not None:
            progress.current_step = current_step
        if metrics:
            progress.metrics.update(metrics)
        if checkpoint_path:
            progress.last_checkpoint_path = checkpoint_path

        self.save_state()

    def complete_stage(
        self,
        stage_name: str,
        model_path: str,
        tokenizer_path: str,
        final_metrics: Dict[str, Any] = None,
    ) -> None:
        """Mark a stage as completed."""
        if not self.state or stage_name not in self.state.stages:
            return

        progress = self.state.stages[stage_name]
        progress.status = "completed"
        progress.end_time = time.time()
        progress.model_path = model_path
        progress.tokenizer_path = tokenizer_path

        if final_metrics:
            progress.metrics.update(final_metrics)

        self.save_state()

        logger.info(f"Completed stage: {stage_name}")

    def fail_stage(self, stage_name: str, error_message: str) -> None:
        """Mark a stage as failed."""
        if not self.state or stage_name not in self.state.stages:
            return

        progress = self.state.stages[stage_name]
        progress.status = "failed"
        progress.end_time = time.time()
        progress.error_message = error_message

        self.save_state()

        logger.error(f"Stage {stage_name} failed: {error_message}")

    def complete_pipeline(self, final_model_path: str) -> None:
        """Mark the entire pipeline as completed."""
        if not self.state:
            return

        self.state.pipeline_completed = True
        self.state.final_model_path = final_model_path
        self.state.current_stage = "completed"
        self.save_state()

        logger.info("Pipeline completed successfully")

    def get_resume_point(self) -> Optional[str]:
        """Get the stage from which to resume training."""
        if not self.state:
            return None

        # Find the first incomplete stage
        for stage_name, progress in self.state.stages.items():
            if progress.status in ["not_started", "failed"]:
                return stage_name
            elif progress.status == "in_progress":
                # Check if we can resume from checkpoint
                if progress.last_checkpoint_path and os.path.exists(
                    progress.last_checkpoint_path
                ):
                    return stage_name
                else:
                    # Restart this stage
                    progress.status = "not_started"
                    progress.current_epoch = 0
                    progress.current_step = 0
                    return stage_name

        return None

    def get_completed_stages(self) -> List[str]:
        """Get list of completed stages."""
        if not self.state:
            return []

        return [
            name
            for name, progress in self.state.stages.items()
            if progress.status == "completed"
        ]

    def validate_file_paths(self) -> bool:
        """Validate that all referenced file paths still exist."""
        if not self.state:
            return True

        # Check dataset paths
        for path in self.state.dataset_paths:
            if not os.path.exists(path):
                logger.warning(f"Dataset path no longer exists: {path}")
                return False

        # Check completed stage model paths
        for progress in self.state.stages.values():
            if progress.status == "completed":
                if progress.model_path and not os.path.exists(progress.model_path):
                    logger.warning(
                        f"Model path no longer exists: {progress.model_path}"
                    )
                    return False
                if progress.tokenizer_path and not os.path.exists(
                    progress.tokenizer_path
                ):
                    logger.warning(
                        f"Tokenizer path no longer exists: {progress.tokenizer_path}"
                    )
                    return False

        return True

    def get_stage_checkpoint(self, stage_name: str) -> Optional[str]:
        """Get the checkpoint path for a stage if available."""
        if not self.state or stage_name not in self.state.stages:
            return None

        progress = self.state.stages[stage_name]
        if progress.last_checkpoint_path and os.path.exists(
            progress.last_checkpoint_path
        ):
            return progress.last_checkpoint_path

        return None

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current training state."""
        if not self.state:
            return {}

        completed_stages = len(self.get_completed_stages())
        total_stages = len(self.state.stages)

        return {
            "pipeline_id": self.state.pipeline_id,
            "current_stage": self.state.current_stage,
            "progress": f"{completed_stages}/{total_stages} stages completed",
            "pipeline_completed": self.state.pipeline_completed,
            "start_time": self.state.start_time,
            "last_update": self.state.last_update,
            "stages": {
                name: progress.status for name, progress in self.state.stages.items()
            },
        }
