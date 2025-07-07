"""
Weights & Biases integration utilities for LMPipeline.

This module provides utilities for logging hyperparameters and training metrics
to W&B while excluding sensitive local file paths.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class WandBLogger:
    """Handles Weights & Biases logging for pipeline training."""

    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ):
        """
        Initialize W&B logger.

        Args:
            project_name: W&B project name
            run_name: Optional run name
            tags: Optional list of tags
            notes: Optional notes for the run
        """
        self.project_name = project_name
        self.run_name = run_name
        self.tags = tags or []
        self.notes = notes
        self.wandb = None
        self.run = None

        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            logger.warning("wandb not installed. W&B logging will be disabled.")

    def init_run(self, config: Dict[str, Any], stage_name: str = "") -> bool:
        """
        Initialize a W&B run.

        Args:
            config: Configuration dictionary to log
            stage_name: Current stage name

        Returns:
            True if initialization successful, False otherwise
        """
        if not self.wandb:
            return False

        try:
            # Clean config for W&B logging
            clean_config = self._clean_config_for_wandb(config)

            # Create run name with stage if provided
            run_name = self.run_name
            if stage_name:
                run_name = f"{self.run_name or 'pipeline'}-{stage_name}"

            # Add stage to tags
            tags = self.tags.copy()
            if stage_name:
                tags.append(f"stage:{stage_name}")

            self.run = self.wandb.init(
                project=self.project_name,
                name=run_name,
                config=clean_config,
                tags=tags,
                notes=self.notes,
                reinit=True,
            )

            logger.info(f"Initialized W&B run: {run_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize W&B run: {e}")
            return False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.wandb or not self.run:
            return

        try:
            # Filter out non-numeric values
            clean_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    clean_metrics[key] = value
                elif hasattr(value, "item"):  # Handle torch tensors
                    try:
                        clean_metrics[key] = value.item()
                    except:
                        continue

            if clean_metrics:
                self.wandb.log(clean_metrics, step=step)

        except Exception as e:
            logger.error(f"Failed to log metrics to W&B: {e}")

    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters to W&B config.

        Args:
            hyperparams: Dictionary of hyperparameters
        """
        if not self.wandb or not self.run:
            return

        try:
            clean_hyperparams = self._clean_config_for_wandb(hyperparams)
            self.wandb.config.update(clean_hyperparams)

        except Exception as e:
            logger.error(f"Failed to log hyperparameters to W&B: {e}")

    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """
        Log model architecture information.

        Args:
            model_info: Dictionary containing model information
        """
        if not self.wandb or not self.run:
            return

        try:
            # Extract relevant model info
            clean_info = {}

            # Model architecture parameters
            arch_params = [
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "intermediate_size",
                "vocab_size",
                "max_position_embeddings",
                "num_layers",
                "n_head",
                "d_model",
                "n_embd",
            ]

            for param in arch_params:
                if param in model_info:
                    clean_info[f"model.{param}"] = model_info[param]

            # Model type and name (without local paths)
            if "model_type" in model_info:
                clean_info["model.type"] = model_info["model_type"]

            if "architectures" in model_info:
                clean_info["model.architectures"] = model_info["architectures"]

            if clean_info:
                self.wandb.config.update(clean_info)

        except Exception as e:
            logger.error(f"Failed to log model info to W&B: {e}")

    def log_stage_completion(
        self, stage_name: str, duration: float, final_metrics: Dict[str, Any]
    ) -> None:
        """
        Log stage completion information.

        Args:
            stage_name: Name of completed stage
            duration: Stage duration in seconds
            final_metrics: Final metrics from the stage
        """
        if not self.wandb or not self.run:
            return

        try:
            completion_data = {
                f"{stage_name}.duration_seconds": duration,
                f"{stage_name}.completed": True,
            }

            # Add final metrics with stage prefix
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    completion_data[f"{stage_name}.final.{key}"] = value

            self.wandb.log(completion_data)

        except Exception as e:
            logger.error(f"Failed to log stage completion to W&B: {e}")

    def finish_run(self) -> None:
        """Finish the current W&B run."""
        if self.wandb and self.run:
            try:
                self.wandb.finish()
                self.run = None
                logger.info("Finished W&B run")
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")

    def _clean_config_for_wandb(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean configuration for W&B logging by removing sensitive paths.

        Args:
            config: Original configuration

        Returns:
            Cleaned configuration safe for W&B
        """
        clean_config = {}

        # Fields to exclude (contain local paths or sensitive info)
        exclude_fields = {
            "output_dir",
            "dataset_name_or_path",
            "resume_from_checkpoint",
            "model_name_or_path",
            "tokenizer_name_or_path",
            "cache_dir",
            "logging_dir",
            "run_name",
            "wandb_project",
            "wandb_run_name",
        }

        # Fields to include with sanitization
        path_fields = {
            "model_name_or_path": "model_name",
            "dataset_name_or_path": "dataset_name",
        }

        for key, value in config.items():
            # Sanitize path fields to only include model/dataset names
            if key in path_fields:
                if isinstance(value, str):
                    # Extract just the model/dataset name
                    clean_name = os.path.basename(value) if "/" in value else value
                    clean_config[path_fields[key]] = clean_name
                continue

            # Skip excluded fields
            if key in exclude_fields:
                continue

            # Include other fields if they're serializable
            if self._is_serializable(value):
                clean_config[key] = value

        return clean_config

    def _is_serializable(self, value: Any) -> bool:
        """Check if a value is serializable for W&B."""
        if value is None:
            return True

        if isinstance(value, (bool, int, float, str)):
            return True

        if isinstance(value, (list, tuple)):
            return all(self._is_serializable(item) for item in value)

        if isinstance(value, dict):
            return all(
                isinstance(k, str) and self._is_serializable(v)
                for k, v in value.items()
            )

        return False


def create_wandb_logger(
    config: Dict[str, Any], pipeline_id: str
) -> Optional[WandBLogger]:
    """
    Create a W&B logger from pipeline configuration.

    Args:
        config: Pipeline configuration
        pipeline_id: Unique pipeline identifier

    Returns:
        WandBLogger instance or None if W&B is disabled
    """
    if not config.get("use_wandb", False):
        return None

    project_name = config.get("wandb_project", "lmpipeline")
    run_name = config.get("wandb_run_name") or f"pipeline-{pipeline_id}"

    # Extract tags from config
    tags = ["lmpipeline"]
    if "stages" in config:
        tags.extend([f"stage:{stage}" for stage in config["stages"]])

    return WandBLogger(
        project_name=project_name,
        run_name=run_name,
        tags=tags,
        notes=f"LMPipeline run {pipeline_id}",
    )
