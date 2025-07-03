"""
Base classes for pipeline algorithms.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Base configuration for pipeline algorithms."""

    stage_name: str = field(metadata={"help": "Name of the stage"})
    output_dir: str = field(metadata={"help": "Output directory for this stage"})
    enabled: bool = field(
        default=True, metadata={"help": "Whether this stage is enabled"}
    )
    save_intermediate: bool = field(
        default=True,
        metadata={"help": "Whether to save intermediate model after this stage"},
    )

    # Model loading/saving options
    load_best_model: bool = field(
        default=True, metadata={"help": "Load the best model from previous stage"}
    )

    # Logging and monitoring
    use_wandb: bool = field(
        default=False, metadata={"help": "Use Weights & Biases logging"}
    )
    wandb_project: str = field(
        default="fnsft-pipeline", metadata={"help": "W&B project name"}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "W&B run name"}
    )


@dataclass
class StageResult:
    """Result from executing a pipeline algorithm."""

    stage_name: str
    success: bool
    model_path: str
    tokenizer_path: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate the result."""
        if self.success and not Path(self.model_path).exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
        if self.success and not Path(self.tokenizer_path).exists():
            raise ValueError(f"Tokenizer path does not exist: {self.tokenizer_path}")


class BaseStage(ABC):
    """Abstract base class for all pipeline algorithms."""

    def __init__(self, config: StageConfig):
        """Initialize the stage with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Return the name of this stage."""
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """Validate the stage configuration."""
        pass

    @abstractmethod
    def execute(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        previous_result: Optional[StageResult] = None,
    ) -> StageResult:
        """
        Execute the stage.

        Args:
            model: The model to train/fine-tune
            tokenizer: The tokenizer
            previous_result: Result from the previous stage (if any)

        Returns:
            StageResult containing the results of this stage
        """
        pass

    def prepare_model_and_tokenizer(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        previous_result: Optional[StageResult] = None,
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Prepare model and tokenizer for this stage.

        This method can be overridden by subclasses to perform stage-specific
        model preparation (e.g., loading adapters, changing model configuration).

        Args:
            model: Input model
            tokenizer: Input tokenizer
            previous_result: Result from previous stage

        Returns:
            Tuple of (prepared_model, prepared_tokenizer)
        """
        return model, tokenizer

    def save_model_and_tokenizer(
        self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, suffix: str = ""
    ) -> tuple[str, str]:
        """
        Save model and tokenizer to the stage output directory.

        Args:
            model: Model to save
            tokenizer: Tokenizer to save
            suffix: Optional suffix for the save directory

        Returns:
            Tuple of (model_path, tokenizer_path)
        """
        save_dir = Path(self.config.output_dir) / f"{self.stage_name}{suffix}"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = str(save_dir)
        tokenizer_path = str(save_dir)

        self.logger.info(f"Saving {self.stage_name} model to {model_path}")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(tokenizer_path)

        return model_path, tokenizer_path

    def setup_logging(self) -> None:
        """Setup logging for this stage."""
        if self.config.use_wandb:
            import wandb

            run_name = (
                self.config.wandb_run_name
                or f"{self.stage_name}-{self.config.stage_name}"
            )

            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config=self.config.__dict__,
                reinit=True,
            )

    def cleanup_logging(self) -> None:
        """Cleanup logging for this stage."""
        if self.config.use_wandb:
            import wandb

            wandb.finish()
