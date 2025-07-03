"""
Main pipeline orchestrator for modular fine-tuning.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .algorithms.base import BaseStage, StageConfig, StageResult
from .algorithms.sft import SFTStage
from .algorithms.dpo import DPOStage
from .algorithms.rlaif import RLAIFStage
from .algorithms.rl import RLStage
from .algorithms.cot_distillation import CoTDistillationStage

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the entire fine-tuning pipeline."""

    # Model configuration
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    output_dir: str = field(metadata={"help": "Base output directory for the pipeline"})
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Use HuggingFace auth token for private models"},
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code when loading model"}
    )
    torch_dtype: str = field(
        default="auto",
        metadata={
            "help": "Torch dtype for model loading (auto, float16, bfloat16, float32)"
        },
    )

    # Pipeline configuration
    stages: List[str] = field(
        default_factory=lambda: ["sft"],
        metadata={"help": "List of stages to execute in order"},
    )
    stage_configs: Dict[str, Dict[str, Any]] = field(
        default_factory=dict, metadata={"help": "Configuration for each stage"}
    )

    # Global settings
    save_final_model: bool = field(
        default=True, metadata={"help": "Save the final model after all stages"}
    )
    cleanup_intermediate: bool = field(
        default=False,
        metadata={"help": "Remove intermediate models to save disk space"},
    )

    # Logging
    log_level: str = field(default="INFO", metadata={"help": "Logging level"})

    @classmethod
    def from_yaml(cls, config_path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        with open(output_path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)


class Pipeline:
    """Main pipeline orchestrator for modular fine-tuning."""

    # Registry of available stages
    STAGE_REGISTRY: Dict[str, Type[BaseStage]] = {}

    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Register default stages first
        self._register_default_stages()

        # Initialize stages
        self.stages: List[BaseStage] = []
        self._initialize_stages()

        # Track results
        self.stage_results: List[StageResult] = []

    @classmethod
    def register_stage(cls, stage_name: str, stage_class: Type[BaseStage]) -> None:
        """Register a stage class with the pipeline."""
        cls.STAGE_REGISTRY[stage_name] = stage_class
        logger.info(f"Registered stage: {stage_name}")

    def _register_default_stages(self) -> None:
        """Register all default stages."""
        self.register_stage("sft", SFTStage)
        self.register_stage("dpo", DPOStage)
        self.register_stage("rlaif", RLAIFStage)
        self.register_stage("rl", RLStage)
        self.register_stage("cot_distillation", CoTDistillationStage)

    def _initialize_stages(self) -> None:
        """Initialize all configured stages."""
        for stage_name in self.config.stages:
            if stage_name not in self.STAGE_REGISTRY:
                raise ValueError(
                    f"Unknown stage: {stage_name}. Available stages: {list(self.STAGE_REGISTRY.keys())}"
                )

            # Get stage configuration
            stage_config_dict = self.config.stage_configs.get(stage_name, {})

            # Add common configuration
            stage_config_dict.update(
                {
                    "stage_name": stage_name,
                    "output_dir": os.path.join(self.config.output_dir, stage_name),
                }
            )

            # Create stage configuration
            stage_class = self.STAGE_REGISTRY[stage_name]
            stage_config = self._create_stage_config(stage_class, stage_config_dict)

            # Create and validate stage
            stage = stage_class(stage_config)
            stage.validate_config()

            self.stages.append(stage)
            self.logger.info(f"Initialized stage: {stage_name}")

    def _create_stage_config(
        self, stage_class: Type[BaseStage], config_dict: Dict[str, Any]
    ) -> StageConfig:
        """Create stage configuration from dictionary."""
        # Map stage classes to their config classes
        config_class_map = {
            SFTStage: "SFTConfig",
            DPOStage: "DPOConfig",
            RLAIFStage: "RLAIFConfig",
            RLStage: "RLConfig",
            CoTDistillationStage: "CoTDistillationConfig",
        }

        if stage_class in config_class_map:
            config_class_name = config_class_map[stage_class]

            # Try to import the config class from the same module as the stage
            module = stage_class.__module__
            try:
                config_module = __import__(module, fromlist=[config_class_name])
                config_class = getattr(config_module, config_class_name)
                return config_class(**config_dict)
            except (ImportError, AttributeError) as e:
                self.logger.warning(
                    f"Could not load config class {config_class_name}: {e}"
                )
                # Fallback to base StageConfig
                return StageConfig(**config_dict)
        else:
            # Fallback to base StageConfig for unknown stages
            return StageConfig(**config_dict)

    def load_model_and_tokenizer(self) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the initial model and tokenizer."""
        self.logger.info(f"Loading model: {self.config.model_name_or_path}")

        # Determine torch dtype
        torch_dtype = torch.float16
        if self.config.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.config.torch_dtype == "float32":
            torch_dtype = torch.float32

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            use_auth_token=self.config.use_auth_token,
        )

        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            use_auth_token=self.config.use_auth_token,
            device_map="auto",
        )

        return model, tokenizer

    def execute(self) -> List[StageResult]:
        """Execute the entire pipeline."""
        self.logger.info(f"Starting pipeline execution with {len(self.stages)} stages")

        # Load initial model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Execute each stage
        previous_result = None
        for i, stage in enumerate(self.stages):
            self.logger.info(
                f"Executing stage {i+1}/{len(self.stages)}: {stage.stage_name}"
            )

            try:
                # Prepare model and tokenizer for this stage
                stage_model, stage_tokenizer = stage.prepare_model_and_tokenizer(
                    model, tokenizer, previous_result
                )

                # Execute the stage
                result = stage.execute(stage_model, stage_tokenizer, previous_result)

                # Store result
                self.stage_results.append(result)
                previous_result = result

                # Update model and tokenizer for next stage
                if result.success:
                    # Load the model from the stage output for the next stage
                    model = AutoModelForCausalLM.from_pretrained(result.model_path)
                    tokenizer = AutoTokenizer.from_pretrained(result.tokenizer_path)

                    self.logger.info(f"Stage {stage.stage_name} completed successfully")
                else:
                    self.logger.error(
                        f"Stage {stage.stage_name} failed: {result.error_message}"
                    )
                    break

            except Exception as e:
                self.logger.error(
                    f"Stage {stage.stage_name} failed with exception: {e}"
                )
                # Create failure result
                result = StageResult(
                    stage_name=stage.stage_name,
                    success=False,
                    model_path="",
                    tokenizer_path="",
                    error_message=str(e),
                )
                self.stage_results.append(result)
                break

        # Save final model if requested and pipeline succeeded
        if (
            self.config.save_final_model
            and self.stage_results
            and self.stage_results[-1].success
        ):
            final_model_path = os.path.join(self.config.output_dir, "final_model")
            self.logger.info(f"Saving final model to {final_model_path}")

            model.save_pretrained(final_model_path)  # type: ignore[attr-defined]
            tokenizer.save_pretrained(final_model_path)  # type: ignore[attr-defined]

        # Cleanup intermediate models if requested
        if self.config.cleanup_intermediate:
            self._cleanup_intermediate_models()

        self.logger.info("Pipeline execution completed")
        return self.stage_results

    def _cleanup_intermediate_models(self) -> None:
        """Remove intermediate model files to save disk space."""
        self.logger.info("Cleaning up intermediate models")

        for result in self.stage_results[:-1]:  # Keep the last model
            if result.success:
                model_path = Path(result.model_path)
                if model_path.exists():
                    import shutil

                    shutil.rmtree(model_path)
                    self.logger.info(f"Removed intermediate model: {model_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline execution."""
        successful_stages = [r for r in self.stage_results if r.success]
        failed_stages = [r for r in self.stage_results if not r.success]

        return {
            "total_stages": len(self.stages),
            "executed_stages": len(self.stage_results),
            "successful_stages": len(successful_stages),
            "failed_stages": len(failed_stages),
            "success_rate": (
                len(successful_stages) / len(self.stage_results)
                if self.stage_results
                else 0
            ),
            "stage_results": [
                {
                    "stage_name": r.stage_name,
                    "success": r.success,
                    "metrics": r.metrics,
                    "error": r.error_message,
                }
                for r in self.stage_results
            ],
        }
