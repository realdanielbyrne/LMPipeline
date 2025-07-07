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
from .algorithms.trl_sft import TRLSFTStage
from .algorithms.dpo import DPOStage
from .algorithms.rlaif import RLAIFStage
from .algorithms.rl import RLStage
from .algorithms.cot_distillation import CoTDistillationStage
from .utils.post_processing import upload_to_hub, convert_to_gguf
from .utils.config_defaults import ConfigDefaults
from .utils.training_state import TrainingStateManager
from .utils.wandb_integration import create_wandb_logger

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

    # Training state persistence
    enable_state_persistence: bool = field(
        default=True,
        metadata={"help": "Enable training state persistence and recovery"},
    )
    auto_resume: bool = field(
        default=True,
        metadata={"help": "Automatically resume from last checkpoint if available"},
    )
    force_restart: bool = field(
        default=False,
        metadata={"help": "Force restart training even if state file exists"},
    )

    # Hugging Face Hub upload configuration
    push_to_hub: bool = field(
        default=False, metadata={"help": "Upload the final model to Hugging Face Hub"}
    )
    hub_repo_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Repository ID for Hugging Face Hub (e.g., 'username/model-name')"
        },
    )
    hub_commit_message: Optional[str] = field(
        default=None, metadata={"help": "Commit message for Hub upload"}
    )
    hub_private: bool = field(
        default=False, metadata={"help": "Create private repository on Hub"}
    )
    hub_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Hugging Face authentication token (or set HF_TOKEN env var)"
        },
    )
    push_adapter_only: bool = field(
        default=False,
        metadata={"help": "Only upload LoRA adapter files to Hub (not the full model)"},
    )

    # GGUF conversion configuration
    convert_to_gguf: bool = field(
        default=False, metadata={"help": "Convert final model to GGUF format"}
    )
    gguf_quantization: str = field(
        default="q4_0",
        metadata={"help": "GGUF quantization type (q4_0, q8_0, f16, etc.)"},
    )
    gguf_output_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output path for GGUF file (defaults to output_dir/model.gguf)"
        },
    )

    @classmethod
    def from_yaml(cls, config_path: str) -> "PipelineConfig":
        """Load configuration from YAML file with intelligent defaults applied."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Apply intelligent defaults
        config_dict = ConfigDefaults.apply_all_defaults(config_dict)

        # Validate that required directories can be created
        failed_dirs = ConfigDefaults.validate_and_create_directories(config_dict)
        if failed_dirs:
            logger.warning(f"Could not create some directories: {failed_dirs}")

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

        # Initialize training state manager
        self.state_manager = None
        if config.enable_state_persistence:
            self.state_manager = TrainingStateManager(
                output_dir=config.output_dir,
                pipeline_config=config.__dict__,
                enable_persistence=True,
            )

            # Handle force restart
            if config.force_restart and self.state_manager.state_file.exists():
                self.logger.info(
                    "Force restart requested - removing existing state file"
                )
                self.state_manager.state_file.unlink()
                self.state_manager._create_new_state(config.__dict__)

        # Initialize W&B logger
        self.wandb_logger = None
        if config.__dict__.get("use_wandb", False):
            pipeline_id = (
                self.state_manager.state.pipeline_id
                if self.state_manager
                else "unknown"
            )
            self.wandb_logger = create_wandb_logger(config.__dict__, pipeline_id)

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
        self.register_stage("trl_sft", TRLSFTStage)
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
                    "state_manager": self.state_manager,
                    "wandb_logger": self.wandb_logger,
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

        # Check for resumption
        resume_point = None
        if self.config.auto_resume and self.state_manager:
            if not self.state_manager.validate_file_paths():
                self.logger.warning(
                    "Some referenced files no longer exist. Starting fresh."
                )
                self.state_manager._create_new_state(self.config.__dict__)
            else:
                resume_point = self.state_manager.get_resume_point()
                if resume_point:
                    completed_stages = self.state_manager.get_completed_stages()
                    self.logger.info(
                        f"Resuming from stage: {resume_point}. "
                        f"Completed stages: {completed_stages}"
                    )

        # Load initial model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Execute each stage
        previous_result = None
        for i, stage in enumerate(self.stages):
            stage_name = stage.stage_name

            # Skip completed stages if resuming
            if (
                resume_point
                and self.state_manager
                and stage_name in self.state_manager.get_completed_stages()
            ):
                self.logger.info(f"Skipping completed stage: {stage_name}")

                # Load the result from state
                if (
                    self.state_manager.state
                    and stage_name in self.state_manager.state.stages
                ):
                    stage_progress = self.state_manager.state.stages[stage_name]
                    if stage_progress.model_path and stage_progress.tokenizer_path:
                        # Create a result object for the completed stage
                        previous_result = StageResult(
                            stage_name=stage_name,
                            success=True,
                            model_path=stage_progress.model_path,
                            tokenizer_path=stage_progress.tokenizer_path,
                            metrics=stage_progress.metrics,
                        )
                        self.stage_results.append(previous_result)

                        # Load the model for the next stage
                        model = AutoModelForCausalLM.from_pretrained(
                            stage_progress.model_path
                        )
                        tokenizer = AutoTokenizer.from_pretrained(
                            stage_progress.tokenizer_path
                        )
                continue

            self.logger.info(f"Executing stage {i+1}/{len(self.stages)}: {stage_name}")

            # Mark stage as started in state manager
            if self.state_manager:
                # Extract training parameters for progress tracking
                stage_config = getattr(stage, "config", None)
                total_epochs = getattr(stage_config, "num_train_epochs", 0)
                self.state_manager.start_stage(stage_name, total_epochs=total_epochs)

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

                # Update state manager
                if self.state_manager:
                    if result.success:
                        self.state_manager.complete_stage(
                            stage_name,
                            result.model_path,
                            result.tokenizer_path,
                            result.metrics,
                        )
                    else:
                        self.state_manager.fail_stage(
                            stage_name, result.error_message or "Unknown error"
                        )

                # Update model and tokenizer for next stage
                if result.success:
                    # Load the model from the stage output for the next stage
                    model = AutoModelForCausalLM.from_pretrained(result.model_path)
                    tokenizer = AutoTokenizer.from_pretrained(result.tokenizer_path)

                    self.logger.info(f"Stage {stage_name} completed successfully")
                else:
                    self.logger.error(
                        f"Stage {stage_name} failed: {result.error_message}"
                    )
                    break

            except Exception as e:
                self.logger.error(f"Stage {stage_name} failed with exception: {e}")

                # Update state manager
                if self.state_manager:
                    self.state_manager.fail_stage(stage_name, str(e))

                # Create failure result
                result = StageResult(
                    stage_name=stage_name,
                    success=False,
                    model_path="",
                    tokenizer_path="",
                    error_message=str(e),
                )
                self.stage_results.append(result)
                break

        # Save final model if requested and pipeline succeeded
        final_model_path = None
        if (
            self.config.save_final_model
            and self.stage_results
            and self.stage_results[-1].success
        ):
            # Generate intelligent final model name
            config_dict = self.config.__dict__.copy()
            final_model_name = ConfigDefaults.generate_final_model_name(config_dict)
            final_model_path = os.path.join(self.config.output_dir, final_model_name)

            self.logger.info(f"Saving final model to {final_model_path}")

            model.save_pretrained(final_model_path)  # type: ignore[attr-defined]
            tokenizer.save_pretrained(final_model_path)  # type: ignore[attr-defined]

        # Run post-processing steps if pipeline succeeded
        if self.stage_results and self.stage_results[-1].success:
            self._run_post_processing(final_model_path)

        # Mark pipeline as completed in state manager
        if self.state_manager and self.stage_results and self.stage_results[-1].success:
            self.state_manager.complete_pipeline(final_model_path or "")

        # Cleanup intermediate models if requested
        if self.config.cleanup_intermediate:
            self._cleanup_intermediate_models()

        # Finish W&B run if active
        if self.wandb_logger:
            self.wandb_logger.finish_run()

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

    def _run_post_processing(self, final_model_path: Optional[str]) -> None:
        """Run post-processing steps like GGUF conversion and Hub upload."""
        if not final_model_path:
            # Use the last successful stage's model path
            if self.stage_results and self.stage_results[-1].success:
                final_model_path = self.stage_results[-1].model_path
                final_tokenizer_path = self.stage_results[-1].tokenizer_path
            else:
                self.logger.warning("No final model path available for post-processing")
                return
        else:
            final_tokenizer_path = final_model_path

        self.logger.info("Starting post-processing steps")

        # GGUF conversion
        gguf_output_path = None
        if self.config.convert_to_gguf:
            try:
                self.logger.info("Converting model to GGUF format")

                # Determine output path - use intelligent naming if not specified
                if self.config.gguf_output_path:
                    gguf_output_path = self.config.gguf_output_path
                else:
                    # Generate intelligent GGUF filename
                    config_dict = self.config.__dict__.copy()
                    gguf_model_name = ConfigDefaults.generate_model_name(
                        base_model_name=self.config.model_name_or_path,
                        stages=self.config.stages,
                        quantization_config=ConfigDefaults._extract_quantization_config(
                            config_dict
                        ),
                        torch_dtype=self.config.torch_dtype,
                        convert_to_gguf=True,
                        gguf_quantization=self.config.gguf_quantization,
                    )
                    gguf_output_path = os.path.join(
                        self.config.output_dir, f"{gguf_model_name}.gguf"
                    )

                convert_to_gguf(
                    model_path=final_model_path,
                    output_path=gguf_output_path,
                    quantization=self.config.gguf_quantization,
                )
                self.logger.info(f"GGUF conversion completed: {gguf_output_path}")

            except Exception as e:
                self.logger.error(f"GGUF conversion failed: {e}")
                # Don't fail the entire pipeline for post-processing errors

        # Hugging Face Hub upload
        if self.config.push_to_hub:
            try:
                if not self.config.hub_repo_id:
                    self.logger.error("hub_repo_id is required for Hub upload")
                    return

                self.logger.info(
                    f"Uploading model to Hugging Face Hub: {self.config.hub_repo_id}"
                )

                # If GGUF conversion was done and we want to upload GGUF, modify repo_id
                upload_repo_id = self.config.hub_repo_id
                upload_model_path = final_model_path

                if gguf_output_path and self.config.convert_to_gguf:
                    # Upload GGUF version with modified repo name
                    if not upload_repo_id.endswith("_GGUF"):
                        upload_repo_id = f"{upload_repo_id}_GGUF"

                    # For GGUF, we need to upload the file differently
                    self.logger.info(f"Uploading GGUF model to: {upload_repo_id}")
                    # TODO: Implement GGUF-specific upload logic
                    # For now, upload the original model

                upload_to_hub(
                    model_path=upload_model_path,
                    tokenizer_path=final_tokenizer_path,
                    repo_id=upload_repo_id,
                    commit_message=self.config.hub_commit_message,
                    private=self.config.hub_private,
                    token=self.config.hub_token,
                    push_adapter_only=self.config.push_adapter_only,
                )
                self.logger.info(f"Hub upload completed: {upload_repo_id}")

            except Exception as e:
                self.logger.error(f"Hub upload failed: {e}")
                # Don't fail the entire pipeline for post-processing errors

        self.logger.info("Post-processing steps completed")

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
