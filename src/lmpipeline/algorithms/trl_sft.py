"""
TRL-based Supervised Fine-Tuning (SFT) stage for the modular pipeline.

This implementation uses TRL's SFTTrainer instead of the standard transformers Trainer,
providing enhanced capabilities for chat template handling and conversational datasets.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .sft import SFTConfig
from .base import BaseStage, StageResult
from ..utils.model_utils import (
    setup_lora,
    load_dataset_from_path,
    split_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class TRLSFTConfig(SFTConfig):
    """Configuration for TRL-based Supervised Fine-Tuning stage."""

    # TRL-specific configuration options
    packing: bool = field(
        default=False,
        metadata={
            "help": "Whether to pack multiple sequences into fixed-length blocks"
        },
    )
    packing_strategy: str = field(
        default="ffd",
        metadata={"help": "Strategy for packing sequences ('ffd' or 'wrapped')"},
    )
    completion_only_loss: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to compute loss only on completion part"},
    )
    assistant_only_loss: bool = field(
        default=False,
        metadata={"help": "Whether to compute loss only on assistant responses"},
    )
    neftune_noise_alpha: Optional[float] = field(
        default=None,
        metadata={"help": "NEFTune noise alpha for performance enhancement"},
    )
    use_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether to use Liger kernel for memory optimization"},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Name of the column containing text data"}
    )
    formatting_func: Optional[str] = field(
        default=None, metadata={"help": "Custom formatting function name (if needed)"}
    )
    chat_template_path: Optional[str] = field(
        default=None, metadata={"help": "Path to custom chat template"}
    )
    eos_token: Optional[str] = field(
        default=None, metadata={"help": "End-of-sequence token for chat templates"}
    )
    pad_token: Optional[str] = field(default=None, metadata={"help": "Padding token"})


class TRLSFTStage(BaseStage):
    """TRL-based Supervised Fine-Tuning stage implementation."""

    def __init__(self, config: TRLSFTConfig):
        """Initialize the TRL SFT stage."""
        super().__init__(config)
        self.config: TRLSFTConfig = config

    @property
    def stage_name(self) -> str:
        """Return the name of this stage."""
        return "trl_sft"

    def validate_config(self) -> None:
        """Validate the stage configuration."""
        if not self.config.dataset_name_or_path:
            raise ValueError("dataset_name_or_path must be specified")

        if self.config.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")

        if self.config.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")

        if self.config.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")

        # Validate TRL-specific options
        if self.config.packing_strategy not in ["ffd", "wrapped"]:
            raise ValueError("packing_strategy must be 'ffd' or 'wrapped'")

    def prepare_model_and_tokenizer(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        previous_result: Optional[StageResult] = None,
    ) -> tuple[Any, Any]:
        """Prepare model and tokenizer for TRL SFT training."""
        self.logger.info("Preparing model for TRL SFT training")

        # Setup LoRA if configured
        model = setup_lora(
            model=model,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target_modules=self.config.lora_target_modules,
            lora_bias=self.config.lora_bias,
        )

        # Set up padding token if not present
        if tokenizer.pad_token is None:
            if self.config.pad_token:
                tokenizer.pad_token = self.config.pad_token
            elif tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Add a pad token if none exists
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def _detect_chat_format(self, data: List[Dict[str, Any]]) -> bool:
        """Detect if the dataset contains chat-formatted data with 'messages' field."""
        if not data:
            return False

        # Check first few samples for 'messages' field
        for i in range(min(5, len(data))):
            if "messages" in data[i]:
                messages = data[i]["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    # Check if messages have the expected structure
                    if isinstance(messages[0], dict) and "role" in messages[0]:
                        self.logger.info(
                            "Detected chat-formatted dataset with 'messages' field"
                        )
                        return True

        return False

    def _prepare_dataset_for_trl(
        self, data: List[Dict[str, Any]], tokenizer: AutoTokenizer
    ) -> List[Dict[str, Any]]:
        """Prepare dataset for TRL SFTTrainer."""
        is_chat_format = self._detect_chat_format(data)

        if is_chat_format:
            self.logger.info(
                "Using chat format - TRL will handle template application automatically"
            )
            # For chat format, return data as-is - TRL will handle the formatting
            return data
        else:
            # For non-chat format, check if we need to convert to standard format
            if self.config.auto_detect_format:
                self.logger.info("Converting dataset to standard format for TRL")
                from ..utils.dataset_utils import DatasetFormatter

                # Detect format and convert
                detected_format = DatasetFormatter.detect_format(data)
                self.logger.info(f"Detected format: {detected_format}")

                converted_data = []
                for item in data:
                    try:
                        converted_item = DatasetFormatter.convert_to_standard_format(
                            item, detected_format
                        )
                        converted_data.append(converted_item)
                    except Exception as e:
                        self.logger.warning(f"Failed to convert item: {e}")
                        converted_data.append(item)

                return converted_data
            else:
                return data

    def execute(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        previous_result: Optional[StageResult] = None,
    ) -> StageResult:
        """Execute the TRL SFT training stage."""
        try:
            self.logger.info("Starting TRL SFT training")
            self.setup_logging()

            # Validate configuration
            self.validate_config()

            # Load and prepare dataset
            data = load_dataset_from_path(
                dataset_name_or_path=self.config.dataset_name_or_path,
                dataset_config_name=self.config.dataset_config_name,
            )
            train_data, val_data = split_dataset(data, self.config.validation_split)

            # Prepare datasets for TRL
            train_data = self._prepare_dataset_for_trl(train_data, tokenizer)
            val_data = (
                self._prepare_dataset_for_trl(val_data, tokenizer) if val_data else None
            )

            # Create TRL trainer and train
            trainer = self._create_trl_trainer(model, tokenizer, train_data, val_data)

            # Start training
            self.logger.info("Starting TRL SFT training...")
            trainer.train(resume_from_checkpoint=self.config.resume_from_checkpoint)

            # Save model
            model_path, tokenizer_path = self.save_model_and_tokenizer(model, tokenizer)

            # Extract metrics
            metrics = {}
            if trainer.state.log_history:
                final_metrics = trainer.state.log_history[-1]
                metrics = {
                    k: v
                    for k, v in final_metrics.items()
                    if isinstance(v, (int, float))
                }

            self.cleanup_logging()

            return StageResult(
                stage_name=self.stage_name,
                success=True,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                metrics=metrics,
            )

        except Exception as e:
            self.logger.error(f"TRL SFT training failed: {e}")
            self.cleanup_logging()

            return StageResult(
                stage_name=self.stage_name,
                success=False,
                model_path="",
                tokenizer_path="",
                error_message=str(e),
            )

    def _create_trl_trainer(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_data: List[Dict[str, Any]],
        val_data: Optional[List[Dict[str, Any]]] = None,
    ):
        """Create TRL SFTTrainer instance."""
        try:
            from trl import SFTTrainer, SFTConfig
        except ImportError:
            raise ImportError(
                "TRL library is required for TRL SFT training. Install it with: pip install trl"
            )

        # Convert data to datasets format
        from datasets import Dataset

        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(val_data) if val_data else None

        # Create TRL SFTConfig
        sft_config = self._create_trl_sft_config()

        # Create PEFT config if LoRA is enabled
        peft_config = None
        if any(
            [
                self.config.lora_r > 0,
                self.config.lora_alpha > 0,
                self.config.lora_target_modules is not None,
            ]
        ):
            from peft import LoraConfig, TaskType

            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules or "all-linear",
                lora_dropout=self.config.lora_dropout,
                bias=self.config.lora_bias,
                task_type=TaskType.CAUSAL_LM,
            )

        # Create SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        return trainer

    def _create_trl_sft_config(self):
        """Create TRL SFTConfig from our configuration."""
        try:
            from trl import SFTConfig
        except ImportError:
            raise ImportError(
                "TRL library is required for TRL SFT training. Install it with: pip install trl"
            )

        # Map our config to TRL SFTConfig parameters
        config_dict = {
            # Basic training parameters
            "output_dir": self.config.output_dir,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "warmup_ratio": self.config.warmup_ratio,
            "warmup_steps": self.config.warmup_steps,
            "lr_scheduler_type": self.config.lr_scheduler_type,
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "eval_steps": self.config.eval_steps,
            "save_total_limit": self.config.save_total_limit,
            "load_best_model_at_end": self.config.load_best_model_at_end,
            "metric_for_best_model": self.config.metric_for_best_model,
            "greater_is_better": self.config.greater_is_better,
            "gradient_checkpointing": True,
            "dataloader_pin_memory": False,
            "remove_unused_columns": False,
            # TRL-specific parameters
            "max_length": self.config.max_seq_length,
            "packing": self.config.packing,
            "packing_strategy": self.config.packing_strategy,
            "completion_only_loss": self.config.completion_only_loss,
            "assistant_only_loss": self.config.assistant_only_loss,
            "dataset_text_field": self.config.dataset_text_field,
            # Optional TRL features
            "neftune_noise_alpha": self.config.neftune_noise_alpha,
            "use_liger_kernel": self.config.use_liger_kernel,
            "chat_template_path": self.config.chat_template_path,
            "eos_token": self.config.eos_token,
            "pad_token": self.config.pad_token,
            # Logging and monitoring
            "report_to": "wandb" if self.config.use_wandb else None,
            "run_name": f"trl-sft-{self.config.stage_name}",
        }

        # Remove None values
        config_dict = {k: v for k, v in config_dict.items() if v is not None}

        return SFTConfig(**config_dict)
