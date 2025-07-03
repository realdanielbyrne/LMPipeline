"""
Supervised Fine-Tuning (SFT) stage for the modular pipeline.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

from .base import BaseStage, StageConfig, StageResult
from ..utils.model_utils import (
    load_quantization_config,
    setup_lora,
    load_dataset_from_path,
    split_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig(StageConfig):
    """Configuration for Supervised Fine-Tuning stage."""

    # Dataset configuration
    dataset_name_or_path: str = field(
        default="",
        metadata={"help": "Path to dataset file or HuggingFace dataset name"},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "Configuration name for HuggingFace dataset"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length for tokenization"}
    )
    instruction_template: str = field(
        default="### Instruction:\n{instruction}\n\n### Response:\n{response}",
        metadata={"help": "Template for formatting instruction-response pairs"},
    )
    validation_split: float = field(
        default=0.1, metadata={"help": "Fraction of data to use for validation"}
    )
    auto_detect_format: bool = field(
        default=True,
        metadata={"help": "Automatically detect and convert dataset format"},
    )

    # Quantization configuration
    use_4bit: bool = field(default=True, metadata={"help": "Use 4-bit quantization"})
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit quantization (overrides 4-bit if True)"},
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16", metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4", metadata={"help": "Quantization type for 4-bit (nf4, fp4)"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True, metadata={"help": "Use double quantization for 4-bit"}
    )

    # LoRA configuration
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Target modules for LoRA (auto-detected if None)"},
    )
    lora_bias: str = field(
        default="none", metadata={"help": "LoRA bias type (none, all, lora_only)"}
    )

    # Training configuration
    num_train_epochs: int = field(
        default=3, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Training batch size per device"}
    )
    per_device_eval_batch_size: int = field(
        default=4, metadata={"help": "Evaluation batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(default=2e-4, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.001, metadata={"help": "Weight decay"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup ratio"})
    lr_scheduler_type: str = field(
        default="cosine", metadata={"help": "Learning rate scheduler type"}
    )
    logging_steps: int = field(default=10, metadata={"help": "Logging steps"})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint steps"})
    eval_steps: int = field(default=500, metadata={"help": "Evaluation steps"})
    save_total_limit: int = field(
        default=3, metadata={"help": "Maximum number of checkpoints to keep"}
    )
    load_best_model_at_end: bool = field(
        default=True, metadata={"help": "Load best model at end of training"}
    )
    metric_for_best_model: str = field(
        default="eval_loss", metadata={"help": "Metric for best model selection"}
    )
    greater_is_better: bool = field(
        default=False, metadata={"help": "Whether higher metric is better"}
    )

    # Additional options
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Resume training from checkpoint"}
    )


class SFTStage(BaseStage):
    """Supervised Fine-Tuning stage implementation."""

    def __init__(self, config: SFTConfig):
        """Initialize the SFT stage."""
        super().__init__(config)
        self.config: SFTConfig = config

    @property
    def stage_name(self) -> str:
        """Return the name of this stage."""
        return "sft"

    def validate_config(self) -> None:
        """Validate the SFT configuration."""
        if not self.config.dataset_name_or_path:
            raise ValueError("dataset_name_or_path is required for SFT stage")

        if self.config.validation_split < 0 or self.config.validation_split >= 1:
            raise ValueError("validation_split must be between 0 and 1")

        if self.config.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")

        if self.config.lora_r <= 0:
            raise ValueError("lora_r must be positive")

    def prepare_model_and_tokenizer(
        self,
        model: AutoModelForCausalLM,  # type: ignore
        tokenizer: AutoTokenizer,  # type: ignore
        previous_result: Optional[StageResult] = None,
    ) -> tuple[Any, Any]:
        """Prepare model and tokenizer for SFT training."""
        self.logger.info("Preparing model for SFT training")

        # Apply quantization if needed
        if self.config.use_4bit or self.config.use_8bit:
            # Note: Quantization is typically applied during model loading
            # This is kept for compatibility but may not be needed
            pass

        # Setup LoRA
        model = setup_lora(
            model=model,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target_modules=self.config.lora_target_modules,
            lora_bias=self.config.lora_bias,
        )

        return model, tokenizer

    def execute(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        previous_result: Optional[StageResult] = None,
    ) -> StageResult:
        """Execute the SFT training stage."""
        try:
            self.logger.info("Starting SFT training")
            self.setup_logging()

            # Load and prepare dataset
            data = load_dataset_from_path(
                dataset_name_or_path=self.config.dataset_name_or_path,
                dataset_config_name=self.config.dataset_config_name,
            )
            train_data, val_data = split_dataset(data, self.config.validation_split)

            # Create datasets
            train_dataset = self._create_instruction_dataset(train_data, tokenizer)
            eval_dataset = None
            if val_data:
                eval_dataset = self._create_instruction_dataset(val_data, tokenizer)

            # Create training arguments
            training_args = self._create_training_arguments()

            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )

            # Create trainer
            trainer = self._create_trainer(
                model,
                tokenizer,
                train_dataset,
                eval_dataset,
                training_args,
                data_collator,
            )

            # Start training
            self.logger.info("Starting training...")
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
            self.logger.error(f"SFT training failed: {e}")
            self.cleanup_logging()

            return StageResult(
                stage_name=self.stage_name,
                success=False,
                model_path="",
                tokenizer_path="",
                error_message=str(e),
            )

    # Removed duplicate methods - now using shared utilities from utils.model_utils

    def _create_instruction_dataset(
        self, data: List[Dict[str, Any]], tokenizer: Any
    ) -> Dataset:
        """Create instruction dataset from data."""
        from ..utils.dataset_utils import InstructionDataset

        return InstructionDataset(
            data=data,
            tokenizer=tokenizer,
            max_length=self.config.max_seq_length,
            instruction_template=self.config.instruction_template,
            auto_detect_format=self.config.auto_detect_format,
        )

    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments for the trainer."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to="wandb" if getattr(self.config, "use_wandb", False) else None,
            run_name=f"sft-{getattr(self.config, 'stage_name', 'unknown')}",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
        )

    def _create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        training_args: TrainingArguments,
        data_collator: DataCollatorForLanguageModeling,
    ) -> Trainer:
        """Create and configure the trainer."""
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        return trainer
