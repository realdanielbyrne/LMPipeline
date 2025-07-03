"""
Direct Preference Optimization (DPO) stage for the modular pipeline.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments

from .base import BaseStage, StageConfig, StageResult

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig(StageConfig):
    """Configuration for Direct Preference Optimization stage."""

    # Dataset configuration
    preference_dataset_path: str = field(
        default="",
        metadata={"help": "Path to preference dataset (chosen/rejected pairs)"},
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length for tokenization"}
    )

    # DPO-specific parameters
    beta: float = field(
        default=0.1,
        metadata={"help": "DPO beta parameter (KL regularization strength)"},
    )
    reference_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to reference model (if None, uses SFT model)"},
    )

    # Training configuration
    num_train_epochs: int = field(
        default=1, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=2, metadata={"help": "Training batch size per device"}
    )
    learning_rate: float = field(
        default=5e-7, metadata={"help": "Learning rate for DPO training"}
    )
    gradient_accumulation_steps: int = field(
        default=4, metadata={"help": "Gradient accumulation steps"}
    )
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio"})

    # Evaluation
    validation_split: float = field(
        default=0.1, metadata={"help": "Fraction of data to use for validation"}
    )

    # Dataset format detection
    auto_detect_format: bool = field(
        default=True,
        metadata={"help": "Automatically detect and convert preference dataset format"},
    )


class PreferenceDatasetFormatter:
    """Handles automatic detection and conversion of different preference dataset formats."""

    # Common preference dataset format mappings
    FORMAT_MAPPINGS = {
        # Standard prompt-chosen-rejected formats
        ("prompt", "chosen", "rejected"): lambda item: {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        },
        # Instruction-preferred-rejected formats
        ("instruction", "preferred", "rejected"): lambda item: {
            "prompt": item["instruction"],
            "chosen": item["preferred"],
            "rejected": item["rejected"],
        },
        # Question-good-bad formats
        ("question", "good_answer", "bad_answer"): lambda item: {
            "prompt": item["question"],
            "chosen": item["good_answer"],
            "rejected": item["bad_answer"],
        },
        # Input-output_1-output_2 with preference
        ("input", "output_1", "output_2", "preference"): lambda item: {
            "prompt": item["input"],
            "chosen": item["output_1"] if item["preference"] == 1 else item["output_2"],
            "rejected": (
                item["output_2"] if item["preference"] == 1 else item["output_1"]
            ),
        },
        # Conversations format with chosen/rejected
        ("conversations", "chosen", "rejected"): lambda item: {
            "prompt": PreferenceDatasetFormatter._extract_conversation_prompt(
                item["conversations"]
            ),
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        },
    }

    @staticmethod
    def detect_format(data: List[Dict[str, Any]]) -> tuple:
        """
        Detect the format of the preference dataset by examining the first few samples.

        Args:
            data: List of preference dataset samples

        Returns:
            Tuple of column names representing the detected format
        """
        if not data:
            raise ValueError("Preference dataset is empty")

        # Check first few samples to determine format
        sample_size = min(5, len(data))
        common_keys = None

        for i in range(sample_size):
            item = data[i]
            if not isinstance(item, dict):
                raise ValueError(f"Preference dataset item {i} is not a dictionary")

            item_keys = set(item.keys())
            if common_keys is None:
                common_keys = item_keys
            else:
                common_keys = common_keys.intersection(item_keys)

        if not common_keys:
            raise ValueError("No common keys found across preference dataset samples")

        # Check for known formats in order of preference
        for format_keys in PreferenceDatasetFormatter.FORMAT_MAPPINGS.keys():
            if set(format_keys).issubset(common_keys):
                return format_keys

        # If no exact match, try to infer based on common patterns
        if {"prompt", "chosen", "rejected"}.issubset(common_keys):
            return ("prompt", "chosen", "rejected")
        elif {"instruction", "chosen", "rejected"}.issubset(common_keys):
            return ("instruction", "chosen", "rejected")
        elif {"question", "answer_a", "answer_b", "preference"}.issubset(common_keys):
            return ("question", "answer_a", "answer_b", "preference")

        # Last resort: look for any combination that might work
        prompt_keys = [
            k
            for k in common_keys
            if any(
                p in k.lower() for p in ["prompt", "instruction", "question", "input"]
            )
        ]
        chosen_keys = [
            k
            for k in common_keys
            if any(
                c in k.lower()
                for c in ["chosen", "preferred", "good", "positive", "better"]
            )
        ]
        rejected_keys = [
            k
            for k in common_keys
            if any(r in k.lower() for r in ["rejected", "bad", "negative", "worse"])
        ]

        if prompt_keys and chosen_keys and rejected_keys:
            return (prompt_keys[0], chosen_keys[0], rejected_keys[0])

        raise ValueError(
            f"Could not detect preference dataset format. Available keys: {list(common_keys)}"
        )

    @staticmethod
    def convert_to_standard_format(
        item: Dict[str, Any], format_keys: tuple
    ) -> Dict[str, str]:
        """
        Convert a preference dataset item to standard prompt-chosen-rejected format.

        Args:
            item: Preference dataset item
            format_keys: Detected format keys

        Returns:
            Dictionary with 'prompt', 'chosen', and 'rejected' keys
        """
        if format_keys in PreferenceDatasetFormatter.FORMAT_MAPPINGS:
            return PreferenceDatasetFormatter.FORMAT_MAPPINGS[format_keys](item)
        else:
            # Fallback: try to infer the format
            return PreferenceDatasetFormatter._infer_and_convert(item, format_keys)

    @staticmethod
    def _extract_conversation_prompt(conversations: List[Dict[str, str]]) -> str:
        """Extract prompt from conversation format."""
        if not conversations:
            return ""

        # Find the last user message as the prompt
        user_messages = [
            msg["content"] for msg in conversations if msg.get("role") == "user"
        ]
        return (
            user_messages[-1] if user_messages else conversations[0].get("content", "")
        )

    @staticmethod
    def _infer_and_convert(item: Dict[str, Any], format_keys: tuple) -> Dict[str, str]:
        """
        Infer and convert unknown format to standard format.

        Args:
            item: Dataset item
            format_keys: Available keys

        Returns:
            Dictionary with 'prompt', 'chosen', and 'rejected' keys
        """
        # Try to find prompt-like field
        prompt_key = None
        for key in format_keys:
            if any(
                p in key.lower()
                for p in ["prompt", "instruction", "question", "input", "text"]
            ):
                prompt_key = key
                break

        # Try to find chosen/rejected fields
        chosen_key = None
        rejected_key = None

        for key in format_keys:
            if any(
                c in key.lower()
                for c in [
                    "chosen",
                    "preferred",
                    "good",
                    "positive",
                    "better",
                    "response_a",
                ]
            ):
                chosen_key = key
            elif any(
                r in key.lower()
                for r in ["rejected", "bad", "negative", "worse", "response_b"]
            ):
                rejected_key = key

        if prompt_key and chosen_key and rejected_key:
            return {
                "prompt": str(item[prompt_key]),
                "chosen": str(item[chosen_key]),
                "rejected": str(item[rejected_key]),
            }
        else:
            # Last resort: use first three fields
            keys_list = list(format_keys)
            if len(keys_list) >= 3:
                return {
                    "prompt": str(item[keys_list[0]]),
                    "chosen": str(item[keys_list[1]]),
                    "rejected": str(item[keys_list[2]]),
                }
            else:
                raise ValueError(
                    f"Cannot convert preference dataset item with keys: {format_keys}"
                )


class DPODataset(Dataset):
    """Dataset class for DPO training with preference pairs."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        auto_detect_format: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.auto_detect_format = auto_detect_format

        # Set pad token if not exists
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token

        # Detect and log dataset format
        if self.auto_detect_format and data:
            self.detected_format = PreferenceDatasetFormatter.detect_format(data)
            logger.info(f"Detected preference dataset format: {self.detected_format}")

            # Convert first sample to show the transformation
            if len(data) > 0:
                sample_converted = (
                    PreferenceDatasetFormatter.convert_to_standard_format(
                        data[0], self.detected_format
                    )
                )
                logger.info(f"Sample conversion: {data[0]} -> {sample_converted}")
        else:
            self.detected_format = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        item = self.data[idx]

        # Convert to standard format if auto-detection is enabled
        if self.auto_detect_format and self.detected_format:
            try:
                converted_item = PreferenceDatasetFormatter.convert_to_standard_format(
                    item, self.detected_format
                )
            except Exception as e:
                logger.warning(
                    f"Failed to convert preference item {idx}: {e}. Using original format."
                )
                converted_item = item
        else:
            converted_item = item

        # Validate required fields
        if not all(key in converted_item for key in ["prompt", "chosen", "rejected"]):
            raise ValueError(
                f"Preference dataset item {idx} must contain 'prompt', 'chosen', and 'rejected' fields. "
                f"Available keys: {list(converted_item.keys())}"
            )

        # Return the raw strings for TRL's DPOTrainer to handle tokenization
        return {
            "prompt": converted_item["prompt"],
            "chosen": converted_item["chosen"],
            "rejected": converted_item["rejected"],
        }


class DPOStage(BaseStage):
    """Direct Preference Optimization stage implementation."""

    def __init__(self, config: DPOConfig):
        """Initialize the DPO stage."""
        super().__init__(config)
        self.config: DPOConfig = config

    @property
    def stage_name(self) -> str:
        """Return the name of this stage."""
        return "dpo"

    def validate_config(self) -> None:
        """Validate the DPO configuration."""
        if not self.config.preference_dataset_path:
            raise ValueError("preference_dataset_path is required for DPO stage")

        if self.config.beta <= 0:
            raise ValueError("beta must be positive")

        if self.config.validation_split < 0 or self.config.validation_split >= 1:
            raise ValueError("validation_split must be between 0 and 1")

        if self.config.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")

        if self.config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def prepare_model_and_tokenizer(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        previous_result: Optional[StageResult] = None,
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Prepare model and tokenizer for DPO training.

        Args:
            model: Input model from previous stage
            tokenizer: Input tokenizer
            previous_result: Result from previous stage

        Returns:
            Tuple of (prepared_model, prepared_tokenizer)
        """
        self.logger.info("Preparing model and tokenizer for DPO training")

        # Ensure model is in training mode
        model.train()

        # Set pad token if not exists
        if getattr(tokenizer, "pad_token", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is not None:
                tokenizer.pad_token = eos_token
                self.logger.info(f"Set pad_token to eos_token: {eos_token}")
            else:
                # Add a pad token if no eos token exists
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))
                self.logger.info("Added new pad token [PAD]")

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            self.logger.info("Enabled gradient checkpointing")

        # For DPO, we typically want to keep the model in full precision
        # or use the same precision as the input model
        if hasattr(model, "config") and hasattr(model.config, "torch_dtype"):
            self.logger.info(f"Model dtype: {model.config.torch_dtype}")

        # Log model info
        if hasattr(model, "num_parameters"):
            total_params = model.num_parameters()
            trainable_params = model.num_parameters(only_trainable=True)
            self.logger.info(
                f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
            )

        return model, tokenizer

    def _handle_training_error(self, error: Exception) -> str:
        """
        Handle and categorize training errors with helpful messages.

        Args:
            error: The exception that occurred

        Returns:
            A helpful error message for the user
        """
        error_str = str(error).lower()

        if "out of memory" in error_str or "cuda out of memory" in error_str:
            return (
                f"GPU memory error: {error}. "
                "Try reducing batch size, sequence length, or enabling gradient checkpointing."
            )
        elif "no module named 'trl'" in error_str:
            return (
                f"TRL library not found: {error}. " "Install it with: pip install trl"
            )
        elif "dataset" in error_str and ("format" in error_str or "key" in error_str):
            return (
                f"Dataset format error: {error}. "
                "Ensure your preference dataset has 'prompt', 'chosen', and 'rejected' fields, "
                "or enable auto_detect_format for automatic conversion."
            )
        elif "reference model" in error_str or "ref_model" in error_str:
            return (
                f"Reference model error: {error}. "
                "Check the reference_model_path or set it to None to use the input model."
            )
        elif "tokenizer" in error_str:
            return (
                f"Tokenizer error: {error}. "
                "Ensure the tokenizer is compatible with the model and has proper special tokens."
            )
        elif "convergence" in error_str or "nan" in error_str or "inf" in error_str:
            return (
                f"Training convergence error: {error}. "
                "Try reducing learning rate, adjusting beta parameter, or checking data quality."
            )
        else:
            return f"DPO training error: {error}"

    def execute(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        previous_result: Optional[StageResult] = None,
    ) -> StageResult:
        """Execute the DPO training stage."""
        try:
            self.logger.info("Starting DPO training")
            self.setup_logging()

            # Validate configuration
            self.validate_config()

            # Load and prepare preference dataset
            self.logger.info("Loading preference dataset...")
            preference_data = self._load_preference_dataset()
            train_data, val_data = self._split_preference_dataset(preference_data)

            # Create datasets
            self.logger.info("Creating DPO datasets...")
            train_dataset = self._create_preference_dataset(train_data, tokenizer)
            eval_dataset = None
            if val_data:
                eval_dataset = self._create_preference_dataset(val_data, tokenizer)

            # Prepare model and tokenizer for DPO training
            model, tokenizer = self.prepare_model_and_tokenizer(
                model, tokenizer, previous_result
            )

            # Create DPO trainer
            self.logger.info("Creating DPO trainer...")
            trainer = self._create_dpo_trainer(
                model, tokenizer, train_dataset, eval_dataset
            )

            # Start training
            self.logger.info("Starting DPO training...")
            train_result = trainer.train()

            # Extract metrics from training
            metrics = {}
            if hasattr(train_result, "metrics"):
                metrics.update(train_result.metrics)

            # Get final metrics from trainer state
            if trainer.state.log_history:
                final_metrics = trainer.state.log_history[-1]
                for key, value in final_metrics.items():
                    if isinstance(value, (int, float)) and not key.startswith("epoch"):
                        metrics[key] = value

            # Save the trained model
            self.logger.info("Saving DPO model...")
            model_path, tokenizer_path = self.save_model_and_tokenizer(model, tokenizer)

            # Log final metrics
            self.logger.info(f"DPO training completed. Final metrics: {metrics}")

            self.cleanup_logging()

            return StageResult(
                stage_name=self.stage_name,
                success=True,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                metrics=metrics,
                artifacts={
                    "training_logs": str(Path(self.config.output_dir) / "logs"),
                    "checkpoints": str(Path(self.config.output_dir) / "checkpoints"),
                },
            )

        except Exception as e:
            # Use enhanced error handling
            error_message = self._handle_training_error(e)
            self.logger.error(error_message)
            self.cleanup_logging()

            return StageResult(
                stage_name=self.stage_name,
                success=False,
                model_path="",
                tokenizer_path="",
                error_message=error_message,
            )

    def _load_preference_dataset(self) -> List[Dict[str, Any]]:
        """Load preference dataset with chosen/rejected pairs."""
        dataset_path = self.config.preference_dataset_path

        try:
            if os.path.isfile(dataset_path):
                self.logger.info(
                    f"Loading preference dataset from local file: {dataset_path}"
                )

                # Load from local file
                if dataset_path.endswith(".json"):
                    try:
                        with open(dataset_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON format in {dataset_path}: {e}")
                elif dataset_path.endswith(".jsonl"):
                    data = []
                    try:
                        with open(dataset_path, "r", encoding="utf-8") as f:
                            for line_num, line in enumerate(f, 1):
                                if line.strip():  # Skip empty lines
                                    try:
                                        data.append(json.loads(line.strip()))
                                    except json.JSONDecodeError as e:
                                        raise ValueError(
                                            f"Invalid JSON on line {line_num} in {dataset_path}: {e}"
                                        )
                    except FileNotFoundError:
                        raise ValueError(
                            f"Preference dataset file not found: {dataset_path}"
                        )
                else:
                    raise ValueError(
                        f"Unsupported file format: {dataset_path}. Supported formats: .json, .jsonl"
                    )
            else:
                self.logger.info(
                    f"Loading preference dataset from HuggingFace hub: {dataset_path}"
                )

                # Load from HuggingFace hub
                try:
                    dataset = load_dataset(dataset_path, split="train")
                    data = [item for item in dataset]
                except Exception as e:
                    # Try with different split names
                    try:
                        dataset = load_dataset(dataset_path, split="preference")
                        data = [item for item in dataset]
                        self.logger.info("Using 'preference' split from dataset")
                    except Exception:
                        # Try loading all splits and use the first one
                        try:
                            dataset = load_dataset(dataset_path)
                            if not dataset:
                                raise ValueError(
                                    f"Dataset {dataset_path} has no splits"
                                )
                            # Get the first available split
                            split_name = list(dataset.keys())[0]
                            data = [item for item in dataset[split_name]]
                            self.logger.info(f"Using split '{split_name}' from dataset")
                        except Exception as final_e:
                            raise ValueError(
                                f"Failed to load preference dataset from {dataset_path}. "
                                f"Tried splits: 'train', 'preference', and first available. "
                                f"Original error: {e}. Final error: {final_e}"
                            )

            self.logger.info(f"Loaded {len(data)} preference examples")

            # Validate that we have data
            if not data:
                raise ValueError("Preference dataset is empty")

            # Ensure type
            if not isinstance(data, list):
                raise ValueError("Loaded preference data is not a list")

            # Validate format if auto-detection is enabled
            if self.config.auto_detect_format:
                try:
                    detected_format = PreferenceDatasetFormatter.detect_format(data)
                    self.logger.info(
                        f"Detected preference dataset format: {detected_format}"
                    )

                    # Validate that we can convert at least the first few samples
                    for i in range(min(3, len(data))):
                        try:
                            PreferenceDatasetFormatter.convert_to_standard_format(
                                data[i], detected_format
                            )
                        except Exception as conv_e:
                            raise ValueError(
                                f"Failed to convert sample {i} to standard format: {conv_e}. "
                                f"Sample: {data[i]}"
                            )

                except Exception as e:
                    raise ValueError(f"Dataset format validation failed: {e}")

            return data

        except Exception as e:
            # Re-raise with more context if it's not already a ValueError
            if not isinstance(e, ValueError):
                raise ValueError(f"Failed to load preference dataset: {e}")
            else:
                raise

    def _split_preference_dataset(
        self, data: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split preference dataset into train and validation sets."""
        if self.config.validation_split <= 0:
            return data, []

        split_idx = int(len(data) * (1 - self.config.validation_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        self.logger.info(
            f"Split preference dataset: {len(train_data)} train, {len(val_data)} validation"
        )
        return train_data, val_data

    def _create_preference_dataset(
        self, data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase
    ) -> DPODataset:
        """Create DPO dataset from preference data."""
        return DPODataset(
            data=data,
            tokenizer=tokenizer,
            max_length=self.config.max_seq_length,
            auto_detect_format=self.config.auto_detect_format,
        )

    def _create_dpo_trainer(self, model, tokenizer, train_dataset, eval_dataset=None):
        """Create DPO trainer using TRL."""
        try:
            from trl import DPOTrainer
        except ImportError:
            raise ImportError(
                "TRL library is required for DPO training. Install it with: pip install trl"
            )

        # Create training arguments
        training_args = self._create_training_arguments()

        # Load reference model if specified, otherwise use the input model
        if self.config.reference_model_path:
            self.logger.info(
                f"Loading reference model from {self.config.reference_model_path}"
            )
            try:
                from transformers import AutoModelForCausalLM

                ref_model = AutoModelForCausalLM.from_pretrained(
                    self.config.reference_model_path,
                    torch_dtype=(
                        torch.float16 if torch.cuda.is_available() else torch.float32
                    ),
                    device_map="auto" if torch.cuda.is_available() else None,
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to load reference model: {e}. Using input model as reference."
                )
                ref_model = None
        else:
            self.logger.info("Using input model as reference model")
            ref_model = None

        # Create DPO trainer
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            beta=self.config.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=self.config.max_seq_length,
            max_prompt_length=self.config.max_seq_length // 2,  # Use half for prompt
        )

        return trainer

    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments for DPO trainer."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_steps=500,
            eval_steps=500 if self.config.validation_split > 0 else None,
            evaluation_strategy="steps" if self.config.validation_split > 0 else "no",
            save_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True if self.config.validation_split > 0 else False,
            metric_for_best_model=(
                "eval_loss" if self.config.validation_split > 0 else None
            ),
            greater_is_better=False,
            report_to="wandb" if getattr(self.config, "use_wandb", False) else None,
            run_name=f"dpo-{getattr(self.config, 'stage_name', 'unknown')}",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
        )
