"""
Shared model utilities for quantization, LoRA setup, and model management.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

logger = logging.getLogger(__name__)


def load_quantization_config(
    use_4bit: bool = True,
    use_8bit: bool = False,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> Optional[BitsAndBytesConfig]:
    """Load quantization configuration."""
    if not (use_4bit or use_8bit):
        return None

    if use_8bit:
        logger.info("Using 8-bit quantization")
        return BitsAndBytesConfig(load_in_8bit=True)

    logger.info("Using 4-bit quantization")
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def setup_lora(
    model: Any,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[List[str]] = None,
    lora_bias: str = "none",
) -> Any:
    """Setup LoRA configuration for the model."""
    logger.info("Setting up LoRA configuration")

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Auto-detect target modules if not specified
    target_modules = lora_target_modules
    if target_modules is None:
        # Common target modules for different architectures
        if hasattr(model, "config") and hasattr(model.config, "model_type"):
            model_type = model.config.model_type.lower()
            if "llama" in model_type or "mistral" in model_type:
                target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            elif "gpt" in model_type:
                target_modules = ["c_attn", "c_proj", "c_fc"]
            else:
                # Fallback: find all linear layers
                target_modules = []
                if hasattr(model, "named_modules"):
                    for name, module in model.named_modules():
                        if isinstance(module, nn.Linear):
                            target_modules.append(name.split(".")[-1])
                target_modules = list(set(target_modules))

        logger.info(f"Auto-detected target modules: {target_modules}")

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,  # type: ignore[arg-type]
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return model


def load_dataset_from_path(
    dataset_name_or_path: str,
    dataset_config_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load dataset from local file or HuggingFace hub."""
    if os.path.isfile(dataset_name_or_path):
        logger.info(f"Loading dataset from local file: {dataset_name_or_path}")
        # Load from local file
        if dataset_name_or_path.endswith(".json"):
            with open(dataset_name_or_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif dataset_name_or_path.endswith(".jsonl"):
            data = []
            with open(dataset_name_or_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"Unsupported file format: {dataset_name_or_path}")
    else:
        logger.info(f"Loading dataset from HuggingFace hub: {dataset_name_or_path}")
        dataset = load_dataset(dataset_name_or_path, dataset_config_name, split="train")
        data = [item for item in dataset]
    
    logger.info(f"Loaded {len(data)} examples")
    return data  # type: ignore


def split_dataset(
    data: List[Dict[str, Any]], validation_split: float = 0.1
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split dataset into train and validation sets."""
    if validation_split <= 0:
        return data, []

    split_idx = int(len(data) * (1 - validation_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data


def save_model_and_tokenizer(
    model: Any, tokenizer: Any, output_dir: str
) -> None:
    """Save the fine-tuned model and tokenizer."""
    logger.info(f"Saving model to {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save model and tokenizer
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    logger.info("Model and tokenizer saved successfully")
