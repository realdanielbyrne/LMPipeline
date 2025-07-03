"""
Shared model utilities for quantization, LoRA setup, and model management.
"""

import json
import logging
import os
import platform
from typing import Any, Dict, List, Optional, Tuple

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


def get_optimal_device() -> Tuple[torch.device, str]:
    """
    Detect and return the optimal device for training/inference.

    Handles compatibility warnings (e.g., RTX 5090 sm_120 compute capability)
    and provides graceful fallbacks.

    Returns:
        Tuple of (device, device_name) where device_name is human-readable
    """
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            try:
                # This may trigger warnings for newer GPUs like RTX 5090
                device_name = f"CUDA ({torch.cuda.get_device_name()})"
                capability = torch.cuda.get_device_capability()

                # Check for newer compute capabilities that may have warnings
                if capability[0] >= 12:  # sm_120 and above
                    logger.info(
                        f"Using CUDA device: {device_name} (Compute Capability: sm_{capability[0]}{capability[1]})"
                    )
                    logger.info(
                        "Note: Newer GPU detected - some PyTorch features may show compatibility warnings but device is usable"
                    )
                else:
                    logger.info(f"Using CUDA device: {device_name}")

            except Exception as e:
                # Fallback if device name detection fails
                device_name = "CUDA (Unknown GPU)"
                logger.warning(f"Could not get CUDA device name: {e}")

            return device, device_name

    except Exception as e:
        logger.warning(f"Error checking CUDA availability: {e}")

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple Silicon MPS"
            logger.info(f"Using MPS device: {device_name}")
            return device, device_name
    except Exception as e:
        logger.warning(f"Error checking MPS availability: {e}")

    # Fallback to CPU
    device = torch.device("cpu")
    try:
        device_name = f"CPU ({platform.processor()})"
    except Exception:
        device_name = "CPU (Unknown)"
    logger.info(f"Using CPU device: {device_name}")
    return device, device_name


def is_quantization_supported() -> bool:
    """
    Check if quantization (BitsAndBytes) is supported on current platform.

    Returns:
        True if quantization is supported, False otherwise
    """
    try:
        import bitsandbytes

        # BitsAndBytes requires CUDA and is not supported on ARM64 (Apple Silicon)
        if torch.cuda.is_available() and platform.machine().lower() != "arm64":
            logger.info("Quantization (BitsAndBytes) is supported")
            return True
        else:
            if platform.machine().lower() == "arm64":
                logger.info(
                    "BitsAndBytes not supported on ARM64 (Apple Silicon) - quantization disabled"
                )
            else:
                logger.warning(
                    "Quantization (BitsAndBytes) requires CUDA - not available on this platform"
                )
            return False
    except ImportError as e:
        logger.warning(f"BitsAndBytes not installed - quantization not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error checking quantization support: {e}")
        return False


def get_recommended_dtype(torch_dtype: Optional[str] = None) -> torch.dtype:
    """
    Get the recommended torch dtype based on the current device or user specification.

    Args:
        torch_dtype: Optional dtype specification. Can be "auto", "float16", "bfloat16", "float32", or None

    Returns:
        Recommended torch dtype
    """
    # Handle automatic dtype selection
    if torch_dtype is None or torch_dtype == "auto":
        device, _ = get_optimal_device()

        if device.type == "cuda":
            # CUDA supports bfloat16 on modern GPUs
            try:
                if torch.cuda.is_bf16_supported():
                    logger.info("Using bfloat16 for CUDA (auto-detected)")
                    return torch.bfloat16
                else:
                    logger.info("Using float16 for CUDA (auto-detected)")
                    return torch.float16
            except Exception as e:
                logger.warning(
                    f"Error checking CUDA bf16 support: {e}, falling back to float16"
                )
                return torch.float16
        elif device.type == "mps":
            # MPS works well with float16
            logger.info("Using float16 for MPS (auto-detected)")
            return torch.float16
        else:
            # CPU typically uses float32
            logger.info("Using float32 for CPU (auto-detected)")
            return torch.float32

    # Handle explicit dtype specification
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if torch_dtype in dtype_mapping:
        selected_dtype = dtype_mapping[torch_dtype]
        logger.info(f"Using {torch_dtype} (user-specified)")
        return selected_dtype
    else:
        logger.warning(f"Unknown dtype '{torch_dtype}', falling back to auto-detection")
        return get_recommended_dtype("auto")


def load_quantization_config(
    use_4bit: bool = True,
    use_8bit: bool = False,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> Optional[BitsAndBytesConfig]:
    """
    Load quantization configuration with cross-platform support.

    Automatically disables quantization on platforms that don't support it (e.g., Apple Silicon).
    """
    if not (use_4bit or use_8bit):
        return None

    # Check if quantization is supported on current platform
    if not is_quantization_supported():
        logger.warning(
            "Quantization requested but not supported on this platform - disabling"
        )
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


def save_model_and_tokenizer(model: Any, tokenizer: Any, output_dir: str) -> None:
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
