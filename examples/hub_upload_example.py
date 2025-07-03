#!/usr/bin/env python3
"""
Example script demonstrating Hugging Face Hub upload functionality.

This script shows different ways to upload fine-tuned models to the Hugging Face Hub
using the new upload functionality in the SFT trainer.
"""

import os
import sys
import subprocess
from pathlib import Path


def example_basic_training_with_hub_upload():
    """Example: Basic training with automatic Hub upload."""
    print("=" * 60)
    print("Example 1: Basic Training with Hub Upload")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m",
        "lmpipeline.sft_trainer",
        "--model_name_or_path",
        "microsoft/DialoGPT-small",
        "--dataset_name_or_path",
        "tatsu-lab/alpaca",
        "--output_dir",
        "./outputs/basic_training_hub",
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "4",
        "--learning_rate",
        "2e-4",
        "--max_seq_length",
        "512",
        "--validation_split",
        "0.1",
        "--logging_steps",
        "10",
        "--save_steps",
        "100",
        "--eval_steps",
        "100",
        # Hub upload options
        "--push_to_hub",
        "--hub_repo_id",
        "your-username/dialogpt-small-alpaca",  # Change this!
        "--hub_commit_message",
        "Fine-tuned DialoGPT on Alpaca dataset",
        "--hub_private",  # Make it private initially
    ]

    print("Command to run:")
    print(" ".join(cmd))
    print("\nNote: Change 'your-username' to your actual Hugging Face username!")
    print(
        "Make sure you have HF_TOKEN environment variable set or run 'huggingface-cli login' first."
    )

    # Uncomment the line below to actually run the training
    # subprocess.run(cmd)


def example_lora_adapter_only_upload():
    """Example: Upload only LoRA adapters to Hub."""
    print("\n" + "=" * 60)
    print("Example 2: LoRA Adapter-Only Upload")
    print("=" * 60)

    cmd = [
        sys.executable,
        "-m",
        "lmpipeline.sft_trainer",
        "--model_name_or_path",
        "meta-llama/Llama-2-7b-hf",
        "--dataset_name_or_path",
        "tatsu-lab/alpaca",
        "--output_dir",
        "./outputs/llama_lora_adapters",
        "--num_train_epochs",
        "2",
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "8",
        "--learning_rate",
        "1e-4",
        "--max_seq_length",
        "2048",
        "--use_4bit",
        "--lora_r",
        "16",
        "--lora_alpha",
        "32",
        "--lora_dropout",
        "0.1",
        # Hub upload options - only adapters
        "--push_to_hub",
        "--hub_repo_id",
        "your-username/llama-7b-alpaca-lora",  # Change this!
        "--hub_commit_message",
        "LoRA adapters for Llama-7B fine-tuned on Alpaca",
        "--push_adapter_only",  # Only upload LoRA files
        "--hub_private",
    ]

    print("Command to run:")
    print(" ".join(cmd))
    print(
        "\nThis will only upload the LoRA adapter files (adapter_config.json, adapter_model.safetensors)"
    )
    print("Users can then load these adapters with the base Llama-7B model.")


def example_config_file_with_hub_upload():
    """Example: Using YAML config file with Hub upload."""
    print("\n" + "=" * 60)
    print("Example 3: Using Config File with Hub Upload")
    print("=" * 60)

    # Create a sample config file
    config_content = """
# Fine-tuning configuration with Hub upload
model_name_or_path: "microsoft/DialoGPT-medium"
dataset_name_or_path: "tatsu-lab/alpaca"
output_dir: "./outputs/config_based_training"

# Training settings
num_train_epochs: 2
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-4
max_seq_length: 1024
validation_split: 0.1

# LoRA settings
use_4bit: true
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05

# Logging
logging_steps: 20
save_steps: 200
eval_steps: 200

# Hub upload settings
push_to_hub: true
hub_repo_id: "your-username/dialogpt-medium-alpaca"  # Change this!
hub_commit_message: "Fine-tuned DialoGPT-medium on Alpaca dataset with LoRA"
hub_private: false
push_adapter_only: false
"""

    config_file = "./examples/hub_upload_config.yaml"
    with open(config_file, "w") as f:
        f.write(config_content.strip())

    print(f"Created config file: {config_file}")
    print("\nConfig file contents:")
    print(config_content)

    cmd = [sys.executable, "-m", "lmpipeline.sft_trainer", "--config", config_file]

    print("Command to run:")
    print(" ".join(cmd))
    print("\nRemember to update the hub_repo_id in the config file!")


def example_environment_setup():
    """Show how to set up environment for Hub uploads."""
    print("\n" + "=" * 60)
    print("Environment Setup for Hub Uploads")
    print("=" * 60)

    print("Before using Hub upload functionality, you need to authenticate:")
    print()
    print("Option 1: Set environment variable")
    print("export HF_TOKEN='your_huggingface_token_here'")
    print()
    print("Option 2: Use Hugging Face CLI")
    print("pip install huggingface_hub")
    print("huggingface-cli login")
    print()
    print("Option 3: Pass token directly")
    print("--hub_token your_token_here")
    print()
    print("To get a token:")
    print("1. Go to https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'write' permissions")
    print("3. Copy the token and use one of the methods above")


def example_post_training_upload():
    """Example: Upload an already trained model."""
    print("\n" + "=" * 60)
    print("Example 4: Upload Already Trained Model")
    print("=" * 60)

    print("If you have already trained a model and want to upload it later:")
    print()

    upload_script = """
#!/usr/bin/env python3
# Note: Hub upload functionality is now integrated into the pipeline
# Use the pipeline with --push_to_hub flag instead

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./outputs/final_model")

# Upload to Hub
upload_to_hub(
    model_path="./outputs/final_model",
    tokenizer=tokenizer,
    repo_id="your-username/my-fine-tuned-model",
    commit_message="Upload fine-tuned model",
    private=False,
    push_adapter_only=False  # Set to True for LoRA adapters only
)
"""

    print("Python script example:")
    print(upload_script)

    script_file = "./examples/upload_existing_model.py"
    with open(script_file, "w") as f:
        f.write(upload_script.strip())

    print(f"\nSaved example script to: {script_file}")


def main():
    """Run all examples."""
    print("Hugging Face Hub Upload Examples")
    print("================================")
    print()
    print("This script demonstrates how to use the new Hub upload functionality")
    print(
        "in the SFT trainer. The examples show different scenarios and configurations."
    )
    print()

    # Run all examples
    example_environment_setup()
    example_basic_training_with_hub_upload()
    example_lora_adapter_only_upload()
    example_config_file_with_hub_upload()
    example_post_training_upload()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Key features of the Hub upload functionality:")
    print("• Automatic authentication handling (token or login)")
    print("• Support for both full model and LoRA adapter uploads")
    print("• Automatic repository creation if it doesn't exist")
    print("• Comprehensive error handling and logging")
    print("• Integration with existing training workflow")
    print("• YAML configuration file support")
    print()
    print("Remember to:")
    print("1. Set up authentication (HF_TOKEN or huggingface-cli login)")
    print("2. Update repository IDs to use your username/organization")
    print("3. Choose appropriate privacy settings for your models")


if __name__ == "__main__":
    main()
