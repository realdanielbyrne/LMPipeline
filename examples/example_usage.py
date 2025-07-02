#!/usr/bin/env python3
"""
Example usage scripts for the SFT trainer.

This file contains examples of how to use the supervised fine-tuning script
with different configurations and datasets.
"""

import subprocess
import sys
from pathlib import Path

def run_basic_training():
    """Example: Basic training with a small model and dataset."""
    cmd = [
        sys.executable, "-m", "fnsft.sft_trainer",
        "--model_name_or_path", "microsoft/DialoGPT-small",
        "--dataset_name_or_path", "tatsu-lab/alpaca",
        "--output_dir", "./outputs/basic_training",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "2e-4",
        "--max_seq_length", "512",
        "--validation_split", "0.1",
        "--logging_steps", "10",
        "--save_steps", "100",
        "--eval_steps", "100",
    ]
    
    print("Running basic training example...")
    print(" ".join(cmd))
    subprocess.run(cmd)


def run_llama_training():
    """Example: Training with Llama model and custom dataset."""
    cmd = [
        sys.executable, "-m", "fnsft.sft_trainer",
        "--model_name_or_path", "meta-llama/Llama-2-7b-hf",
        "--dataset_name_or_path", "./data/custom_instructions.jsonl",
        "--output_dir", "./outputs/llama_training",
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "8",
        "--learning_rate", "1e-4",
        "--max_seq_length", "2048",
        "--use_4bit",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--lora_dropout", "0.1",
        "--validation_split", "0.05",
        "--warmup_ratio", "0.03",
        "--weight_decay", "0.001",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "5",
        "--save_steps", "250",
        "--eval_steps", "250",
        "--save_total_limit", "3",
        "--load_best_model_at_end",
        "--metric_for_best_model", "eval_loss",
        "--use_wandb",
        "--wandb_project", "llama-sft",
        "--convert_to_gguf",
        "--gguf_quantization", "q4_0",
    ]
    
    print("Running Llama training example...")
    print(" ".join(cmd))
    subprocess.run(cmd)


def run_mistral_training():
    """Example: Training with Mistral model and 8-bit quantization."""
    cmd = [
        sys.executable, "-m", "fnsft.sft_trainer",
        "--model_name_or_path", "mistralai/Mistral-7B-v0.1",
        "--dataset_name_or_path", "OpenAssistant/oasst1",
        "--dataset_config_name", "default",
        "--output_dir", "./outputs/mistral_training",
        "--num_train_epochs", "2",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "5e-5",
        "--max_seq_length", "1024",
        "--use_8bit",
        "--lora_r", "32",
        "--lora_alpha", "64",
        "--lora_dropout", "0.05",
        "--lora_target_modules", "q_proj", "k_proj", "v_proj", "o_proj",
        "--validation_split", "0.1",
        "--torch_dtype", "bfloat16",
        "--warmup_ratio", "0.05",
        "--weight_decay", "0.01",
        "--logging_steps", "20",
        "--save_steps", "500",
        "--eval_steps", "500",
        "--instruction_template", "### Human: {instruction}\n\n### Assistant: {response}",
    ]
    
    print("Running Mistral training example...")
    print(" ".join(cmd))
    subprocess.run(cmd)


def run_resume_training():
    """Example: Resume training from a checkpoint."""
    cmd = [
        sys.executable, "-m", "fnsft.sft_trainer",
        "--model_name_or_path", "microsoft/DialoGPT-medium",
        "--dataset_name_or_path", "tatsu-lab/alpaca",
        "--output_dir", "./outputs/resumed_training",
        "--resume_from_checkpoint", "./outputs/basic_training/checkpoint-500",
        "--num_train_epochs", "5",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "1e-4",
        "--max_seq_length", "1024",
        "--validation_split", "0.1",
        "--logging_steps", "10",
        "--save_steps", "200",
        "--eval_steps", "200",
    ]
    
    print("Running resume training example...")
    print(" ".join(cmd))
    subprocess.run(cmd)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SFT training examples")
    parser.add_argument("--example", type=str, choices=["basic", "llama", "mistral", "resume"],
                       default="basic", help="Which example to run")
    
    args = parser.parse_args()
    
    if args.example == "basic":
        run_basic_training()
    elif args.example == "llama":
        run_llama_training()
    elif args.example == "mistral":
        run_mistral_training()
    elif args.example == "resume":
        run_resume_training()
    else:
        print(f"Unknown example: {args.example}")
        sys.exit(1)
