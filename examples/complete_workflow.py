#!/usr/bin/env python3
"""
Complete workflow example for supervised fine-tuning.

This script demonstrates a complete workflow from data preparation
to model training and evaluation.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """Create a sample instruction dataset for training."""
    
    # Sample instruction-response pairs
    sample_instructions = [
        {
            "instruction": "What is the capital of {country}?",
            "response": "The capital of {country} is {capital}.",
            "countries": [
                ("France", "Paris"),
                ("Germany", "Berlin"),
                ("Italy", "Rome"),
                ("Spain", "Madrid"),
                ("Japan", "Tokyo"),
                ("Brazil", "Brasília"),
                ("Canada", "Ottawa"),
                ("Australia", "Canberra"),
                ("India", "New Delhi"),
                ("China", "Beijing")
            ]
        },
        {
            "instruction": "Explain the concept of {concept} in simple terms.",
            "response": "{concept} is a fundamental concept that involves {explanation}.",
            "concepts": [
                ("gravity", "the force that attracts objects toward each other"),
                ("photosynthesis", "the process by which plants convert sunlight into energy"),
                ("democracy", "a system of government where people vote for their representatives"),
                ("evolution", "the process by which species change over time through natural selection"),
                ("artificial intelligence", "computer systems that can perform tasks requiring human intelligence")
            ]
        },
        {
            "instruction": "How do you {action}?",
            "response": "To {action}, you should follow these steps: {steps}",
            "actions": [
                ("bake a cake", "1. Preheat oven, 2. Mix ingredients, 3. Pour into pan, 4. Bake for specified time"),
                ("learn a language", "1. Start with basics, 2. Practice daily, 3. Immerse yourself, 4. Use language apps"),
                ("plant a garden", "1. Choose location, 2. Prepare soil, 3. Select plants, 4. Water regularly"),
                ("write a resume", "1. List experience, 2. Highlight skills, 3. Format professionally, 4. Proofread carefully"),
                ("save money", "1. Create budget, 2. Track expenses, 3. Reduce unnecessary spending, 4. Set savings goals")
            ]
        }
    ]
    
    dataset = []
    
    # Generate samples
    for i in range(num_samples):
        template = sample_instructions[i % len(sample_instructions)]
        
        if "countries" in template:
            country, capital = template["countries"][i % len(template["countries"])]
            instruction = template["instruction"].format(country=country)
            response = template["response"].format(country=country, capital=capital)
        elif "concepts" in template:
            concept, explanation = template["concepts"][i % len(template["concepts"])]
            instruction = template["instruction"].format(concept=concept)
            response = template["response"].format(concept=concept, explanation=explanation)
        elif "actions" in template:
            action, steps = template["actions"][i % len(template["actions"])]
            instruction = template["instruction"].format(action=action)
            response = template["response"].format(action=action, steps=steps)
        
        dataset.append({
            "instruction": instruction,
            "response": response
        })
    
    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created dataset with {len(dataset)} samples at {output_path}")


def run_training_experiment():
    """Run a complete training experiment."""
    
    # Configuration
    model_name = "microsoft/DialoGPT-small"  # Small model for quick testing
    dataset_path = "./data/sample_training_data.jsonl"
    output_dir = "./outputs/complete_workflow_experiment"
    
    print("=== Complete SFT Workflow Example ===")
    print()
    
    # Step 1: Create sample dataset
    print("Step 1: Creating sample dataset...")
    create_sample_dataset(dataset_path, num_samples=50)
    print()
    
    # Step 2: Run training
    print("Step 2: Starting training...")
    training_cmd = [
        sys.executable, "-m", "lmpipeline.sft_trainer",
        "--model_name_or_path", model_name,
        "--dataset_name_or_path", dataset_path,
        "--output_dir", output_dir,
        "--num_train_epochs", "1",  # Quick training for demo
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "2",
        "--learning_rate", "5e-5",
        "--max_seq_length", "512",
        "--validation_split", "0.2",
        "--logging_steps", "5",
        "--save_steps", "20",
        "--eval_steps", "20",
        "--lora_r", "8",
        "--lora_alpha", "16",
        "--lora_dropout", "0.1",
        "--warmup_ratio", "0.1",
        "--weight_decay", "0.01",
        "--load_best_model_at_end",
        "--metric_for_best_model", "eval_loss",
    ]
    
    print("Training command:")
    print(" ".join(training_cmd))
    print()
    
    try:
        result = subprocess.run(training_cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print("Training output:")
        print(result.stdout[-1000:])  # Show last 1000 characters
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print("Error output:")
        print(e.stderr)
        return False
    
    print()
    
    # Step 3: Check outputs
    print("Step 3: Checking outputs...")
    final_model_path = Path(output_dir) / "final_model"
    
    if final_model_path.exists():
        print(f"✓ Final model saved at: {final_model_path}")
        
        # List model files
        model_files = list(final_model_path.glob("*"))
        print("Model files:")
        for file in model_files:
            print(f"  - {file.name}")
    else:
        print("✗ Final model not found")
    
    # Check for checkpoints
    checkpoint_dirs = list(Path(output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        print(f"✓ Found {len(checkpoint_dirs)} checkpoints")
        for checkpoint in sorted(checkpoint_dirs):
            print(f"  - {checkpoint.name}")
    else:
        print("✗ No checkpoints found")
    
    # Check for logs
    log_file = Path("sft_training.log")
    if log_file.exists():
        print(f"✓ Training log available at: {log_file}")
        
        # Show last few lines of log
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print("Last few log lines:")
            for line in lines[-5:]:
                print(f"  {line.strip()}")
    else:
        print("✗ Training log not found")
    
    print()
    print("=== Workflow Complete ===")
    print(f"Model and outputs saved in: {output_dir}")
    print()
    print("Next steps:")
    print("1. Evaluate the model on test data")
    print("2. Convert to GGUF format for deployment")
    print("3. Deploy with Ollama or other inference engines")
    
    return True


def show_usage_examples():
    """Show various usage examples."""
    
    print("=== Usage Examples ===")
    print()
    
    examples = [
        {
            "name": "Basic Training",
            "description": "Simple training with default settings",
            "command": [
                "python", "-m", "lmpipeline.sft_trainer",
                "--model_name_or_path", "microsoft/DialoGPT-small",
                "--dataset_name_or_path", "tatsu-lab/alpaca",
                "--output_dir", "./outputs/basic_training",
                "--num_train_epochs", "3"
            ]
        },
        {
            "name": "Advanced LoRA Training",
            "description": "Training with custom LoRA configuration",
            "command": [
                "python", "-m", "lmpipeline.sft_trainer",
                "--model_name_or_path", "meta-llama/Llama-2-7b-hf",
                "--dataset_name_or_path", "./data/custom_data.jsonl",
                "--output_dir", "./outputs/llama_lora",
                "--use_4bit",
                "--lora_r", "32",
                "--lora_alpha", "64",
                "--lora_dropout", "0.05",
                "--num_train_epochs", "5",
                "--learning_rate", "1e-4"
            ]
        },
        {
            "name": "Configuration File",
            "description": "Using YAML configuration file",
            "command": [
                "python", "-m", "lmpipeline.sft_trainer",
                "--config", "configs/llama_7b_config.yaml"
            ]
        },
        {
            "name": "With W&B Logging",
            "description": "Training with Weights & Biases logging",
            "command": [
                "python", "-m", "lmpipeline.sft_trainer",
                "--model_name_or_path", "mistralai/Mistral-7B-v0.1",
                "--dataset_name_or_path", "OpenAssistant/oasst1",
                "--output_dir", "./outputs/mistral_wandb",
                "--use_wandb",
                "--wandb_project", "my-sft-project",
                "--num_train_epochs", "3"
            ]
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   {example['description']}")
        print(f"   Command: {' '.join(example['command'])}")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete SFT workflow example")
    parser.add_argument("--action", choices=["workflow", "examples"], default="workflow",
                       help="Action to perform")
    
    args = parser.parse_args()
    
    if args.action == "workflow":
        run_training_experiment()
    elif args.action == "examples":
        show_usage_examples()
