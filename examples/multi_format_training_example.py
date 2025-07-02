#!/usr/bin/env python3
"""
Example script demonstrating training with different dataset formats.

This script shows how the automatic dataset format detection works with
various common dataset formats used in supervised fine-tuning.
"""

import json
import os
import tempfile
from pathlib import Path

# Example datasets in different formats
ALPACA_FORMAT_DATA = [
    {"instruction": "What is machine learning?", "input": "", "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."},
    {"instruction": "Explain the concept of neural networks", "input": "in simple terms", "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and learn patterns from data."},
    {"instruction": "What is the difference between AI and ML?", "input": "", "output": "AI (Artificial Intelligence) is the broader concept of machines being able to carry out tasks in a smart way, while ML (Machine Learning) is a subset of AI that focuses on the idea that machines can learn from data."}
]

PROMPT_COMPLETION_DATA = [
    {"prompt": "Define artificial intelligence", "completion": "Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans."},
    {"prompt": "What are the main types of machine learning?", "completion": "The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning."},
    {"prompt": "Explain deep learning", "completion": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."}
]

QA_FORMAT_DATA = [
    {"question": "What is supervised learning?", "answer": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data to make predictions on new, unseen data."},
    {"question": "How does unsupervised learning work?", "answer": "Unsupervised learning finds hidden patterns in data without labeled examples, using techniques like clustering and dimensionality reduction."},
    {"question": "What is reinforcement learning?", "answer": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment and receiving rewards or penalties."}
]

CONVERSATIONAL_DATA = [
    {
        "messages": [
            {"role": "user", "content": "What is natural language processing?"},
            {"role": "assistant", "content": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language in a valuable way."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Can you explain computer vision?"},
            {"role": "assistant", "content": "Computer vision is a field of AI that trains computers to interpret and understand visual information from the world, such as images and videos."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is the difference between AI and automation?"},
            {"role": "assistant", "content": "While automation follows pre-programmed rules to perform tasks, AI can adapt and learn from data to make intelligent decisions in new situations."}
        ]
    }
]


def create_test_datasets():
    """Create test dataset files in different formats."""
    os.makedirs("temp_datasets", exist_ok=True)
    
    datasets = {
        "alpaca_format.jsonl": ALPACA_FORMAT_DATA,
        "prompt_completion_format.jsonl": PROMPT_COMPLETION_DATA,
        "qa_format.jsonl": QA_FORMAT_DATA,
        "conversational_format.jsonl": CONVERSATIONAL_DATA
    }
    
    for filename, data in datasets.items():
        filepath = f"temp_datasets/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Created {filepath} with {len(data)} samples")
    
    return list(datasets.keys())


def demonstrate_format_detection():
    """Demonstrate automatic format detection without training."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    from fnsft.sft_trainer import DatasetFormatter
    
    print("=== Automatic Dataset Format Detection Demo ===\n")
    
    test_datasets = {
        "Alpaca Format": ALPACA_FORMAT_DATA,
        "Prompt-Completion Format": PROMPT_COMPLETION_DATA,
        "Question-Answer Format": QA_FORMAT_DATA,
        "Conversational Format": CONVERSATIONAL_DATA
    }
    
    for format_name, data in test_datasets.items():
        print(f"Testing: {format_name}")
        print(f"Sample: {data[0]}")
        
        # Detect format
        detected_format = DatasetFormatter.detect_format(data)
        print(f"Detected format: {detected_format}")
        
        # Convert sample
        converted = DatasetFormatter.convert_to_standard_format(data[0], detected_format)
        print(f"Converted to: {converted}")
        print("-" * 80)
        print()


def run_training_example(dataset_file, format_name):
    """Run a training example with a specific dataset format."""
    print(f"\n=== Training Example: {format_name} ===")
    
    # This would be the actual training command
    training_command = f"""
python -m fnsft.sft_trainer \\
    --model_name_or_path microsoft/DialoGPT-small \\
    --dataset_name_or_path temp_datasets/{dataset_file} \\
    --output_dir outputs/{format_name.lower().replace(' ', '_')}_model \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size 2 \\
    --learning_rate 5e-5 \\
    --max_seq_length 512 \\
    --validation_split 0.2 \\
    --auto_detect_format \\
    --logging_steps 1
"""
    
    print("Training command:")
    print(training_command)
    print("\nNote: This is a demonstration. To actually run training, execute the command above.")
    print("The auto_detect_format flag is enabled by default, so the system will automatically")
    print("detect and convert your dataset format to the standard instruction-response format.")


def cleanup_temp_files():
    """Clean up temporary dataset files."""
    import shutil
    if os.path.exists("temp_datasets"):
        shutil.rmtree("temp_datasets")
        print("Cleaned up temporary dataset files.")


def main():
    """Main demonstration function."""
    print("Multi-Format Dataset Training Example")
    print("=" * 50)
    print()
    
    try:
        # Demonstrate format detection
        demonstrate_format_detection()
        
        # Create test datasets
        print("=== Creating Test Datasets ===")
        dataset_files = create_test_datasets()
        print()
        
        # Show training examples for each format
        format_names = ["Alpaca Format", "Prompt-Completion Format", "QA Format", "Conversational Format"]
        
        for dataset_file, format_name in zip(dataset_files, format_names):
            run_training_example(dataset_file, format_name)
        
        print("\n=== Summary ===")
        print("This example demonstrates how FNSFT can automatically handle different dataset formats:")
        print("1. Alpaca format (instruction + input + output)")
        print("2. Prompt-completion format")
        print("3. Question-answer format") 
        print("4. Conversational format (messages)")
        print()
        print("Key benefits:")
        print("- No manual data preprocessing required")
        print("- Automatic format detection and conversion")
        print("- Consistent training pipeline regardless of input format")
        print("- Graceful handling of unknown formats")
        print()
        print("To train with your own dataset:")
        print("1. Ensure your data is in one of the supported formats")
        print("2. Run the training command with --auto_detect_format (default)")
        print("3. The system will automatically detect and convert your format")
        
    finally:
        # Clean up
        cleanup_temp_files()


if __name__ == "__main__":
    main()
