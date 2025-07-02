#!/usr/bin/env python3
"""
Test script to demonstrate automatic dataset format detection and conversion.

This script tests the enhanced InstructionDataset class with various dataset formats
commonly used in supervised fine-tuning.
"""

import json
import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fnsft.sft_trainer import DatasetFormatter, InstructionDataset
from transformers import AutoTokenizer


def load_test_dataset(file_path: str):
    """Load a JSONL test dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def test_format_detection():
    """Test automatic format detection on different dataset formats."""
    print("=== Testing Automatic Dataset Format Detection ===\n")
    
    test_datasets = {
        "Alpaca Format (instruction + input + output)": [
            {"instruction": "What is AI?", "input": "", "output": "AI is artificial intelligence."},
            {"instruction": "Explain ML", "input": "in simple terms", "output": "ML is machine learning."}
        ],
        "Prompt-Completion Format": [
            {"prompt": "What is AI?", "completion": "AI is artificial intelligence."},
            {"prompt": "Explain ML", "completion": "ML is machine learning."}
        ],
        "Question-Answer Format": [
            {"question": "What is AI?", "answer": "AI is artificial intelligence."},
            {"question": "Explain ML", "answer": "ML is machine learning."}
        ],
        "Conversational Format": [
            {"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is artificial intelligence."}]},
            {"messages": [{"role": "user", "content": "Explain ML"}, {"role": "assistant", "content": "ML is machine learning."}]}
        ],
        "Text Format": [
            {"text": "### Instruction:\nWhat is AI?\n\n### Response:\nAI is artificial intelligence."},
            {"text": "### Instruction:\nExplain ML\n\n### Response:\nML is machine learning."}
        ],
        "Context-Question-Answer Format": [
            {"context": "AI is a broad field", "question": "What is AI?", "answer": "AI is artificial intelligence."},
            {"context": "ML is a subset of AI", "question": "Explain ML", "answer": "ML is machine learning."}
        ]
    }
    
    for format_name, data in test_datasets.items():
        print(f"Testing: {format_name}")
        print(f"Sample data: {data[0]}")
        
        try:
            detected_format = DatasetFormatter.detect_format(data)
            print(f"Detected format: {detected_format}")
            
            converted_sample = DatasetFormatter.convert_to_standard_format(data[0], detected_format)
            print(f"Converted sample: {converted_sample}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 80)
        print()


def test_dataset_integration():
    """Test the enhanced InstructionDataset class."""
    print("=== Testing Enhanced InstructionDataset Class ===\n")
    
    # Load a small tokenizer for testing
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        print("Skipping dataset integration test.")
        return
    
    # Test with different formats
    test_data = [
        {"instruction": "What is the capital of France?", "input": "", "output": "Paris"},
        {"instruction": "Explain photosynthesis", "input": "briefly", "output": "Plants convert sunlight to energy"}
    ]
    
    print("Creating InstructionDataset with auto-detection enabled...")
    dataset = InstructionDataset(
        data=test_data,
        tokenizer=tokenizer,
        max_length=512,
        auto_detect_format=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    
    # Decode the tokenized text to see the result
    decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print(f"Decoded text: {decoded_text}")
    
    print("-" * 80)
    print()


def test_file_formats():
    """Test loading different file formats."""
    print("=== Testing File Format Loading ===\n")
    
    test_files = [
        "examples/test_datasets/alpaca_format.jsonl",
        "examples/test_datasets/prompt_completion_format.jsonl", 
        "examples/test_datasets/qa_format.jsonl",
        "examples/test_datasets/conversational_format.jsonl"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"Testing file: {file_path}")
            try:
                data = load_test_dataset(file_path)
                detected_format = DatasetFormatter.detect_format(data)
                print(f"Detected format: {detected_format}")
                
                converted_sample = DatasetFormatter.convert_to_standard_format(data[0], detected_format)
                print(f"Original: {data[0]}")
                print(f"Converted: {converted_sample}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
        
        print("-" * 80)
        print()


def main():
    """Run all tests."""
    print("Dataset Format Detection and Conversion Test Suite")
    print("=" * 60)
    print()
    
    test_format_detection()
    test_dataset_integration()
    test_file_formats()
    
    print("All tests completed!")


if __name__ == "__main__":
    main()
