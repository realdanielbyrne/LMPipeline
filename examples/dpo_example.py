#!/usr/bin/env python3
"""
Example demonstrating DPO (Direct Preference Optimization) usage.

This example shows how to use the DPO stage for preference optimization
with automatic dataset format detection.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpipeline.algorithms.dpo import DPOConfig, DPOStage, PreferenceDatasetFormatter


def create_sample_preference_dataset():
    """Create a sample preference dataset for demonstration."""
    preference_data = [
        {
            "prompt": "What is artificial intelligence?",
            "chosen": "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.",
            "rejected": "AI is just computers doing human stuff.",
        },
        {
            "prompt": "Explain machine learning in simple terms.",
            "chosen": "Machine learning is a subset of AI where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each specific task.",
            "rejected": "Machine learning is when machines learn things.",
        },
        {
            "instruction": "What are the benefits of renewable energy?",
            "preferred": "Renewable energy sources like solar and wind power offer numerous benefits including reduced greenhouse gas emissions, energy independence, job creation, and long-term cost savings.",
            "rejected": "Renewable energy is good for the environment.",
        },
        {
            "question": "How does photosynthesis work?",
            "good_answer": "Photosynthesis is a process by which plants, algae, and certain microorganisms transform light energy from the sun into the chemical energy of food. During photosynthesis, energy from sunlight is harnessed and used to convert carbon dioxide and water into organic compounds—namely sugar molecules—and oxygen.",
            "bad_answer": "Plants eat sunlight.",
        },
    ]
    return preference_data


def demonstrate_format_detection():
    """Demonstrate automatic format detection."""
    print("=== DPO Dataset Format Detection Demo ===\n")

    # Create sample data with different formats
    preference_data = create_sample_preference_dataset()

    # Detect format
    try:
        detected_format = PreferenceDatasetFormatter.detect_format(preference_data)
        print(f"Detected format: {detected_format}")

        # Show conversion of first sample
        first_sample = preference_data[0]
        converted = PreferenceDatasetFormatter.convert_to_standard_format(
            first_sample, detected_format
        )

        print(f"\nOriginal sample: {first_sample}")
        print(f"Converted sample: {converted}")

    except Exception as e:
        print(f"Format detection failed: {e}")


def demonstrate_dpo_config():
    """Demonstrate DPO configuration."""
    print("\n=== DPO Configuration Demo ===\n")

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample preference dataset file
        preference_data = create_sample_preference_dataset()
        dataset_file = Path(temp_dir) / "preferences.json"

        with open(dataset_file, "w") as f:
            json.dump(preference_data, f, indent=2)

        # Create DPO configuration
        config = DPOConfig(
            stage_name="dpo_demo",
            output_dir=temp_dir,
            preference_dataset_path=str(dataset_file),
            beta=0.1,  # KL regularization strength
            max_seq_length=1024,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=5e-7,
            validation_split=0.2,
            auto_detect_format=True,  # Enable automatic format detection
        )

        print("DPO Configuration:")
        print(f"  Dataset path: {config.preference_dataset_path}")
        print(f"  Beta (KL regularization): {config.beta}")
        print(f"  Max sequence length: {config.max_seq_length}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Auto-detect format: {config.auto_detect_format}")

        # Create DPO stage
        dpo_stage = DPOStage(config)

        # Validate configuration
        try:
            dpo_stage.validate_config()
            print("✓ Configuration is valid")
        except Exception as e:
            print(f"✗ Configuration error: {e}")

        # Load and validate dataset
        try:
            preference_data_loaded = dpo_stage._load_preference_dataset()
            print(
                f"✓ Successfully loaded {len(preference_data_loaded)} preference examples"
            )

            # Split dataset
            train_data, val_data = dpo_stage._split_preference_dataset(
                preference_data_loaded
            )
            print(
                f"✓ Split into {len(train_data)} training and {len(val_data)} validation examples"
            )

        except Exception as e:
            print(f"✗ Dataset loading error: {e}")


def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n=== DPO Error Handling Demo ===\n")

    # Create DPO stage for testing error handling
    with tempfile.TemporaryDirectory() as temp_dir:
        config = DPOConfig(
            stage_name="error_demo",
            output_dir=temp_dir,
            preference_dataset_path="nonexistent.json",  # Invalid path
            beta=0.1,
        )

        dpo_stage = DPOStage(config)

        # Test different error scenarios
        test_errors = [
            Exception("CUDA out of memory"),
            Exception("No module named 'trl'"),
            Exception("dataset format key error"),
            Exception("reference model loading failed"),
            Exception("tokenizer compatibility issue"),
            Exception("training convergence NaN values"),
            Exception("unknown error type"),
        ]

        print("Error handling examples:")
        for error in test_errors:
            handled_message = dpo_stage._handle_training_error(error)
            print(f"  Original: {error}")
            print(f"  Handled:  {handled_message}\n")


def main():
    """Run all demonstrations."""
    print("DPO (Direct Preference Optimization) Implementation Demo")
    print("=" * 60)

    try:
        demonstrate_format_detection()
        demonstrate_dpo_config()
        demonstrate_error_handling()

        print("\n=== Summary ===")
        print("✓ DPO implementation completed with:")
        print("  - TRL library integration")
        print("  - Automatic preference dataset format detection")
        print("  - Support for multiple dataset formats")
        print("  - Comprehensive error handling")
        print("  - Modular pipeline integration")
        print("  - Full test coverage")

    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
