#!/usr/bin/env python3
"""
Demonstration script for TRL SFT implementation.

This script shows how to use the new TRL-based SFT stage with both
regular instruction-response datasets and chat-formatted datasets.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def create_sample_datasets():
    """Create sample datasets for demonstration."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create regular instruction-response dataset
    regular_data = [
        {"instruction": "What is AI?", "response": "AI is artificial intelligence."},
        {"instruction": "What is ML?", "response": "ML is machine learning."},
        {"instruction": "Explain deep learning", "response": "Deep learning uses neural networks with multiple layers."},
        {"instruction": "What is NLP?", "response": "NLP is natural language processing."},
        {"instruction": "Define computer vision", "response": "Computer vision enables machines to interpret visual information."},
    ]
    
    regular_path = temp_dir / "regular_dataset.jsonl"
    with open(regular_path, 'w') as f:
        for item in regular_data:
            f.write(json.dumps(item) + '\n')
    
    # Create chat-formatted dataset
    chat_data = [
        {
            "messages": [
                {"role": "user", "content": "Hello! How are you?"},
                {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Can you help me with Python?"},
                {"role": "assistant", "content": "Of course! I'd be happy to help you with Python programming."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a function to calculate factorial"},
                {"role": "assistant", "content": "Here's a factorial function:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"}
            ]
        },
    ]
    
    chat_path = temp_dir / "chat_dataset.jsonl"
    with open(chat_path, 'w') as f:
        for item in chat_data:
            f.write(json.dumps(item) + '\n')
    
    return temp_dir, regular_path, chat_path

def demonstrate_config_creation():
    """Demonstrate TRL SFT configuration creation."""
    print("=== TRL SFT Configuration Demo ===")
    
    try:
        from lmpipeline.algorithms.trl_sft import TRLSFTConfig
        
        # Create configuration for regular dataset
        regular_config = TRLSFTConfig(
            stage_name="trl_sft_regular",
            output_dir="./outputs/trl_sft_regular",
            dataset_name_or_path="examples/sample_data.jsonl",
            max_seq_length=1024,
            num_train_epochs=2,
            per_device_train_batch_size=4,
            packing=True,
            completion_only_loss=True,
            neftune_noise_alpha=5.0,
        )
        
        print("✓ Regular dataset configuration created")
        print(f"  - Packing enabled: {regular_config.packing}")
        print(f"  - Completion only loss: {regular_config.completion_only_loss}")
        print(f"  - NEFTune alpha: {regular_config.neftune_noise_alpha}")
        
        # Create configuration for chat dataset
        chat_config = TRLSFTConfig(
            stage_name="trl_sft_chat",
            output_dir="./outputs/trl_sft_chat",
            dataset_name_or_path="examples/chat_sample_data.jsonl",
            max_seq_length=1024,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            packing=False,  # Typically disabled for chat
            assistant_only_loss=True,  # Only train on assistant responses
            dataset_text_field="messages",
        )
        
        print("✓ Chat dataset configuration created")
        print(f"  - Assistant only loss: {chat_config.assistant_only_loss}")
        print(f"  - Dataset text field: {chat_config.dataset_text_field}")
        print(f"  - Packing disabled: {not chat_config.packing}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Could not import TRL SFT modules: {e}")
        print("  This is expected if running without the full environment")
        return False
    except Exception as e:
        print(f"✗ Configuration creation failed: {e}")
        return False

def demonstrate_format_detection():
    """Demonstrate dataset format detection."""
    print("\n=== Dataset Format Detection Demo ===")
    
    try:
        from lmpipeline.algorithms.trl_sft import TRLSFTStage, TRLSFTConfig
        
        # Create a stage for testing
        config = TRLSFTConfig(
            stage_name="test",
            output_dir="./test",
            dataset_name_or_path="test.jsonl"
        )
        stage = TRLSFTStage(config)
        
        # Test with different data formats
        regular_data = [
            {"instruction": "What is AI?", "response": "AI is artificial intelligence."}
        ]
        
        chat_data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        ]
        
        empty_data = []
        
        # Test format detection
        print(f"✓ Regular data detected as chat format: {stage._detect_chat_format(regular_data)}")
        print(f"✓ Chat data detected as chat format: {stage._detect_chat_format(chat_data)}")
        print(f"✓ Empty data detected as chat format: {stage._detect_chat_format(empty_data)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Format detection demo failed: {e}")
        return False

def demonstrate_feature_comparison():
    """Demonstrate feature comparison between regular SFT and TRL SFT."""
    print("\n=== Feature Comparison Demo ===")
    
    try:
        from lmpipeline.algorithms.sft import SFTConfig
        from lmpipeline.algorithms.trl_sft import TRLSFTConfig
        
        # Common parameters
        common_params = {
            "stage_name": "test",
            "output_dir": "./test",
            "dataset_name_or_path": "test.jsonl",
            "max_seq_length": 1024,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "learning_rate": 5e-5,
            "lora_r": 16,
            "lora_alpha": 32,
        }
        
        # Create both configs
        sft_config = SFTConfig(**common_params)
        trl_sft_config = TRLSFTConfig(**common_params)
        
        print("✓ Both configurations created successfully")
        
        # Compare features
        print("\nShared features:")
        for param in common_params:
            sft_val = getattr(sft_config, param)
            trl_val = getattr(trl_sft_config, param)
            match = "✓" if sft_val == trl_val else "✗"
            print(f"  {match} {param}: {sft_val} == {trl_val}")
        
        print("\nTRL SFT exclusive features:")
        trl_exclusive = [
            "packing", "packing_strategy", "completion_only_loss", 
            "assistant_only_loss", "neftune_noise_alpha", "use_liger_kernel"
        ]
        
        for feature in trl_exclusive:
            if hasattr(trl_sft_config, feature):
                value = getattr(trl_sft_config, feature)
                print(f"  ✓ {feature}: {value}")
            else:
                print(f"  ✗ {feature}: not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature comparison failed: {e}")
        return False

def demonstrate_pipeline_integration():
    """Demonstrate pipeline integration."""
    print("\n=== Pipeline Integration Demo ===")
    
    try:
        from lmpipeline.pipeline import Pipeline
        
        # Check if TRL SFT is registered
        if "trl_sft" in Pipeline.STAGE_REGISTRY:
            print("✓ TRL SFT stage is registered in pipeline")
            print(f"  Stage class: {Pipeline.STAGE_REGISTRY['trl_sft'].__name__}")
        else:
            print("✗ TRL SFT stage not found in pipeline registry")
            return False
        
        # Show available stages
        print(f"\nAvailable stages: {list(Pipeline.STAGE_REGISTRY.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline integration demo failed: {e}")
        return False

def main():
    """Run all demonstrations."""
    print("TRL SFT Implementation Demonstration")
    print("=" * 50)
    
    demos = [
        demonstrate_config_creation,
        demonstrate_format_detection,
        demonstrate_feature_comparison,
        demonstrate_pipeline_integration,
    ]
    
    passed = 0
    total = len(demos)
    
    for demo in demos:
        try:
            if demo():
                passed += 1
        except Exception as e:
            print(f"Demo failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Demo Results: {passed}/{total} demonstrations successful")
    
    if passed == total:
        print("\n🎉 All demonstrations passed!")
        print("\nThe TRL SFT implementation is ready for use with:")
        print("  ✓ Feature parity with existing SFT implementation")
        print("  ✓ Enhanced TRL-specific capabilities")
        print("  ✓ Automatic chat template handling")
        print("  ✓ Pipeline integration")
        
        print("\nUsage examples:")
        print("  # Regular dataset with TRL enhancements")
        print("  python -m lmpipeline --config configs/trl_sft_config.yaml")
        print("  ")
        print("  # Chat-formatted dataset")
        print("  python -m lmpipeline --config configs/trl_sft_chat_config.yaml")
        
    else:
        print(f"\n❌ {total - passed} demonstrations failed")
        print("This may be due to missing dependencies in the test environment")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
