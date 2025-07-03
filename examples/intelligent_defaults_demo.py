#!/usr/bin/env python3
"""
Demonstration of the intelligent default configuration system in LMPipeline.

This example shows how the system automatically:
1. Creates default directory structures
2. Generates intelligent model names
3. Applies sensible defaults for checkpointing
4. Handles fallback scenarios gracefully

Author: Daniel Byrne
"""

import os
import tempfile
import yaml
from pathlib import Path

# Add the src directory to the path so we can import lmpipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpipeline.utils.config_defaults import ConfigDefaults


def demo_basic_defaults():
    """Demonstrate basic default configuration application."""
    print("=== Basic Defaults Demo ===")
    
    # Minimal configuration - just the required fields
    minimal_config = {
        "model_name_or_path": "microsoft/DialoGPT-medium",
        "stages": ["sft"]
    }
    
    print("Original config:")
    print(yaml.dump(minimal_config, default_flow_style=False))
    
    # Apply intelligent defaults
    enhanced_config = ConfigDefaults.apply_all_defaults(minimal_config)
    
    print("Enhanced config with intelligent defaults:")
    print(yaml.dump(enhanced_config, default_flow_style=False))
    print()


def demo_model_naming():
    """Demonstrate intelligent model naming."""
    print("=== Model Naming Demo ===")
    
    test_cases = [
        {
            "name": "Basic SFT",
            "config": {
                "model_name_or_path": "meta-llama/Llama-2-7b-hf",
                "stages": ["sft"],
                "torch_dtype": "float16"
            }
        },
        {
            "name": "Multi-stage with quantization",
            "config": {
                "model_name_or_path": "microsoft/DialoGPT-medium",
                "stages": ["sft", "dpo"],
                "torch_dtype": "float16",
                "stage_configs": {
                    "sft": {"use_4bit": True}
                }
            }
        },
        {
            "name": "With GGUF conversion",
            "config": {
                "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.1",
                "stages": ["sft"],
                "convert_to_gguf": True,
                "gguf_quantization": "q4_0",
                "stage_configs": {
                    "sft": {"use_4bit": True}
                }
            }
        }
    ]
    
    for case in test_cases:
        print(f"Case: {case['name']}")
        print(f"Model: {case['config']['model_name_or_path']}")
        
        generated_name = ConfigDefaults.generate_final_model_name(case['config'])
        print(f"Generated name: {generated_name}")
        
        if case['config'].get('convert_to_gguf'):
            enhanced_config = ConfigDefaults.apply_model_naming_defaults(case['config'])
            gguf_path = enhanced_config.get('gguf_output_path', 'Not set')
            print(f"GGUF path: {gguf_path}")
        
        print()


def demo_directory_creation():
    """Demonstrate directory creation with fallback handling."""
    print("=== Directory Creation Demo ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Working in temporary directory: {temp_dir}")
        
        # Test with custom environment variables
        original_env = os.environ.copy()
        try:
            # Set custom paths
            custom_output = os.path.join(temp_dir, "custom_output")
            custom_checkpoints = os.path.join(temp_dir, "custom_checkpoints")
            
            os.environ[ConfigDefaults.ENV_OUTPUT_DIR] = custom_output
            os.environ[ConfigDefaults.ENV_CHECKPOINTS_DIR] = custom_checkpoints
            
            config = {
                "model_name_or_path": "test-model",
                "stages": ["sft", "dpo"],
                "stage_configs": {
                    "sft": {},
                    "dpo": {}
                }
            }
            
            print("Applying defaults with custom environment variables...")
            enhanced_config = ConfigDefaults.apply_all_defaults(config)
            
            print(f"Output directory: {enhanced_config['output_dir']}")
            print("Stage checkpoint directories:")
            for stage, stage_config in enhanced_config['stage_configs'].items():
                checkpoint_dir = stage_config.get('checkpoint_dir', 'Not set')
                print(f"  {stage}: {checkpoint_dir}")
            
            # Verify directories were created
            print("\nCreated directories:")
            for root, dirs, files in os.walk(temp_dir):
                level = root.replace(temp_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for dir_name in dirs:
                    print(f"{subindent}{dir_name}/")
            
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)
    
    print()


def demo_fallback_handling():
    """Demonstrate fallback handling when directories cannot be created."""
    print("=== Fallback Handling Demo ===")
    
    # Simulate a scenario where the preferred directory cannot be created
    # by trying to use a read-only location (this is just for demonstration)
    
    config = {
        "model_name_or_path": "test-model",
        "stages": ["sft"]
    }
    
    print("Testing fallback directory selection...")
    
    # Test the fallback mechanism
    preferred_dir = "/nonexistent/readonly/directory"
    fallback_dir = ConfigDefaults.get_fallback_directory(preferred_dir)
    
    print(f"Preferred directory: {preferred_dir}")
    print(f"Fallback directory: {fallback_dir}")
    print()


def demo_environment_variables():
    """Demonstrate environment variable configuration."""
    print("=== Environment Variables Demo ===")
    
    print("Available environment variables for configuration:")
    env_vars = [
        (ConfigDefaults.ENV_MODELS_DIR, "Base models directory"),
        (ConfigDefaults.ENV_CHECKPOINTS_DIR, "Training checkpoints directory"),
        (ConfigDefaults.ENV_OUTPUT_DIR, "Final model output directory")
    ]
    
    for var, description in env_vars:
        current_value = os.environ.get(var, "Not set")
        default_value = getattr(ConfigDefaults, f"DEFAULT_{var.split('_')[-2]}_DIR", "N/A")
        print(f"  {var}:")
        print(f"    Description: {description}")
        print(f"    Current value: {current_value}")
        print(f"    Default value: {default_value}")
        print()
    
    print("Example usage:")
    print("export LMPIPELINE_OUTPUT_DIR=/path/to/my/models")
    print("export LMPIPELINE_CHECKPOINTS_DIR=/path/to/my/checkpoints")
    print()


def main():
    """Run all demonstrations."""
    print("LMPipeline Intelligent Defaults Demonstration")
    print("=" * 50)
    print()
    
    demo_basic_defaults()
    demo_model_naming()
    demo_directory_creation()
    demo_fallback_handling()
    demo_environment_variables()
    
    print("Demonstration complete!")
    print("\nKey benefits of the intelligent defaults system:")
    print("✓ Reduces required configuration")
    print("✓ Creates sensible directory structures automatically")
    print("✓ Generates descriptive model names")
    print("✓ Provides fallback mechanisms for error handling")
    print("✓ Supports environment variable customization")
    print("✓ Logs all applied defaults for transparency")


if __name__ == "__main__":
    main()
