#!/usr/bin/env python3
"""
Example demonstrating the modular fine-tuning pipeline.

This example shows how to use the new pipeline system to run
multiple training stages in sequence.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpipeline.pipeline import Pipeline, PipelineConfig


def create_sample_config():
    """Create a sample pipeline configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = PipelineConfig(
            model_name_or_path="microsoft/DialoGPT-small",  # Small model for demo
            output_dir=str(Path(temp_dir) / "pipeline_output"),
            stages=["sft"],  # Start with just SFT for demo
            stage_configs={
                "sft": {
                    "dataset_name_or_path": "examples/sample_data.jsonl",
                    "max_seq_length": 512,  # Smaller for demo
                    "num_train_epochs": 1,  # Quick training
                    "per_device_train_batch_size": 1,
                    "learning_rate": 2e-4,
                    "validation_split": 0.1,
                    "use_4bit": False,  # Disable quantization for demo
                    "lora_r": 8,  # Smaller LoRA rank
                    "save_steps": 100,
                    "eval_steps": 100,
                    "logging_steps": 10,
                }
            },
            save_final_model=True,
            log_level="INFO"
        )
        return config


def demo_pipeline_creation():
    """Demonstrate pipeline creation and validation."""
    print("🔧 Creating pipeline configuration...")
    config = create_sample_config()
    
    print("✅ Configuration created successfully!")
    print(f"   Model: {config.model_name_or_path}")
    print(f"   Stages: {' → '.join(config.stages)}")
    print(f"   Output: {config.output_dir}")
    
    print("\n🔧 Initializing pipeline...")
    pipeline = Pipeline(config)
    
    print("✅ Pipeline initialized successfully!")
    print(f"   Available stages: {list(Pipeline.STAGE_REGISTRY.keys())}")
    print(f"   Configured stages: {len(pipeline.stages)}")
    
    # Show pipeline plan
    print("\n📋 Pipeline Execution Plan:")
    for i, stage in enumerate(pipeline.stages, 1):
        print(f"   {i}. {stage.stage_name.upper()}")
        print(f"      Dataset: {stage.config.dataset_name_or_path}")
        print(f"      Output: {stage.config.output_dir}")
    
    return pipeline


def demo_multi_stage_config():
    """Demonstrate multi-stage pipeline configuration."""
    print("\n🔗 Multi-Stage Pipeline Configuration Example:")
    
    multi_config = PipelineConfig(
        model_name_or_path="microsoft/DialoGPT-medium",
        output_dir="./outputs/multi_stage_pipeline",
        stages=["sft", "dpo", "cot_distillation"],
        stage_configs={
            "sft": {
                "dataset_name_or_path": "instruction_data.jsonl",
                "num_train_epochs": 3,
                "learning_rate": 2e-4,
            },
            "dpo": {
                "preference_dataset_path": "preference_data.jsonl",
                "beta": 0.1,
                "learning_rate": 5e-7,
            },
            "cot_distillation": {
                "reasoning_dataset_path": "reasoning_data.jsonl",
                "teacher_model_path": "gpt-4",
                "teacher_model_type": "api",
            }
        }
    )
    
    print(f"   Stages: {' → '.join(multi_config.stages)}")
    print("   Stage configurations:")
    for stage_name in multi_config.stages:
        stage_config = multi_config.stage_configs.get(stage_name, {})
        print(f"     {stage_name.upper()}:")
        for key, value in list(stage_config.items())[:3]:  # Show first 3 configs
            print(f"       {key}: {value}")
        if len(stage_config) > 3:
            print(f"       ... and {len(stage_config) - 3} more parameters")


def demo_stage_extensibility():
    """Demonstrate how to extend the pipeline with custom stages."""
    print("\n🛠️  Stage Extensibility Example:")
    
    # Show how stages are registered
    print("   Built-in stages:")
    for stage_name in Pipeline.STAGE_REGISTRY.keys():
        print(f"     - {stage_name}")
    
    print("\n   To add a custom stage:")
    print("   1. Create a class inheriting from BaseStage")
    print("   2. Implement required methods (stage_name, validate_config, execute)")
    print("   3. Register with Pipeline.register_stage('my_stage', MyStage)")
    print("   4. Use in configuration: stages: ['sft', 'my_stage']")


def main():
    """Main demonstration function."""
    print("🚀 FNSFT Modular Pipeline Demonstration")
    print("=" * 50)
    
    try:
        # Demo 1: Basic pipeline creation
        pipeline = demo_pipeline_creation()
        
        # Demo 2: Multi-stage configuration
        demo_multi_stage_config()
        
        # Demo 3: Extensibility
        demo_stage_extensibility()
        
        print("\n✅ Pipeline demonstration completed successfully!")
        print("\nNext steps:")
        print("1. Create your own configuration YAML file")
        print("2. Run: fnsft-pipeline --config your_config.yaml")
        print("3. Or use: fnsft-pipeline --config configs/sft_only_config.yaml")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
