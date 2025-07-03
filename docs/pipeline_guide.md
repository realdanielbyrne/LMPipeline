# Modular Fine-Tuning Pipeline Guide

The FNSFT package now supports a modular pipeline architecture that allows you to chain multiple fine-tuning techniques in a configurable sequence. This enables you to build sophisticated training workflows that combine different approaches like SFT, DPO, RLAIF, RL, and Chain-of-Thought distillation.

## Overview

The pipeline system consists of:

- **Pipeline Orchestrator**: Manages the execution of multiple stages
- **Stage Interface**: Abstract base class for all training stages
- **Configuration System**: YAML-based configuration for the entire pipeline
- **Stage Implementations**: Specific implementations for each training technique

## Supported Stages

### 1. Supervised Fine-Tuning (SFT)
- **Purpose**: Initial instruction-following training
- **Status**: ✅ Fully implemented
- **Key Features**: 
  - Automatic dataset format detection
  - LoRA/QLoRA support
  - Quantization support

### 2. Direct Preference Optimization (DPO)
- **Purpose**: Align model outputs with human preferences
- **Status**: 🚧 Stub implementation (ready for development)
- **Key Features**: 
  - Preference dataset support
  - KL regularization
  - Reference model comparison

### 3. Reinforcement Learning from AI Feedback (RLAIF)
- **Purpose**: Use AI feedback for reinforcement learning
- **Status**: 🚧 Stub implementation (ready for development)
- **Key Features**: 
  - AI feedback model integration
  - PPO training
  - Response generation and evaluation

### 4. Reinforcement Learning (RL)
- **Purpose**: Traditional RL with reward models
- **Status**: 🚧 Stub implementation (ready for development)
- **Key Features**: 
  - Multiple RL algorithms (PPO, A2C, TRPO)
  - Reward model integration
  - KL divergence control

### 5. Chain-of-Thought Distillation
- **Purpose**: Teach reasoning capabilities through distillation
- **Status**: 🚧 Stub implementation (ready for development)
- **Key Features**: 
  - Teacher model integration
  - Synthetic data generation
  - Reasoning task evaluation

## Quick Start

### 1. SFT-Only Pipeline (Backward Compatible)

```bash
# Using the new pipeline system for SFT only
fnsft-pipeline --config configs/sft_only_config.yaml
```

### 2. Multi-Stage Pipeline

```bash
# Run a complete multi-stage pipeline
fnsft-pipeline --config configs/pipeline_config.yaml
```

### 3. Custom Stage Selection

```bash
# Run only specific stages
fnsft-pipeline --config configs/pipeline_config.yaml --stages sft dpo
```

## Configuration

### Basic Configuration Structure

```yaml
# Model configuration
model_name_or_path: "microsoft/DialoGPT-medium"
torch_dtype: "float16"

# Pipeline configuration
output_dir: "./outputs/pipeline_run"
stages:
  - "sft"
  - "dpo"

# Stage-specific configurations
stage_configs:
  sft:
    dataset_name_or_path: "path/to/sft_data.jsonl"
    num_train_epochs: 3
    # ... other SFT parameters
  
  dpo:
    preference_dataset_path: "path/to/preference_data.jsonl"
    beta: 0.1
    # ... other DPO parameters
```

### Configuration Inheritance

Each stage inherits from the base `StageConfig` which provides:
- `stage_name`: Automatically set by the pipeline
- `output_dir`: Automatically set based on pipeline output dir
- `enabled`: Whether the stage should run
- `save_intermediate`: Whether to save model after this stage
- Logging and monitoring options

## Usage Examples

### Example 1: SFT → DPO Pipeline

```yaml
model_name_or_path: "microsoft/DialoGPT-medium"
output_dir: "./outputs/sft_dpo_pipeline"
stages:
  - "sft"
  - "dpo"

stage_configs:
  sft:
    dataset_name_or_path: "examples/sample_data.jsonl"
    num_train_epochs: 3
    learning_rate: 2e-4
    
  dpo:
    preference_dataset_path: "path/to/preferences.jsonl"
    beta: 0.1
    learning_rate: 5e-7
```

### Example 2: Full Reasoning Pipeline

```yaml
model_name_or_path: "microsoft/DialoGPT-medium"
output_dir: "./outputs/reasoning_pipeline"
stages:
  - "sft"
  - "cot_distillation"
  - "dpo"

stage_configs:
  sft:
    dataset_name_or_path: "instruction_data.jsonl"
    
  cot_distillation:
    reasoning_dataset_path: "reasoning_data.jsonl"
    teacher_model_path: "gpt-4"
    teacher_model_type: "api"
    
  dpo:
    preference_dataset_path: "reasoning_preferences.jsonl"
```

## Command Line Interface

### Basic Usage

```bash
fnsft-pipeline --config path/to/config.yaml
```

### Override Options

```bash
# Override model
fnsft-pipeline --config config.yaml --model_name_or_path "different/model"

# Override output directory
fnsft-pipeline --config config.yaml --output_dir "./custom_output"

# Run specific stages
fnsft-pipeline --config config.yaml --stages sft dpo

# Dry run (validate config without executing)
fnsft-pipeline --config config.yaml --dry_run
```

### Logging Options

```bash
# Set log level
fnsft-pipeline --config config.yaml --log_level DEBUG

# Save final model
fnsft-pipeline --config config.yaml --save_final_model

# Cleanup intermediate models
fnsft-pipeline --config config.yaml --cleanup_intermediate
```

## Extending the Pipeline

### Adding a New Stage

1. **Create Stage Implementation**:
```python
from fnsft.stages.base import BaseStage, StageConfig, StageResult

@dataclass
class MyStageConfig(StageConfig):
    my_parameter: str = field(metadata={"help": "My custom parameter"})

class MyStage(BaseStage):
    @property
    def stage_name(self) -> str:
        return "my_stage"
    
    def validate_config(self) -> None:
        # Validate configuration
        pass
    
    def execute(self, model, tokenizer, previous_result=None) -> StageResult:
        # Implement your training logic
        pass
```

2. **Register the Stage**:
```python
from fnsft.pipeline import Pipeline
Pipeline.register_stage("my_stage", MyStage)
```

3. **Use in Configuration**:
```yaml
stages:
  - "sft"
  - "my_stage"

stage_configs:
  my_stage:
    my_parameter: "value"
```

## Migration from Legacy SFT Trainer

The new pipeline system is backward compatible. You can:

1. **Continue using the old interface**:
```bash
fnsft-train --model_name_or_path "model" --dataset_name_or_path "data.jsonl"
```

2. **Migrate to pipeline with SFT-only config**:
```bash
fnsft-pipeline --config configs/sft_only_config.yaml
```

3. **Gradually add more stages**:
```yaml
stages:
  - "sft"  # Start with just SFT
  # - "dpo"  # Add DPO later
```

## Best Practices

1. **Start Simple**: Begin with SFT-only pipeline and add stages incrementally
2. **Validate Configs**: Use `--dry_run` to validate configurations before execution
3. **Monitor Resources**: Each stage may have different memory/compute requirements
4. **Save Intermediate Models**: Keep `save_intermediate: true` for debugging
5. **Use Appropriate Datasets**: Each stage requires specific dataset formats
6. **Tune Hyperparameters**: Each stage may need different learning rates and batch sizes

## Troubleshooting

### Common Issues

1. **Configuration Errors**: Use `--dry_run` to validate before execution
2. **Memory Issues**: Reduce batch sizes or use gradient accumulation
3. **Stage Failures**: Check logs for specific error messages
4. **Missing Dependencies**: Some stages may require additional packages

### Getting Help

- Check the logs in `pipeline_training.log`
- Use `--log_level DEBUG` for detailed output
- Validate configurations with `--dry_run`
- Review stage-specific documentation
