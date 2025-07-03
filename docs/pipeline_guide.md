# Modular Fine-Tuning Pipeline Guide

The lm-pipeline package now supports a modular pipeline architecture that allows you to chain multiple fine-tuning algorithms in a configurable sequence. This enables you to build sophisticated training workflows that combine different approaches like SFT, DPO, RLAIF, RL, and Chain-of-Thought distillation.

## Overview

The pipeline system consists of:

- **Pipeline Orchestrator**: Manages the execution of multiple algorithms
- **Algorithm Interface**: Abstract base class for all training algorithms
- **Configuration System**: YAML-based configuration for the entire pipeline
- **Algorithm Implementations**: Specific implementations for each training technique

## Supported Algorithms

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
lmpipeline-pipeline --config configs/sft_only_config.yaml
```

### 2. Multi-Stage Pipeline

```bash
# Run a complete multi-stage pipeline
lmpipeline-pipeline --config configs/pipeline_config.yaml
```

### 3. Custom Algorithm Selection

```bash
# Run only specific algorithms
lmpipeline-pipeline --config configs/pipeline_config.yaml --stages sft dpo
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
lm-pipeline-pipeline --config path/to/config.yaml
```

### Override Options

```bash
# Override model
lm-pipeline-pipeline --config config.yaml --model_name_or_path "different/model"

# Override output directory
lm-pipeline-pipeline --config config.yaml --output_dir "./custom_output"

# Run specific stages
lm-pipeline-pipeline --config config.yaml --stages sft dpo

# Dry run (validate config without executing)
lm-pipeline-pipeline --config config.yaml --dry_run
```

### Logging Options

```bash
# Set log level
lm-pipeline-pipeline --config config.yaml --log_level DEBUG

# Save final model
lm-pipeline-pipeline --config config.yaml --save_final_model

# Cleanup intermediate models
lm-pipeline-pipeline --config config.yaml --cleanup_intermediate
```

## Adding New Algorithms

The lm-pipeline pipeline is designed to be highly extensible, allowing you to add custom training algorithms that integrate seamlessly with the existing pipeline infrastructure. This section provides a comprehensive guide for implementing and integrating new algorithms.

### Algorithm Extension Architecture

The pipeline uses a modular architecture where each algorithm inherits from the `BaseStage` class (note: we maintain "Stage" in the class name for backward compatibility). This provides:

- **Standardized Interface**: All algorithms implement the same core methods
- **Automatic Integration**: New algorithms work with existing pipeline orchestration
- **Configuration Management**: Automatic handling of algorithm-specific settings
- **Model Passing**: Seamless model transfer between algorithms
- **Monitoring & Logging**: Built-in progress tracking and metrics collection

### Step-by-Step Implementation Guide

#### 1. Create Algorithm Configuration Class

First, define a configuration class that inherits from `StageConfig`:

```python
from dataclasses import dataclass, field
from typing import Optional
from lm-pipeline.algorithms.base import StageConfig

@dataclass
class MyAlgorithmConfig(StageConfig):
    """Configuration for My Custom Algorithm."""

    # Algorithm-specific parameters
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for the algorithm"}
    )
    num_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    custom_parameter: str = field(
        default="default_value",
        metadata={"help": "Custom algorithm parameter"}
    )

    # Optional: Override base parameters if needed
    save_intermediate: bool = field(
        default=True,
        metadata={"help": "Save model after this algorithm"}
    )
```

#### 2. Implement Algorithm Class

Create the main algorithm class that inherits from `BaseStage`:

```python
import logging
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm-pipeline.algorithms.base import BaseStage, StageResult

logger = logging.getLogger(__name__)

class MyAlgorithm(BaseStage):
    """Custom training algorithm implementation."""

    def __init__(self, config: MyAlgorithmConfig):
        """Initialize the algorithm."""
        super().__init__(config)
        self.config: MyAlgorithmConfig = config

    @property
    def stage_name(self) -> str:
        """Return the name of this algorithm."""
        return "my_algorithm"

    def validate_config(self) -> None:
        """Validate algorithm configuration."""
        if self.config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.config.num_epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        # Add more validation as needed

    def execute(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        previous_result: Optional[StageResult] = None,
    ) -> StageResult:
        """Execute the algorithm."""
        try:
            self.logger.info(f"Starting {self.stage_name} training")
            self.setup_logging()

            # Prepare model and tokenizer (optional override)
            model, tokenizer = self.prepare_model_and_tokenizer(
                model, tokenizer, previous_result
            )

            # Implement your training logic here
            trained_model = self._train_model(model, tokenizer)

            # Save the model
            model_path, tokenizer_path = self.save_model_and_tokenizer(
                trained_model, tokenizer
            )

            # Collect metrics (optional)
            metrics = self._collect_metrics(trained_model)

            return StageResult(
                stage_name=self.stage_name,
                success=True,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                metrics=metrics,
            )

        except Exception as e:
            self.logger.error(f"Algorithm failed: {e}")
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                model_path="",
                tokenizer_path="",
                error_message=str(e),
            )
        finally:
            self.cleanup_logging()

    def _train_model(self, model, tokenizer):
        """Implement your specific training logic."""
        # This is where you implement the core algorithm
        # Examples:
        # - Load and preprocess datasets
        # - Set up training arguments
        # - Create trainer instance
        # - Execute training loop
        # - Return trained model

        self.logger.info("Implementing custom training logic...")
        # Your implementation here
        return model

    def _collect_metrics(self, model) -> dict:
        """Collect algorithm-specific metrics."""
        return {
            "algorithm_type": self.stage_name,
            "learning_rate": self.config.learning_rate,
            "epochs": self.config.num_epochs,
        }
```

#### 3. Register the Algorithm

Register your algorithm with the pipeline:

```python
from lm-pipeline.pipeline import Pipeline

# Register the algorithm
Pipeline.register_stage("my_algorithm", MyAlgorithm)
```

#### 4. Create Algorithm Module File

Save your algorithm in the algorithms directory:

```
src/lm-pipeline/algorithms/my_algorithm.py
```

And update the algorithms `__init__.py`:

```python
# In src/lm-pipeline/algorithms/__init__.py
from .my_algorithm import MyAlgorithm, MyAlgorithmConfig

__all__ = [
    # ... existing exports ...
    "MyAlgorithm",
    "MyAlgorithmConfig",
]
```

#### 5. Configuration Integration

Use your algorithm in pipeline configurations:

```yaml
# Pipeline configuration
model_name_or_path: "microsoft/DialoGPT-medium"
output_dir: "./outputs/custom_pipeline"
stages:
  - "sft"
  - "my_algorithm"
  - "dpo"

stage_configs:
  my_algorithm:
    learning_rate: 2e-4
    num_epochs: 5
    custom_parameter: "custom_value"
    save_intermediate: true
```

### Advanced Implementation Patterns

#### Custom Model Preparation

Override `prepare_model_and_tokenizer` for algorithm-specific setup:

```python
def prepare_model_and_tokenizer(self, model, tokenizer, previous_result=None):
    """Prepare model for this specific algorithm."""
    # Example: Add special tokens
    special_tokens = ["<custom_token>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    # Example: Freeze certain layers
    for param in model.base_model.parameters():
        param.requires_grad = False

    return model, tokenizer
```

#### Dataset Integration

Integrate with the existing dataset utilities:

```python
from lm-pipeline.utils.model_utils import load_dataset_from_path, split_dataset

def _load_and_prepare_data(self):
    """Load and prepare dataset for training."""
    data = load_dataset_from_path(
        dataset_name_or_path=self.config.dataset_path,
        dataset_config_name=self.config.dataset_config_name,
    )
    train_data, val_data = split_dataset(data, self.config.validation_split)
    return train_data, val_data
```

### Testing Your Algorithm

Create comprehensive tests for your algorithm:

```python
# tests/test_my_algorithm.py
import unittest
import tempfile
from pathlib import Path
from src.lm-pipeline.algorithms.my_algorithm import MyAlgorithm, MyAlgorithmConfig

class TestMyAlgorithm(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = MyAlgorithmConfig(
            stage_name="my_algorithm",
            output_dir=self.temp_dir,
            learning_rate=1e-4,
            num_epochs=1,
        )
        self.algorithm = MyAlgorithm(self.config)

    def test_config_validation(self):
        """Test configuration validation."""
        self.algorithm.validate_config()  # Should not raise

        # Test invalid config
        invalid_config = MyAlgorithmConfig(
            stage_name="my_algorithm",
            output_dir=self.temp_dir,
            learning_rate=-1,  # Invalid
        )
        invalid_algorithm = MyAlgorithm(invalid_config)
        with self.assertRaises(ValueError):
            invalid_algorithm.validate_config()

    def test_stage_name(self):
        """Test stage name property."""
        self.assertEqual(self.algorithm.stage_name, "my_algorithm")
```

### Best Practices

#### 1. Error Handling

- Always wrap training logic in try-catch blocks
- Return meaningful error messages in `StageResult`
- Log errors appropriately for debugging

#### 2. Resource Management

- Clean up GPU memory after training
- Use context managers for file operations
- Implement proper cleanup in `cleanup_logging()`

#### 3. Configuration Design

- Use descriptive parameter names
- Provide sensible defaults
- Include helpful metadata for documentation

#### 4. Integration Consistency

- Follow existing naming conventions
- Use shared utilities from `lm-pipeline.utils`
- Maintain compatibility with pipeline orchestration

#### 5. Documentation

- Document all configuration parameters
- Provide usage examples
- Include algorithm-specific considerations

### Example: Minimal Algorithm Implementation

Here's a complete minimal example:

```python
# src/lm-pipeline/algorithms/simple_example.py
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseStage, StageConfig, StageResult

@dataclass
class SimpleExampleConfig(StageConfig):
    message: str = field(default="Hello from custom algorithm!")

class SimpleExample(BaseStage):
    def __init__(self, config: SimpleExampleConfig):
        super().__init__(config)
        self.config: SimpleExampleConfig = config

    @property
    def stage_name(self) -> str:
        return "simple_example"

    def validate_config(self) -> None:
        if not self.config.message:
            raise ValueError("Message cannot be empty")

    def execute(self, model, tokenizer, previous_result=None) -> StageResult:
        self.logger.info(self.config.message)

        # Save model without modification
        model_path, tokenizer_path = self.save_model_and_tokenizer(model, tokenizer)

        return StageResult(
            stage_name=self.stage_name,
            success=True,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            metrics={"message": self.config.message},
        )
```

This algorithm can be used immediately in any pipeline configuration and demonstrates the minimal requirements for algorithm integration.

## Migration from Legacy SFT Trainer

The new pipeline system is backward compatible. You can:

1. **Continue using the old interface**:

```bash
lmpipeline-train --model_name_or_path "model" --dataset_name_or_path "data.jsonl"
```

2. **Migrate to pipeline with SFT-only config**:

```bash
lmpipeline-pipeline --config configs/sft_only_config.yaml
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
