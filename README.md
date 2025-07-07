# LM Pipeline

**The first truly modular fine-tuning framework that transforms language models into domain experts through composable algorithm orchestration.** LM Pipeline revolutionizes how researchers and practitioners approach model fine-tuning by breaking down the artificial barriers between different training methodologies. Instead of being locked into single-algorithm frameworks, you can now seamlessly chain together Supervised Fine-Tuning, Direct Preference Optimization, Reinforcement Learning from AI Feedback, traditional RL, and Chain-of-Thought Distillation in any combination that serves your specific use case.

What makes LM Pipeline truly groundbreaking is its **plugin-based architecture** that treats each machine learning algorithm as an intelligent, composable building block. Each algorithm implements a standardized interface that enables automatic model passing, configuration management, and result tracking across the entire pipeline. This means you can experiment with novel training sequences like SFT → DPO → RLAIF → CoT Distillation, or develop entirely custom algorithms that integrate seamlessly with existing components—all without touching a single line of core pipeline code.

From rapid research prototyping on consumer hardware with 4-bit quantization to production-scale training on enterprise clusters with 70B+ models, LM Pipeline adapts to your computational constraints while maintaining state-of-the-art performance. The framework eliminates the weeks of custom engineering typically required for multi-stage training workflows, democratizing access to advanced fine-tuning techniques that were previously locked away in research labs. **Your next breakthrough is just a YAML configuration file away.**

## 🚀 Features

### 🔗 Modular Pipeline Architecture

- **🎯 Multi-Algorithm Training**: Chain SFT → DPO → RLAIF → RL → CoT Distillation
- **⚙️ Configurable Workflows**: Run any combination of algorithms in any order
- **🔄 Algorithm Orchestration**: Automatic model passing between algorithms
- **📊 Pipeline Monitoring**: Track progress and metrics across all algorithms
- **🛠️ Extensible Design**: Easy to add custom training algorithms

### 🔧 Core Training Features

- **🔧 Quantized Model Support**: 4-bit and 8-bit quantization using BitsAndBytes
- **⚡ Parameter Efficient Fine-Tuning**: LoRA/QLoRA for memory-efficient training
- **🏗️ Multiple Architectures**: Support for Llama, Mistral, GPT, and other transformer models
- **📊 Flexible Data Loading**: HuggingFace datasets or local JSON/JSONL files
- **🎯 Advanced Training**: Early stopping, checkpointing, validation, and comprehensive logging
- **📈 Experiment Tracking**: Weights & Biases integration
- **🔄 GGUF Conversion**: Convert trained models for Ollama compatibility
- **💾 Memory Optimization**: Gradient checkpointing and memory-saving techniques
- **⚙️ Configuration Management**: YAML configuration file support
- **🤗 Hub Integration**: Seamless upload to Hugging Face Hub with authentication handling
- **🔧 Cross-Platform Compatibility**: Automatic device detection and optimization for CUDA, Apple Silicon MPS, and CPU
- **🎯 Automatic Dtype Selection**: Smart precision selection based on hardware capabilities
- **⚡ Graceful Fallbacks**: Automatic quantization disabling on unsupported platforms

## 🧠 Algorithms

LM Pipeline's revolutionary **plugin-based architecture** treats machine learning algorithms as intelligent, composable building blocks that can be seamlessly chained together in configurable training workflows. Each algorithm is implemented as a "stage" that adheres to a standardized interface, enabling automatic model passing, configuration management, and result tracking across the entire pipeline.

### 🏗️ Plugin Architecture Design

The algorithm plugin system is built around three core components:

- **🔌 Standardized Interface**: All algorithms implement the `BaseStage` abstract class, ensuring consistent behavior and interoperability
- **📋 Configuration Management**: Each algorithm defines its own `StageConfig` with automatic validation and intelligent defaults
- **🔄 Result Passing**: Structured data flow between stages via `StageResult` objects containing model paths, metrics, and artifacts
- **📊 Pipeline Orchestration**: The `Pipeline` class manages execution flow, error handling, and resource management across all stages

### 🎯 Algorithm Chaining & Workflows

The plugin interface enables powerful workflow compositions:

```yaml
# Example: Complete fine-tuning pipeline
stages:
  - "sft"              # Foundation training on instruction data
  - "dpo"              # Preference alignment with human feedback
  - "rlaif"            # AI-driven preference optimization
  - "cot_distillation" # Reasoning capability enhancement
```

Each stage automatically receives the trained model from the previous stage, enabling sophisticated multi-algorithm training sequences that would typically require weeks of custom engineering.

### 🔧 Extensibility & Custom Development

The standardized plugin interface makes it trivial to develop custom algorithms:

- **🎨 Custom Algorithm Development**: Implement the `BaseStage` interface to create new training methodologies
- **🔗 Seamless Integration**: Custom algorithms integrate automatically with existing pipeline infrastructure
- **⚙️ Configuration-Driven**: No code changes needed to use custom algorithms—just update your YAML configuration
- **🧪 Independent Testing**: Each algorithm is self-contained and independently testable

### 📚 Built-in Algorithm Library

LM Pipeline includes a comprehensive suite of state-of-the-art training algorithms:

#### 🎓 **Supervised Fine-Tuning (SFT)**

Foundation training stage that adapts pre-trained models to instruction-following tasks using supervised learning on curated datasets.

#### 🎯 **Direct Preference Optimization (DPO)**

Advanced alignment technique that directly optimizes model outputs based on human preference data without requiring a separate reward model.

#### 🤖 **Reinforcement Learning from AI Feedback (RLAIF)**

Scalable preference optimization using AI-generated feedback signals, enabling alignment at scale without extensive human annotation.

#### 🎮 **Reinforcement Learning (RL)**

Traditional RL training with reward models, supporting PPO, A2C, and TRPO algorithms for fine-grained policy optimization.

#### 🧠 **Chain-of-Thought Distillation (CoT)**

Knowledge distillation technique that transfers reasoning capabilities from larger teacher models to smaller student models.

### 📖 Algorithm Documentation

Detailed documentation for each algorithm, including configuration parameters, usage examples, and best practices, is provided in separate dedicated files to maintain clarity and focus in this main README.

## 📦 Installation

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows 10/11, macOS, or Linux
- **Memory**: Minimum 8GB RAM (16GB+ recommended for larger models)
- **Storage**: 10GB+ free space for models and datasets

### Platform-Specific Installation

#### 🪟 Windows with CUDA Support

For optimal performance on Windows with NVIDIA GPUs:

```bash
git clone <repository-url>
cd LMPipeline

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install LMPipeline with quantization support
pip install -e .[quantization]

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**CUDA Requirements:**

- NVIDIA GPU with CUDA Compute Capability 5.0+
- CUDA 11.8 or 12.1+ installed
- Latest NVIDIA drivers

**Note for RTX 40/50-series (RTX 4090, RTX 5090):** These GPUs work with current PyTorch but may show compatibility warnings:

```bash
# RTX 5090 users may see warnings about sm_120 compute capability
# The device is still detected and usable despite the warning
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**RTX 5090 Compatibility:** The RTX 5090 with sm_120 compute capability shows a compatibility warning with PyTorch 2.5.1, but the device is fully functional for training and inference.

#### 🍎 Apple Silicon (M1/M2/M3/M4)

For Apple Silicon Macs with Metal Performance Shaders (MPS) acceleration:

```bash
git clone <repository-url>
cd LMPipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install LMPipeline (quantization not supported on ARM64)
pip install -e .

# Verify MPS installation
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Apple Silicon Notes:**

- BitsAndBytes quantization is automatically disabled on ARM64 (graceful fallback)
- Automatic dtype selection chooses optimal precision for MPS
- MPS acceleration provides significant speedup over CPU
- Memory is shared between CPU and GPU (unified memory architecture)
- Cross-platform utilities automatically detect and configure MPS support

#### 🐧 Linux with CUDA

For Linux systems with NVIDIA GPUs:

```bash
git clone <repository-url>
cd LMPipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install LMPipeline with all features
pip install -e .[all]

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 💻 CPU-Only Installation

For systems without GPU acceleration:

```bash
git clone <repository-url>
cd LMPipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### Alternative: Using Poetry

```bash
# If you prefer Poetry for dependency management
poetry install

# For CUDA support, install PyTorch separately first
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Activate environment
poetry shell
```

### 🔧 Cross-Platform Features

LMPipeline includes automatic cross-platform compatibility features:

- **Automatic Device Detection**: Detects CUDA, MPS, or CPU and selects optimal device
- **Smart Dtype Selection**: Automatically chooses bfloat16/float16/float32 based on hardware
- **Graceful Quantization Fallback**: Disables BitsAndBytes on unsupported platforms (ARM64)
- **Platform-Aware Error Handling**: Handles RTX 5090 compatibility warnings and import errors

```python
from lmpipeline.utils.model_utils import get_optimal_device, get_recommended_dtype

# Automatic device and dtype detection
device, device_name = get_optimal_device()
dtype = get_recommended_dtype("auto")  # or specify "float16", "bfloat16", "float32"

print(f"Using device: {device} ({device_name})")
print(f"Recommended dtype: {dtype}")
```

## 🚀 Quick Start

### Basic SFT Training

```bash
python -m lmpipeline \
    --config configs/sft_only_config.yaml \
    --model_name_or_path microsoft/DialoGPT-small \
    --output_dir ./outputs/my_model
```

### Advanced Multi-Stage Pipeline

```bash
python -m lmpipeline \
    --config configs/pipeline_config.yaml \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir ./outputs/llama_pipeline \
    --stages sft dpo \
    --convert_to_gguf \
    --push_to_hub \
    --hub_repo_id your-username/my-model
```

### 🔗 Modular Pipeline Usage Examples

#### SFT-Only Pipeline

```bash
# Use the new pipeline system for SFT training
lmpipeline-pipeline --config configs/sft_only_config.yaml
```

#### Multi-Algorithm Pipeline

```bash
# Run a complete SFT → DPO → RLAIF pipeline
lmpipeline-pipeline --config configs/pipeline_config.yaml

# Run specific algorithms only
lmpipeline-pipeline --config configs/pipeline_config.yaml --stages sft dpo

# Dry run to validate configuration
lmpipeline-pipeline --config configs/pipeline_config.yaml --dry_run
```

#### Pipeline Configuration Example

```yaml
# Multi-algorithm pipeline configuration
model_name_or_path: "microsoft/DialoGPT-medium"
output_dir: "./outputs/pipeline_run"
stages:
  - "sft"
  - "dpo"
  - "cot_distillation"

stage_configs:
  sft:
    dataset_name_or_path: "examples/sample_data.jsonl"
    num_train_epochs: 3
    learning_rate: 2e-4

  dpo:
    preference_dataset_path: "path/to/preferences.jsonl"
    beta: 0.1
    learning_rate: 5e-7

  cot_distillation:
    reasoning_dataset_path: "path/to/reasoning_data.jsonl"
    teacher_model_path: "gpt-4"
    teacher_model_type: "api"
```

### Using Configuration Files

```bash
python -m lmpipeline --config configs/sft_only_config.yaml
```

## 🎯 Supported Models

- **Llama**: meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-13b-hf
- **Mistral**: mistralai/Mistral-7B-v0.1, mistralai/Mixtral-8x7B-v0.1
- **GPT**: microsoft/DialoGPT-medium, microsoft/DialoGPT-large
- **CodeLlama**: codellama/CodeLlama-7b-Python-hf
- **And many others**: Any causal language model from HuggingFace

## 📊 Data Format

LM Pipeline supports **automatic dataset format detection and conversion**! You can use datasets in various formats without manual preprocessing.

### Supported Formats

#### 1. Alpaca Format (Instruction + Input + Output)

```json
{
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
}
```

#### 2. Prompt-Completion Format

```json
{
    "prompt": "What is the capital of France?",
    "completion": "The capital of France is Paris."
}
```

#### 3. Question-Answer Format

```json
{
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris."
}
```

#### 4. Conversational Format

```json
{
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
}
```

#### 5. Text Format (Pre-formatted)

```json
{
    "text": "### Instruction:\nWhat is the capital of France?\n\n### Response:\nThe capital of France is Paris."
}
```

#### 6. Context-Question-Answer Format

```json
{
    "context": "Paris is the capital and largest city of France.",
    "question": "What is the capital of France?",
    "answer": "Paris"
}
```

### Automatic Format Detection

The system automatically detects your dataset format and converts it to a standardized format:

```bash
# Auto-detection is enabled by default in SFT stage
python -m lmpipeline \
    --config configs/sft_config.yaml \
    --model_name_or_path your_model

# Configure dataset format detection in your config file
# See configs/sft_only_config.yaml for examples
```

### JSONL Format

Your dataset should be in JSONL format (one JSON object per line):

```jsonl
{"instruction": "What is machine learning?", "response": "Machine learning is a subset of AI..."}
{"prompt": "Explain neural networks", "completion": "Neural networks are computing systems..."}
{"question": "What is deep learning?", "answer": "Deep learning uses neural networks with multiple layers..."}
```

**Note**: You can mix different formats in the same file - the system will detect the most common format and convert all entries accordingly.

## 💾 Memory Requirements

### GPU Memory Requirements

| Model Size | 4-bit + LoRA | 8-bit + LoRA | FP16 + LoRA | Full Fine-tuning |
|------------|--------------|--------------|-------------|------------------|
| 7B         | 6-8 GB       | 10-12 GB     | 14-16 GB    | 28+ GB           |
| 13B        | 10-12 GB     | 18-20 GB     | 26-28 GB    | 52+ GB           |
| 30B        | 20-24 GB     | 36-40 GB     | 60-64 GB    | 120+ GB          |

### Platform-Specific Considerations

#### 🪟 Windows CUDA

- **RTX 3090/4090**: 24GB VRAM - Can handle 13B models with 4-bit quantization
- **RTX 3080/4080**: 10-16GB VRAM - Suitable for 7B models with quantization
- **RTX 3070/4070**: 8-12GB VRAM - Best for smaller models or CPU offloading
- **RTX 5090**: 32GB VRAM - Can handle 30B+ models with quantization

#### 🍎 Apple Silicon

- **M1/M2 (8GB)**: 7B models with FP16, aggressive memory optimization required
- **M1/M2 (16GB)**: 7B models comfortably, 13B with careful tuning
- **M1/M2 Pro/Max (32GB)**: 13B models with FP16, some 30B models possible
- **M3/M4 Max (64GB+)**: 30B models with FP16, excellent for large model training

**Note**: Apple Silicon uses unified memory, so system RAM is shared with GPU.

#### 🐧 Linux CUDA

- Same as Windows CUDA requirements
- Better memory management and optimization options
- Support for multi-GPU setups with model parallelism

### Memory Optimization Tips

#### For Limited VRAM (< 12GB)

```bash
# Use 4-bit quantization with small LoRA rank
--use_4bit --lora_r 8 --lora_alpha 16 --per_device_train_batch_size 1

# Enable gradient checkpointing and CPU offloading
--gradient_checkpointing --dataloader_pin_memory false
```

#### For Apple Silicon

```bash
# Use FP16 instead of quantization
--fp16 --per_device_train_batch_size 2 --gradient_accumulation_steps 4

# Optimize for unified memory
--dataloader_num_workers 0 --max_seq_length 1024
```

#### For High VRAM (24GB+)

```bash
# Use larger batch sizes and LoRA ranks
--per_device_train_batch_size 4 --lora_r 32 --lora_alpha 64

# Enable mixed precision for speed
--bf16 --tf32
```

## 🤗 Hugging Face Hub Integration

LM Pipeline includes seamless integration with Hugging Face Hub for uploading your fine-tuned models.

### Authentication Setup

Before uploading, authenticate with Hugging Face:

```bash
# Option 1: Environment variable
export HF_TOKEN="your_huggingface_token"

# Option 2: CLI login
pip install huggingface_hub
huggingface-cli login
```

### Upload Full Model

```bash
python -m lmpipeline.sft_trainer \
    --model_name_or_path microsoft/DialoGPT-small \
    --dataset_name_or_path tatsu-lab/alpaca \
    --output_dir ./outputs/my_model \
    --num_train_epochs 3 \
    --push_to_hub \
    --hub_repo_id "your-username/dialogpt-alpaca" \
    --hub_commit_message "Fine-tuned DialoGPT on Alpaca dataset"
```

### Upload LoRA Adapters Only

```bash
python -m lmpipeline.sft_trainer \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_name_or_path tatsu-lab/alpaca \
    --output_dir ./outputs/llama_lora \
    --use_4bit \
    --lora_r 16 \
    --push_to_hub \
    --hub_repo_id "your-username/llama-7b-alpaca-lora" \
    --push_adapter_only \
    --hub_private
```

### Hub Upload Options

- `--push_to_hub`: Enable Hub upload
- `--hub_repo_id`: Repository ID (e.g., "username/model-name")
- `--hub_commit_message`: Custom commit message
- `--hub_private`: Create private repository
- `--hub_token`: Authentication token (optional if using env var)
- `--push_adapter_only`: Upload only LoRA adapter files

### YAML Configuration

```yaml
# Hub upload settings
push_to_hub: true
hub_repo_id: "your-username/my-fine-tuned-model"
hub_commit_message: "Fine-tuned model with LoRA"
hub_private: false
push_adapter_only: false
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

### Platform-Specific Testing

#### Windows CUDA Testing

```bash
# Test CUDA functionality
poetry run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# Run tests with CUDA
poetry run pytest tests/ -v
```

#### Apple Silicon Testing

```bash
# Test MPS functionality
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Run tests (some quantization tests will be skipped on ARM64)
python -m pytest tests/ -v
```

## 🔧 Troubleshooting

### Windows CUDA Issues

**Problem**: `CUDA available: False` despite having NVIDIA GPU

- **Solution**: Reinstall PyTorch with CUDA support:

  ```bash
  pip uninstall torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

**Problem**: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

- **Cause**: GPU architecture not supported by current PyTorch build
- **Solution**: For RTX 40/50-series, use nightly builds:

  ```bash
  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
  ```

**Problem**: `ImportError: cannot import name 'bitsandbytes'`

- **Solution**: Install quantization dependencies:

  ```bash
  pip install -e .[quantization]
  ```

### Apple Silicon Issues

**Problem**: `MPS available: False` on M1/M2/M3 Mac

- **Solution**: Update to latest PyTorch version:

  ```bash
  pip install --upgrade torch torchvision torchaudio
  ```

**Problem**: `bitsandbytes` installation fails on ARM64

- **Expected**: BitsAndBytes doesn't support ARM64 architecture
- **Solution**: Use FP16 training instead of quantization:

  ```bash
  # Add to your training command
  --fp16 --no_use_4bit --no_use_8bit
  ```

**Problem**: Out of memory errors on Apple Silicon

- **Cause**: Unified memory architecture
- **Solution**: Reduce batch size and enable gradient checkpointing:

  ```bash
  --per_device_train_batch_size 1 --gradient_checkpointing
  ```

### General Issues

**Problem**: `ModuleNotFoundError: No module named 'lmpipeline'`

- **Solution**: Install in development mode:

  ```bash
  pip install -e .
  ```

**Problem**: YAML configuration file not found

- **Solution**: Use absolute paths or run from project root:

  ```bash
  cd LMPipeline
  lmpipeline-pipeline --config configs/your_config.yaml
  ```

**Problem**: Out of memory during training

- **Solutions**:
  - Reduce batch size: `--per_device_train_batch_size 1`
  - Enable gradient checkpointing: `--gradient_checkpointing`
  - Use quantization: `--use_4bit` (not available on Apple Silicon)
  - Reduce sequence length: `--max_seq_length 1024`

## 📖 Documentation

For detailed documentation, see [docs/README.md](docs/README.md).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License.

## � Extending LMPipeline with Custom Algorithms

LMPipeline features a modular architecture that makes it easy to add custom training algorithms. This section provides a comprehensive guide for developers who want to extend the pipeline with their own algorithms.

### 🏗️ Architecture Overview

The LMPipeline uses a **plugin-based architecture** where each training algorithm is implemented as a "stage" that can be chained together in configurable sequences. The system consists of:

- **Pipeline Orchestrator** (`Pipeline`): Manages execution flow and model passing between stages
- **Stage Registry** (`STAGE_REGISTRY`): Dynamic registration system for custom algorithms
- **Base Stage Interface** (`BaseStage`): Abstract base class defining the contract for all algorithms
- **Configuration System**: YAML-based configuration with automatic validation
- **Result Passing**: Structured data flow between stages via `StageResult` objects

#### Key Benefits of the Architecture

- **Modularity**: Each algorithm is self-contained and independently testable
- **Composability**: Mix and match algorithms in any order
- **Extensibility**: Add new algorithms without modifying core pipeline code
- **Configuration-Driven**: No code changes needed to use custom algorithms
- **Type Safety**: Strong typing and validation throughout

### 📋 Implementation Requirements

To create a custom algorithm, you must implement the following interface:

#### Required Base Classes

All custom algorithms must inherit from `BaseStage` and implement:

```python
from lmpipeline.algorithms.base import BaseStage, StageConfig, StageResult
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class YourCustomConfig(StageConfig):
    """Configuration for your custom algorithm."""
    # Add your algorithm-specific parameters here
    custom_param: str = field(default="default_value")
    learning_rate: float = field(default=1e-4)

class YourCustomStage(BaseStage):
    """Your custom training algorithm."""

    @property
    def stage_name(self) -> str:
        return "your_custom_algorithm"

    def validate_config(self) -> None:
        """Validate algorithm-specific configuration."""
        pass

    def execute(self, model, tokenizer, previous_result=None) -> StageResult:
        """Execute your training algorithm."""
        pass
```

#### Required Method Signatures

| Method | Purpose | Return Type | Required |
|--------|---------|-------------|----------|
| `stage_name` | Unique identifier for your algorithm | `str` | ✅ |
| `validate_config()` | Validate configuration parameters | `None` | ✅ |
| `execute()` | Main algorithm implementation | `StageResult` | ✅ |
| `prepare_model_and_tokenizer()` | Pre-processing setup | `tuple[model, tokenizer]` | ❌ |

#### Configuration Schema Requirements

Your configuration class must:

- Inherit from `StageConfig`
- Use `@dataclass` decorator
- Include `field()` metadata for parameter documentation
- Provide sensible defaults for all parameters

#### Error Handling Expectations

- **Validation Errors**: Raise `ValueError` with descriptive messages in `validate_config()`
- **Runtime Errors**: Return `StageResult` with `success=False` and `error_message`
- **Resource Cleanup**: Use try/finally blocks for cleanup operations
- **Logging**: Use `self.logger` for consistent logging throughout

### 📝 Step-by-Step Implementation Guide

#### Step 1: Create Your Algorithm File

Create a new file in `src/lmpipeline/algorithms/your_algorithm.py`:

```python
"""
Custom algorithm implementation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseStage, StageConfig, StageResult

logger = logging.getLogger(__name__)

@dataclass
class YourAlgorithmConfig(StageConfig):
    """Configuration for your custom algorithm."""

    # Algorithm-specific parameters
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for your algorithm"}
    )
    num_iterations: int = field(
        default=100,
        metadata={"help": "Number of training iterations"}
    )
    custom_dataset_path: str = field(
        default="",
        metadata={"help": "Path to your algorithm's dataset"}
    )

class YourAlgorithmStage(BaseStage):
    """Your custom training algorithm implementation."""

    def __init__(self, config: YourAlgorithmConfig):
        super().__init__(config)
        self.config: YourAlgorithmConfig = config

    @property
    def stage_name(self) -> str:
        return "your_algorithm"

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.config.num_iterations <= 0:
            raise ValueError("Number of iterations must be positive")

        if not self.config.custom_dataset_path:
            raise ValueError("Dataset path is required")

        if not Path(self.config.custom_dataset_path).exists():
            raise ValueError(f"Dataset not found: {self.config.custom_dataset_path}")

    def execute(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        previous_result: Optional[StageResult] = None,
    ) -> StageResult:
        """Execute your custom training algorithm."""
        try:
            self.logger.info(f"Starting {self.stage_name} training")
            self.setup_logging()

            # Your algorithm implementation here
            trained_model = self._run_training(model, tokenizer)

            # Save the trained model
            model_path, tokenizer_path = self.save_model_and_tokenizer(
                trained_model, tokenizer
            )

            # Collect metrics
            metrics = self._collect_metrics()

            return StageResult(
                stage_name=self.stage_name,
                success=True,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                metrics=metrics,
                artifacts={"training_log": "path/to/log.txt"}
            )

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                model_path="",
                tokenizer_path="",
                error_message=str(e)
            )
        finally:
            self.cleanup_logging()

    def _run_training(self, model, tokenizer):
        """Implement your training logic here."""
        # Your custom training implementation
        self.logger.info(f"Training with LR: {self.config.learning_rate}")

        # Example: Simple training loop
        for i in range(self.config.num_iterations):
            # Your training step
            self.logger.info(f"Training iteration {i+1}/{self.config.num_iterations}")

        return model

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect training metrics."""
        return {
            "final_loss": 0.123,
            "training_iterations": self.config.num_iterations,
            "learning_rate": self.config.learning_rate
        }
```

#### Step 2: Register Your Algorithm

Add your algorithm to `src/lmpipeline/algorithms/__init__.py`:

```python
from .your_algorithm import YourAlgorithmStage, YourAlgorithmConfig

__all__ = [
    # ... existing exports
    "YourAlgorithmStage",
    "YourAlgorithmConfig",
]
```

#### Step 3: Update Pipeline Registration

Modify `src/lmpipeline/pipeline.py` to include your algorithm:

```python
# Add import
from .algorithms.your_algorithm import YourAlgorithmStage

# Update _register_default_stages method
def _register_default_stages(self) -> None:
    """Register all default stages."""
    # ... existing registrations
    self.register_stage("your_algorithm", YourAlgorithmStage)

# Update _create_stage_config method
config_class_map = {
    # ... existing mappings
    YourAlgorithmStage: "YourAlgorithmConfig",
}
```

#### Step 4: Create Configuration File

Create a YAML configuration file to use your algorithm:

```yaml
# configs/custom_algorithm_config.yaml
model_name_or_path: "microsoft/DialoGPT-small"
output_dir: "./outputs/custom_algorithm_run"

stages:
  - "your_algorithm"

stage_configs:
  your_algorithm:
    learning_rate: 2e-4
    num_iterations: 50
    custom_dataset_path: "path/to/your/dataset.jsonl"
    use_wandb: true
    wandb_project: "custom-algorithm-experiments"
```

### 🚀 Concrete Example: Custom Tokenizer Algorithm

Here's a complete working example of a custom algorithm that implements a specialized tokenizer training stage:

<augment_code_snippet path="examples/custom_tokenizer_stage.py" mode="EXCERPT">

````python
"""
Example: Custom Tokenizer Training Stage
This algorithm fine-tunes a tokenizer on domain-specific data.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmpipeline.algorithms.base import BaseStage, StageConfig, StageResult

logger = logging.getLogger(__name__)

@dataclass
class CustomTokenizerConfig(StageConfig):
    """Configuration for custom tokenizer training."""

    # Tokenizer-specific parameters
    vocab_size: int = field(
        default=32000,
        metadata={"help": "Target vocabulary size for the tokenizer"}
    )
    training_corpus_path: str = field(
        default="",
        metadata={"help": "Path to text corpus for tokenizer training"}
    )
    special_tokens: list = field(
        default_factory=lambda: ["<pad>", "<unk>", "<s>", "</s>"],
        metadata={"help": "Special tokens to include in vocabulary"}
    )
    min_frequency: int = field(
        default=2,
        metadata={"help": "Minimum frequency for token inclusion"}
    )

class CustomTokenizerStage(BaseStage):
    """Custom tokenizer training algorithm."""

    def __init__(self, config: CustomTokenizerConfig):
        super().__init__(config)
        self.config: CustomTokenizerConfig = config

    @property
    def stage_name(self) -> str:
        return "custom_tokenizer"

    def validate_config(self) -> None:
        """Validate tokenizer configuration."""
        if self.config.vocab_size <= 0:
            raise ValueError("Vocabulary size must be positive")

        if not self.config.training_corpus_path:
            raise ValueError("Training corpus path is required")

        corpus_path = Path(self.config.training_corpus_path)
        if not corpus_path.exists():
            raise ValueError(f"Training corpus not found: {corpus_path}")

        if self.config.min_frequency < 1:
            raise ValueError("Minimum frequency must be at least 1")

    def execute(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        previous_result: Optional[StageResult] = None,
    ) -> StageResult:
        """Execute tokenizer training."""
        try:
            self.logger.info("Starting custom tokenizer training")
            self.setup_logging()

            # Train the custom tokenizer
            new_tokenizer = self._train_tokenizer()

            # Resize model embeddings to match new vocabulary
            model = self._resize_model_embeddings(model, new_tokenizer)

            # Save the updated model and tokenizer
            model_path, tokenizer_path = self.save_model_and_tokenizer(
                model, new_tokenizer
            )

            # Collect metrics
            metrics = {
                "original_vocab_size": len(tokenizer.get_vocab()),
                "new_vocab_size": len(new_tokenizer.get_vocab()),
                "vocab_size_target": self.config.vocab_size,
                "training_corpus_size": self._get_corpus_size()
            }

            return StageResult(
                stage_name=self.stage_name,
                success=True,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                metrics=metrics,
                artifacts={
                    "tokenizer_config": f"{tokenizer_path}/tokenizer_config.json",
                    "vocab_file": f"{tokenizer_path}/vocab.json"
                }
            )

        except Exception as e:
            self.logger.error(f"Tokenizer training failed: {str(e)}")
            return StageResult(
                stage_name=self.stage_name,
                success=False,
                model_path="",
                tokenizer_path="",
                error_message=str(e)
            )
        finally:
            self.cleanup_logging()

    def _train_tokenizer(self) -> AutoTokenizer:
        """Train a new tokenizer on the corpus."""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

        # Initialize a BPE tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Configure trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            special_tokens=self.config.special_tokens,
            min_frequency=self.config.min_frequency
        )

        # Train on corpus
        self.logger.info(f"Training tokenizer on {self.config.training_corpus_path}")
        tokenizer.train([self.config.training_corpus_path], trainer)

        # Convert to HuggingFace tokenizer
        return AutoTokenizer.from_pretrained(
            tokenizer,
            use_fast=True
        )

    def _resize_model_embeddings(self, model, new_tokenizer):
        """Resize model embeddings to match new vocabulary."""
        old_vocab_size = model.config.vocab_size
        new_vocab_size = len(new_tokenizer.get_vocab())

        if old_vocab_size != new_vocab_size:
            self.logger.info(
                f"Resizing embeddings: {old_vocab_size} -> {new_vocab_size}"
            )
            model.resize_token_embeddings(new_vocab_size)

        return model

    def _get_corpus_size(self) -> int:
        """Get the size of the training corpus."""
        try:
            with open(self.config.training_corpus_path, 'r') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
````

</augment_code_snippet>

### 🔗 Integration Instructions

#### Automatic Registration

The pipeline automatically discovers and registers your algorithm when you:

1. **Add the import** to `algorithms/__init__.py`
2. **Register in pipeline** using `Pipeline.register_stage()`
3. **Use in configuration** by adding to the `stages` list

#### Manual Registration (Alternative)

For external plugins, register your algorithm at runtime:

```python
from lmpipeline import Pipeline
from your_package import YourCustomStage

# Register your custom stage
Pipeline.register_stage("your_algorithm", YourCustomStage)

# Now use it in configuration
config = PipelineConfig(
    stages=["sft", "your_algorithm", "dpo"],
    # ... other config
)
```

#### Configuration Integration

Your algorithm becomes available in YAML configurations:

```yaml
stages:
  - "sft"           # Built-in algorithm
  - "your_algorithm" # Your custom algorithm
  - "dpo"           # Another built-in algorithm

stage_configs:
  your_algorithm:
    learning_rate: 1e-4
    num_iterations: 100
    custom_dataset_path: "data/custom.jsonl"
```

### 🧪 Testing Guidelines

#### Test Structure

Create comprehensive tests for your custom algorithm:

```python
# tests/test_your_algorithm.py
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.lmpipeline.algorithms.your_algorithm import (
    YourAlgorithmStage,
    YourAlgorithmConfig
)
from src.lmpipeline.algorithms.base import StageResult

class TestYourAlgorithm(unittest.TestCase):
    """Test your custom algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = YourAlgorithmConfig(
            stage_name="test_algorithm",
            output_dir=self.temp_dir,
            learning_rate=1e-4,
            num_iterations=10,
            custom_dataset_path="test_data.jsonl"
        )
        self.stage = YourAlgorithmStage(self.config)

    def test_stage_name(self):
        """Test stage name property."""
        self.assertEqual(self.stage.stage_name, "your_algorithm")

    def test_config_validation_success(self):
        """Test successful configuration validation."""
        # Create mock dataset file
        dataset_path = Path(self.temp_dir) / "test_data.jsonl"
        dataset_path.write_text('{"text": "test data"}\n')

        self.config.custom_dataset_path = str(dataset_path)

        # Should not raise any exception
        self.stage.validate_config()

    def test_config_validation_failure(self):
        """Test configuration validation failures."""
        # Test invalid learning rate
        self.config.learning_rate = -1
        with self.assertRaises(ValueError):
            self.stage.validate_config()

        # Test missing dataset
        self.config.learning_rate = 1e-4
        self.config.custom_dataset_path = "nonexistent.jsonl"
        with self.assertRaises(ValueError):
            self.stage.validate_config()

    @patch('your_module.AutoModelForCausalLM')
    @patch('your_module.AutoTokenizer')
    def test_execute_success(self, mock_tokenizer, mock_model):
        """Test successful algorithm execution."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()

        # Create mock dataset
        dataset_path = Path(self.temp_dir) / "test_data.jsonl"
        dataset_path.write_text('{"text": "test data"}\n')
        self.config.custom_dataset_path = str(dataset_path)

        # Execute the stage
        result = self.stage.execute(
            mock_model_instance,
            mock_tokenizer_instance
        )

        # Verify results
        self.assertIsInstance(result, StageResult)
        self.assertTrue(result.success)
        self.assertEqual(result.stage_name, "your_algorithm")
        self.assertIn("final_loss", result.metrics)

    def test_execute_failure(self):
        """Test algorithm execution failure handling."""
        # Use invalid configuration to trigger failure
        self.config.custom_dataset_path = "nonexistent.jsonl"

        result = self.stage.execute(Mock(), Mock())

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

if __name__ == "__main__":
    unittest.main()
```

#### Running Tests

```bash
# Run your specific algorithm tests
python -m pytest tests/test_your_algorithm.py -v

# Run all tests including your algorithm
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src/lmpipeline --cov-report=html
```

#### Integration Testing

Test your algorithm within the full pipeline:

```python
def test_algorithm_in_pipeline(self):
    """Test algorithm integration with pipeline."""
    config = PipelineConfig(
        model_name_or_path="microsoft/DialoGPT-small",
        output_dir=self.temp_dir,
        stages=["your_algorithm"],
        stage_configs={
            "your_algorithm": {
                "learning_rate": 1e-4,
                "num_iterations": 5,
                "custom_dataset_path": "test_data.jsonl"
            }
        }
    )

    pipeline = Pipeline(config)
    results = pipeline.run()

    self.assertEqual(len(results), 1)
    self.assertTrue(results[0].success)
```

### 🎯 Best Practices

#### Performance Optimization

- **Memory Management**: Use gradient checkpointing for large models
- **Batch Processing**: Implement efficient data loading with DataLoader
- **GPU Utilization**: Leverage mixed precision training when possible
- **Caching**: Cache preprocessed data to avoid redundant computation

```python
def _setup_training_optimizations(self):
    """Setup performance optimizations."""
    # Enable gradient checkpointing
    if hasattr(self.model, 'gradient_checkpointing_enable'):
        self.model.gradient_checkpointing_enable()

    # Use mixed precision if available
    if torch.cuda.is_available():
        self.scaler = torch.cuda.amp.GradScaler()
```

#### Error Handling

- **Graceful Degradation**: Continue pipeline execution when possible
- **Detailed Logging**: Provide actionable error messages
- **Resource Cleanup**: Always clean up resources in finally blocks
- **Validation**: Validate inputs early and provide clear feedback

```python
def execute(self, model, tokenizer, previous_result=None):
    """Execute with robust error handling."""
    try:
        # Validate inputs
        self._validate_inputs(model, tokenizer)

        # Main algorithm logic
        result = self._run_algorithm(model, tokenizer)

        return result

    except ValidationError as e:
        self.logger.error(f"Validation failed: {e}")
        return self._create_failure_result(str(e))

    except RuntimeError as e:
        self.logger.error(f"Runtime error: {e}")
        return self._create_failure_result(str(e))

    except Exception as e:
        self.logger.exception("Unexpected error occurred")
        return self._create_failure_result(f"Unexpected error: {e}")

    finally:
        # Always cleanup resources
        self._cleanup_resources()
```

#### Logging Best Practices

- **Structured Logging**: Use consistent log formats
- **Progress Tracking**: Log progress for long-running operations
- **Metric Logging**: Log important metrics at regular intervals
- **Debug Information**: Include debug logs for troubleshooting

```python
def _setup_logging(self):
    """Setup comprehensive logging."""
    # Log algorithm start
    self.logger.info(f"Starting {self.stage_name} with config: {self.config}")

    # Log progress during training
    for epoch in range(self.config.num_epochs):
        self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")

        # Log metrics
        if epoch % 10 == 0:
            self.logger.info(f"Current loss: {current_loss:.4f}")
```

#### Maintaining Compatibility

- **Version Compatibility**: Test with multiple Python/PyTorch versions
- **Dependency Management**: Pin critical dependencies appropriately
- **API Stability**: Maintain backward compatibility in configuration
- **Documentation**: Keep documentation updated with code changes

```python
# Example: Backward compatibility handling
def _handle_legacy_config(self, config_dict):
    """Handle legacy configuration formats."""
    # Map old parameter names to new ones
    if 'old_param_name' in config_dict:
        config_dict['new_param_name'] = config_dict.pop('old_param_name')
        self.logger.warning("Using legacy parameter name, please update config")

    return config_dict
```

### 📚 Additional Resources

- **Pipeline Guide**: See `docs/pipeline_guide.md` for detailed pipeline documentation
- **Example Algorithms**: Check `src/lmpipeline/algorithms/` for reference implementations
- **Configuration Examples**: Browse `configs/` for sample configurations
- **API Reference**: Full API documentation available in `docs/`

### 🤝 Contributing Your Algorithm

Consider contributing your algorithm back to the LMPipeline project:

1. **Fork the repository** and create a feature branch
2. **Implement your algorithm** following these guidelines
3. **Add comprehensive tests** with good coverage
4. **Update documentation** including this README
5. **Submit a pull request** with a clear description

Your contributions help make LMPipeline more powerful for the entire community!

## �🙏 Acknowledgments

- HuggingFace Transformers team for the excellent library
- Microsoft for BitsAndBytes quantization
- The open-source ML community for inspiration and tools
