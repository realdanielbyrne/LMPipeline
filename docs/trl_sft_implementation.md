# TRL SFT Implementation

This document describes the new TRL-based Supervised Fine-Tuning (SFT) implementation that provides enhanced capabilities for chat template handling and conversational datasets.

## Overview

The TRL SFT implementation (`TRLSFTStage`) uses TRL's `SFTTrainer` instead of the standard transformers `Trainer`, offering:

- **Enhanced chat template handling**: Automatic detection and application of chat templates for datasets with "messages" fields
- **Advanced training features**: Sequence packing, NEFTune noise injection, Liger kernel optimization
- **Feature parity**: Full compatibility with existing SFT configuration parameters
- **Flexible dataset support**: Works with both regular instruction-response and chat-formatted datasets

## Key Features

### 1. Automatic Chat Template Handling

For datasets with "messages" fields (like HuggingFaceTB/smoltalk), the TRL SFT trainer automatically:
- Detects chat-formatted data structure
- Retrieves and applies the model's chat template from the hub
- Handles multi-turn conversations with proper role formatting
- No additional configuration required

### 2. Enhanced Training Capabilities

**Sequence Packing**: Combine multiple short sequences into fixed-length blocks for improved efficiency
```yaml
packing: true
packing_strategy: "ffd"  # First-fit decreasing
```

**NEFTune Enhancement**: Add noise to embeddings for improved performance
```yaml
neftune_noise_alpha: 5.0
```

**Loss Computation Options**:
- `completion_only_loss`: Compute loss only on completion parts
- `assistant_only_loss`: Compute loss only on assistant responses (for chat data)

**Memory Optimization**:
```yaml
use_liger_kernel: true  # Reduce memory usage by 60%
```

### 3. Dataset Format Support

**Regular Instruction-Response**:
```json
{"instruction": "What is AI?", "response": "AI is artificial intelligence."}
```

**Chat-Formatted (Auto-detected)**:
```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

## Configuration

### Basic Configuration

```yaml
stages:
  - "trl_sft"

stage_configs:
  trl_sft:
    # Standard SFT parameters (all supported)
    dataset_name_or_path: "path/to/dataset.jsonl"
    max_seq_length: 2048
    num_train_epochs: 3
    per_device_train_batch_size: 4
    learning_rate: 5e-5
    
    # LoRA configuration
    lora_r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    
    # TRL-specific enhancements
    packing: true
    completion_only_loss: null  # Auto-detect
    assistant_only_loss: false
    neftune_noise_alpha: 5.0
```

### Chat Dataset Configuration

```yaml
stage_configs:
  trl_sft:
    dataset_name_or_path: "examples/chat_sample_data.jsonl"
    dataset_text_field: "messages"
    assistant_only_loss: true  # Train only on assistant responses
    packing: false  # Typically disabled for chat
```

## Usage Examples

### 1. Regular Dataset with TRL Enhancements

```bash
python -m lmpipeline --config configs/trl_sft_config.yaml
```

### 2. Chat-Formatted Dataset

```bash
python -m lmpipeline --config configs/trl_sft_chat_config.yaml
```

### 3. Programmatic Usage

```python
from lmpipeline.algorithms.trl_sft import TRLSFTStage, TRLSFTConfig

config = TRLSFTConfig(
    stage_name="trl_sft",
    output_dir="./outputs/trl_sft",
    dataset_name_or_path="path/to/dataset.jsonl",
    packing=True,
    neftune_noise_alpha=5.0,
)

stage = TRLSFTStage(config)
result = stage.execute(model, tokenizer)
```

## Configuration Parameters

### TRL-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `packing` | bool | `false` | Enable sequence packing for efficiency |
| `packing_strategy` | str | `"ffd"` | Packing strategy ("ffd" or "wrapped") |
| `completion_only_loss` | bool | `null` | Compute loss only on completion part |
| `assistant_only_loss` | bool | `false` | Compute loss only on assistant responses |
| `neftune_noise_alpha` | float | `null` | NEFTune noise alpha for performance boost |
| `use_liger_kernel` | bool | `false` | Enable Liger kernel for memory optimization |
| `dataset_text_field` | str | `"text"` | Text field name for datasets |
| `chat_template_path` | str | `null` | Custom chat template path |
| `eos_token` | str | `null` | End-of-sequence token |
| `pad_token` | str | `null` | Padding token |

### Inherited SFT Parameters

All standard SFT parameters are supported:
- Dataset configuration (`dataset_name_or_path`, `max_seq_length`, etc.)
- Training parameters (`num_train_epochs`, `learning_rate`, etc.)
- LoRA configuration (`lora_r`, `lora_alpha`, etc.)
- Quantization settings (`use_4bit`, `use_8bit`, etc.)

## Implementation Details

### Chat Template Detection

The implementation automatically detects chat-formatted datasets by examining the first few samples for:
1. Presence of "messages" field
2. List structure with role-content pairs
3. Valid role values ("user", "assistant", "system")

### Dataset Preparation

For chat datasets:
- Data is passed directly to TRL SFTTrainer
- TRL handles chat template application automatically
- Model's tokenizer chat template is used

For regular datasets:
- Existing format detection and conversion is applied
- Data is converted to standard instruction-response format
- Compatible with existing dataset utilities

### Error Handling

- Graceful fallback for TRL import errors
- Comprehensive configuration validation
- Detailed error messages for debugging

## Testing

Comprehensive unit tests are provided in `tests/test_trl_sft_stage.py`:

```bash
python -m pytest tests/test_trl_sft_stage.py -v
```

Test coverage includes:
- Configuration creation and validation
- Chat format detection
- Dataset preparation
- Trainer creation
- Error handling
- Pipeline integration

## Performance Benefits

- **Memory Efficiency**: Up to 60% memory reduction with Liger kernel
- **Training Speed**: Improved throughput with sequence packing
- **Model Quality**: Enhanced performance with NEFTune noise injection
- **Chat Handling**: Optimized for conversational datasets

## Migration from Standard SFT

To migrate from standard SFT to TRL SFT:

1. Change stage name from `"sft"` to `"trl_sft"` in configuration
2. Optionally add TRL-specific parameters for enhanced features
3. No other changes required - full backward compatibility maintained

## Troubleshooting

**TRL Import Error**: Ensure TRL is installed: `pip install trl`

**Chat Template Issues**: Verify model has chat template or specify custom template

**Memory Issues**: Enable Liger kernel or reduce batch size

**Performance**: Try enabling packing and NEFTune for better results
