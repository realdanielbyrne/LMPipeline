# Supervised Fine-Tuning (SFT) for Quantized Language Models

A comprehensive Python script for supervised fine-tuning of quantized language models using modern techniques like LoRA/QLoRA, with support for various model architectures and datasets.

## Features

- **Quantized Model Support**: 4-bit and 8-bit quantization using BitsAndBytes
- **Parameter Efficient Fine-Tuning**: LoRA/QLoRA for memory-efficient training
- **Multiple Model Architectures**: Support for Llama, Mistral, GPT, and other transformer models
- **Flexible Data Loading**: HuggingFace datasets or local JSON/JSONL files
- **Advanced Training Features**: Early stopping, checkpointing, validation, logging
- **Experiment Tracking**: Weights & Biases integration
- **GGUF Conversion**: Convert trained models for Ollama compatibility
- **Memory Optimization**: Gradient checkpointing and other memory-saving techniques

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using Poetry

```bash
poetry install
```

### Development Installation

```bash
git clone <repository-url>
cd LMPipeline
pip install -e .
```

## Quick Start

### Basic Usage

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

## Command Line Arguments

### Model Arguments

- `--model_name_or_path`: HuggingFace model ID or local path (required)
- `--use_auth_token`: Use HuggingFace authentication token
- `--trust_remote_code`: Trust remote code when loading model
- `--torch_dtype`: Torch dtype (auto, float16, bfloat16, float32)

### Data Arguments

- `--dataset_name_or_path`: Dataset name or file path (required)
- `--dataset_config_name`: HuggingFace dataset configuration
- `--max_seq_length`: Maximum sequence length (default: 2048)
- `--instruction_template`: Template for formatting data
- `--validation_split`: Fraction for validation (default: 0.1)

### Quantization Arguments

- `--use_4bit`: Enable 4-bit quantization (default: True)
- `--use_8bit`: Enable 8-bit quantization
- `--bnb_4bit_compute_dtype`: Compute dtype for 4-bit
- `--bnb_4bit_quant_type`: Quantization type (nf4, fp4)
- `--bnb_4bit_use_double_quant`: Use double quantization

### LoRA Arguments

- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha parameter (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.1)
- `--lora_target_modules`: Target modules for LoRA
- `--lora_bias`: LoRA bias type (none, all, lora_only)

### Training Arguments

- `--output_dir`: Output directory (required)
- `--num_train_epochs`: Number of epochs (default: 3)
- `--per_device_train_batch_size`: Batch size per device (default: 4)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 1)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--weight_decay`: Weight decay (default: 0.001)
- `--warmup_ratio`: Warmup ratio (default: 0.03)
- `--lr_scheduler_type`: Learning rate scheduler (default: cosine)

### Additional Options

- `--resume_from_checkpoint`: Resume from checkpoint path
- `--use_wandb`: Enable Weights & Biases logging
- `--wandb_project`: W&B project name
- `--convert_to_gguf`: Convert to GGUF format
- `--gguf_quantization`: GGUF quantization type

## Data Format

### JSON/JSONL Format

The script supports two data formats:

#### Instruction-Response Format

```json
{
    "instruction": "What is the capital of France?",
    "response": "The capital of France is Paris."
}
```

#### Text Format

```json
{
    "text": "### Instruction:\nWhat is the capital of France?\n\n### Response:\nThe capital of France is Paris."
}
```

### HuggingFace Datasets

You can use any HuggingFace dataset that contains instruction-response pairs or text data:

```bash
--dataset_name_or_path tatsu-lab/alpaca
--dataset_name_or_path OpenAssistant/oasst1
--dataset_name_or_path databricks/databricks-dolly-15k
```

## Supported Models

The script supports various transformer architectures:

- **Llama**: meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-13b-hf
- **Mistral**: mistralai/Mistral-7B-v0.1, mistralai/Mixtral-8x7B-v0.1
- **GPT**: microsoft/DialoGPT-medium, microsoft/DialoGPT-large
- **CodeLlama**: codellama/CodeLlama-7b-Python-hf
- **And many others**: Any causal language model from HuggingFace

## Memory Requirements

Approximate GPU memory requirements:

| Model Size | 4-bit + LoRA | 8-bit + LoRA | Full Fine-tuning |
|------------|--------------|--------------|------------------|
| 7B         | 6-8 GB       | 10-12 GB     | 28+ GB           |
| 13B        | 10-12 GB     | 18-20 GB     | 52+ GB           |
| 30B        | 20-24 GB     | 36-40 GB     | 120+ GB          |

## Output Structure

After training, the output directory contains:

```
outputs/
├── checkpoint-500/          # Training checkpoints
├── checkpoint-1000/
├── final_model/            # Final trained model
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── model.gguf              # GGUF format (if converted)
├── runs/                   # TensorBoard logs
└── sft_training.log        # Training logs
```

## Examples

See the `examples/` directory for complete usage examples:

- `example_usage.py`: Various training scenarios
- `sample_data.jsonl`: Sample instruction dataset

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size, increase gradient accumulation, or use smaller model
2. **Import Errors**: Ensure all dependencies are installed correctly
3. **Model Loading Issues**: Check model name and authentication token
4. **Data Format Errors**: Verify JSON/JSONL format and required fields

### Performance Tips

1. Use gradient checkpointing for memory efficiency
2. Enable mixed precision training (fp16/bf16)
3. Use appropriate batch size and gradient accumulation
4. Monitor GPU utilization and adjust accordingly

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is licensed under the MIT License.
