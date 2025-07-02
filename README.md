# FNSFT - Fine-tuning for Neural Supervised Fine-Tuning

A production-ready Python framework for supervised fine-tuning of quantized language models using modern parameter-efficient techniques.

## 🚀 Features

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

## 📦 Installation

### Quick Install

```bash
git clone <repository-url>
cd fnsft
pip install -r requirements.txt
```

### Development Install

```bash
pip install -e .
```

## 🚀 Quick Start

### Basic Training

```bash
python -m fnsft.sft_trainer \
    --model_name_or_path microsoft/DialoGPT-small \
    --dataset_name_or_path tatsu-lab/alpaca \
    --output_dir ./outputs/my_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4
```

### Advanced Training with LoRA

```bash
python -m fnsft.sft_trainer \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_name_or_path ./examples/sample_data.jsonl \
    --output_dir ./outputs/llama_sft \
    --use_4bit \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --max_seq_length 2048 \
    --validation_split 0.1 \
    --use_wandb \
    --convert_to_gguf
```

### Using Configuration Files

```bash
python -m fnsft.sft_trainer --config configs/llama_7b_config.yaml
```

## 📁 Project Structure

```text
fnsft/
├── src/fnsft/
│   ├── __init__.py
│   └── sft_trainer.py          # Main training script
├── examples/
│   ├── example_usage.py        # Usage examples
│   └── sample_data.jsonl       # Sample dataset
├── configs/
│   └── llama_7b_config.yaml    # Example configuration
├── docs/
│   └── README.md               # Detailed documentation
├── tests/
│   └── test_sft_trainer.py     # Unit tests
├── requirements.txt            # Dependencies
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## 🎯 Supported Models

- **Llama**: meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-13b-hf
- **Mistral**: mistralai/Mistral-7B-v0.1, mistralai/Mixtral-8x7B-v0.1
- **GPT**: microsoft/DialoGPT-medium, microsoft/DialoGPT-large
- **CodeLlama**: codellama/CodeLlama-7b-Python-hf
- **And many others**: Any causal language model from HuggingFace

## 📊 Data Format

### Instruction-Response Format (Recommended)

```json
{
    "instruction": "What is the capital of France?",
    "response": "The capital of France is Paris."
}
```

### Text Format

```json
{
    "text": "### Instruction:\nWhat is the capital of France?\n\n### Response:\nThe capital of France is Paris."
}
```

## 💾 Memory Requirements

| Model Size | 4-bit + LoRA | 8-bit + LoRA | Full Fine-tuning |
|------------|--------------|--------------|------------------|
| 7B         | 6-8 GB       | 10-12 GB     | 28+ GB           |
| 13B        | 10-12 GB     | 18-20 GB     | 52+ GB           |
| 30B        | 20-24 GB     | 36-40 GB     | 120+ GB          |

## 🤗 Hugging Face Hub Integration

FNSFT includes seamless integration with Hugging Face Hub for uploading your fine-tuned models.

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
python -m fnsft.sft_trainer \
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
python -m fnsft.sft_trainer \
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

## 📖 Documentation

For detailed documentation, see [docs/README.md](docs/README.md).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- HuggingFace Transformers team for the excellent library
- Microsoft for BitsAndBytes quantization
- The open-source ML community for inspiration and tools
