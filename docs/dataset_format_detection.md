# Automatic Dataset Format Detection and Conversion

The LM Pipeline library now includes automatic dataset format detection and conversion capabilities, making it easier to work with datasets from different sources without manual preprocessing.

## Overview

The enhanced `InstructionDataset` class can automatically detect and convert various common dataset formats used in supervised fine-tuning. This eliminates the need for manual data preprocessing and makes the training process more streamlined.

## Supported Dataset Formats

### 1. Alpaca Format (Instruction + Input + Output)

```json
{
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
}
```

When `input` is provided, it's combined with the instruction:

```json
{
    "instruction": "Translate the following text",
    "input": "Hello world",
    "output": "Hola mundo"
}
```

### 2. Prompt-Completion Format

```json
{
    "prompt": "What is the capital of France?",
    "completion": "The capital of France is Paris."
}
```

### 3. Question-Answer Format

```json
{
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris."
}
```

### 4. Conversational Format

```json
{
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
}
```

Multi-turn conversations are also supported:

```json
{
    "messages": [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is artificial intelligence."},
        {"role": "user", "content": "Tell me more"},
        {"role": "assistant", "content": "AI involves machine learning..."}
    ]
}
```

### 5. Context-Question-Answer Format

```json
{
    "context": "Paris is the capital and largest city of France.",
    "question": "What is the capital of France?",
    "answer": "Paris"
}
```

### 6. Text Format (Pre-formatted)

```json
{
    "text": "### Instruction:\nWhat is the capital of France?\n\n### Response:\nParis"
}
```

## How It Works

### Automatic Detection

The system examines the first few samples of your dataset to identify common column patterns:

1. **Priority-based Detection**: Formats are detected in order of preference
2. **Common Key Analysis**: Looks for standard field names across samples
3. **Fallback Logic**: Handles unknown formats gracefully

### Conversion Process

Once a format is detected, the system converts it to a standardized instruction-response format:

- **Instruction**: The prompt, question, or instruction text
- **Response**: The expected output, answer, or completion

## Usage

### Basic Usage

```python
from lmpipeline.sft_trainer import InstructionDataset
from transformers import AutoTokenizer

# Load your data (any supported format)
data = [
    {"prompt": "What is AI?", "completion": "AI is artificial intelligence."},
    {"question": "Explain ML", "answer": "ML is machine learning."}
]

tokenizer = AutoTokenizer.from_pretrained("your-model")

# Create dataset with auto-detection enabled (default)
dataset = InstructionDataset(
    data=data,
    tokenizer=tokenizer,
    auto_detect_format=True  # This is the default
)
```

### Disabling Auto-Detection

```python
# Use legacy behavior (manual format specification)
dataset = InstructionDataset(
    data=data,
    tokenizer=tokenizer,
    auto_detect_format=False
)
```

### Command Line Usage

```bash
# Auto-detection is enabled by default
python -m lmpipeline.sft_trainer \
    --dataset_name_or_path your_dataset.jsonl \
    --model_name_or_path your_model

# Explicitly disable auto-detection
python -m lmpipeline.sft_trainer \
    --dataset_name_or_path your_dataset.jsonl \
    --model_name_or_path your_model \
    --no_auto_detect_format
```

## Configuration Options

### DataArguments

```python
@dataclass
class DataArguments:
    auto_detect_format: bool = True  # Enable/disable auto-detection
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
```

### Command Line Arguments

- `--auto_detect_format`: Enable auto-detection (default: True)
- `--no_auto_detect_format`: Disable auto-detection
- `--instruction_template`: Custom template for formatting

## Examples

### Example 1: Mixed Format Dataset

```python
# Your dataset can have mixed formats - the system will detect the most common one
data = [
    {"instruction": "What is AI?", "output": "Artificial Intelligence"},
    {"prompt": "Explain ML", "completion": "Machine Learning"},
    {"question": "What is DL?", "answer": "Deep Learning"}
]

# The system will detect and convert all to a standard format
dataset = InstructionDataset(data, tokenizer)
```

### Example 2: Conversational Data

```python
data = [
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
    }
]

dataset = InstructionDataset(data, tokenizer)
# Converts to: instruction="Hello\nHow are you?" response="Hi there!\nI'm doing well, thank you!"
```

## Logging and Debugging

The system provides detailed logging about format detection:

```
INFO - Detected dataset format: ('instruction', 'input', 'output')
INFO - Sample conversion: {'instruction': 'What is AI?', 'input': '', 'output': 'AI is artificial intelligence.'} -> {'instruction': 'What is AI?', 'response': 'AI is artificial intelligence.'}
```

## Error Handling

### Graceful Fallbacks

- **Unknown Formats**: Attempts to infer instruction/response fields
- **Conversion Errors**: Falls back to original format processing
- **Empty Datasets**: Handles gracefully without errors

### Warning Messages

```
WARNING - Unknown dataset format detected. Available keys: ('custom_field',)
WARNING - Failed to convert item 5: Invalid format. Using original format.
```

## Best Practices

1. **Consistent Formatting**: Use consistent field names across your dataset
2. **Test First**: Run a small sample to verify detection works correctly
3. **Custom Templates**: Adjust instruction templates for your specific use case
4. **Monitor Logs**: Check logs to ensure proper format detection

## Migration from Legacy Code

### Before (Manual Format Handling)

```python
# Had to manually ensure data was in instruction-response format
data = []
for item in raw_data:
    if "prompt" in item:
        data.append({
            "instruction": item["prompt"],
            "response": item["completion"]
        })
    # ... more manual conversion logic
```

### After (Automatic Detection)

```python
# Just load your data - format detection handles the rest
dataset = InstructionDataset(raw_data, tokenizer)
```

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/test_sft_trainer.py::TestDatasetFormatter -v
python examples/test_dataset_formats.py
```
