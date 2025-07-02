#!/usr/bin/env python3
from fnsft.sft_trainer import upload_to_hub
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./outputs/final_model")

# Upload to Hub
upload_to_hub(
    model_path="./outputs/final_model",
    tokenizer=tokenizer,
    repo_id="your-username/my-fine-tuned-model",
    commit_message="Upload fine-tuned model",
    private=False,
    push_adapter_only=False  # Set to True for LoRA adapters only
)