#!/usr/bin/env python3
# Note: upload_to_hub functionality is now part of the pipeline post-processing
# Use the pipeline with --push_to_hub flag instead

# Example: Use the pipeline with Hub upload
# python -m lmpipeline \
#     --config configs/sft_only_config.yaml \
#     --model_name_or_path ./outputs/final_model \
#     --push_to_hub \
#     --hub_repo_id your-username/my-fine-tuned-model \
#     --hub_commit_message "Upload fine-tuned model" \
#     --hub_private false

print("Use the pipeline with --push_to_hub flag for model uploads")
print("Example command:")
print(
    "python -m lmpipeline --config configs/sft_only_config.yaml --push_to_hub --hub_repo_id your-username/model"
)
