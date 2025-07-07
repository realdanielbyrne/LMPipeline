# Training State Persistence System

The LMPipeline includes a comprehensive training state persistence system that enables recovery from interrupted training sessions. This system provides robust checkpoint management, progress tracking, and seamless resumption capabilities.

## Overview

The training state persistence system consists of three main components:

1. **Local Status Log**: JSON-based status file tracking pipeline progress
2. **Weights & Biases Integration**: Cloud-based logging of hyperparameters and metrics
3. **Recovery Mechanism**: Automatic resumption from the last successful checkpoint

## Features

### Local Status Log (`training_status.json`)

The system creates a human-readable JSON file in your pipeline's output directory that tracks:

- **Pipeline State**: Current stage, completion status, timestamps
- **Stage Progress**: Individual stage status, epochs, steps, checkpoints
- **File Paths**: Model checkpoints, dataset files, output paths
- **Configuration Hash**: Detects configuration changes between runs
- **Metrics**: Training metrics and performance data
- **Error Handling**: Failure messages and recovery points

### Weights & Biases Integration

When enabled, the system logs to W&B while excluding sensitive local paths:

- **Hyperparameters**: Learning rates, batch sizes, model architecture
- **Training Metrics**: Loss, accuracy, perplexity over time
- **Stage Transitions**: Completion status and duration
- **Model Information**: Architecture parameters (hidden_size, num_layers, etc.)

### Recovery Mechanism

The system provides intelligent recovery capabilities:

- **Automatic Detection**: Identifies incomplete stages on startup
- **File Validation**: Verifies that referenced files still exist
- **Stage Skipping**: Bypasses completed stages during resumption
- **Checkpoint Recovery**: Resumes from the last saved checkpoint
- **Configuration Validation**: Ensures consistency between runs

## Configuration

### Pipeline Configuration

Add these options to your `PipelineConfig`:

```yaml
# Training state persistence
enable_state_persistence: true    # Enable/disable state tracking
auto_resume: true                 # Automatically resume from checkpoints
force_restart: false             # Force fresh start (ignores existing state)

# Weights & Biases integration
use_wandb: true
wandb_project: "my-lmpipeline-project"
wandb_run_name: "experiment-1"
```

### Stage Configuration

Each stage automatically inherits state tracking capabilities. No additional configuration required.

## Usage Examples

### Basic Usage

```python
from lmpipeline import Pipeline, PipelineConfig

# Create configuration with state persistence enabled
config = PipelineConfig(
    model_name_or_path="microsoft/DialoGPT-medium",
    output_dir="./outputs/my_pipeline",
    stages=["sft", "dpo"],
    enable_state_persistence=True,
    auto_resume=True,
    use_wandb=True,
    wandb_project="my-project"
)

# Initialize and run pipeline
pipeline = Pipeline(config)
results = pipeline.execute()
```

### Resuming Interrupted Training

If training is interrupted, simply run the same command again:

```bash
python -m lmpipeline.pipeline_main --config configs/my_config.yaml
```

The system will automatically:
1. Load the existing state file
2. Validate file paths and configuration
3. Skip completed stages
4. Resume from the last incomplete stage

### Force Restart

To ignore existing state and start fresh:

```python
config = PipelineConfig(
    # ... other config ...
    force_restart=True  # Ignores existing state file
)
```

### Disabling State Persistence

```python
config = PipelineConfig(
    # ... other config ...
    enable_state_persistence=False  # Disables all state tracking
)
```

## State File Structure

The `training_status.json` file contains:

```json
{
  "pipeline_id": "pipeline_1704067200",
  "config_hash": "abc123def456",
  "current_stage": "dpo",
  "start_time": 1704067200.0,
  "last_update": 1704070800.0,
  "output_dir": "./outputs/my_pipeline",
  "model_name_or_path": "microsoft/DialoGPT-medium",
  "dataset_paths": ["./data/train.jsonl"],
  "pipeline_completed": false,
  "stages": {
    "sft": {
      "stage_name": "sft",
      "status": "completed",
      "start_time": 1704067200.0,
      "end_time": 1704069000.0,
      "current_epoch": 3,
      "total_epochs": 3,
      "current_step": 1500,
      "total_steps": 1500,
      "model_path": "./outputs/my_pipeline/sft",
      "tokenizer_path": "./outputs/my_pipeline/sft",
      "metrics": {
        "final_loss": 0.25,
        "final_accuracy": 0.92
      }
    },
    "dpo": {
      "stage_name": "dpo",
      "status": "in_progress",
      "start_time": 1704069000.0,
      "current_epoch": 1,
      "total_epochs": 2,
      "current_step": 500,
      "total_steps": 1000,
      "last_checkpoint_path": "./outputs/my_pipeline/dpo/checkpoint-500"
    }
  }
}
```

## Recovery Scenarios

### Scenario 1: Mid-Stage Interruption

If training stops during a stage:
- System detects incomplete stage on restart
- Attempts to resume from last checkpoint
- If no valid checkpoint, restarts the stage

### Scenario 2: Between Stages

If interruption occurs between stages:
- Completed stages are skipped
- Training resumes from next incomplete stage
- Previous stage outputs are reused

### Scenario 3: Configuration Changes

If pipeline configuration changes:
- System detects configuration hash mismatch
- Warns user and starts fresh training
- Previous state file is preserved with timestamp

### Scenario 4: Missing Files

If referenced files are missing:
- System validates all file paths on startup
- Reports missing files and starts fresh
- User can manually fix paths and retry

## Best Practices

### 1. Regular Checkpointing

Configure frequent checkpointing in your stage configs:

```yaml
stage_configs:
  sft:
    save_steps: 100        # Save checkpoint every 100 steps
    save_total_limit: 5    # Keep 5 most recent checkpoints
```

### 2. Backup State Files

The state file is critical for recovery. Consider backing it up:

```bash
cp outputs/my_pipeline/training_status.json outputs/my_pipeline/training_status_backup.json
```

### 3. Monitor W&B Logs

Use W&B dashboard to monitor training progress and detect issues early.

### 4. Validate Resumption

After resumption, check logs to ensure the correct stages were skipped:

```
INFO - Resuming from stage: dpo. Completed stages: ['sft']
INFO - Skipping completed stage: sft
INFO - Executing stage 2/2: dpo
```

## Troubleshooting

### State File Corruption

If the state file is corrupted:

```python
# Force restart to create new state file
config.force_restart = True
```

### Checkpoint Issues

If checkpoint resumption fails:

```python
# The system will automatically restart the stage
# Check logs for specific error messages
```

### W&B Connection Issues

W&B failures don't affect local state persistence:

```python
# Disable W&B temporarily
config.use_wandb = False
```

### Configuration Conflicts

If you need to modify configuration but keep progress:

1. Backup the state file
2. Make minimal changes to non-critical parameters
3. Test with a small dataset first

## API Reference

### TrainingStateManager

```python
from lmpipeline.utils import TrainingStateManager

manager = TrainingStateManager(
    output_dir="./outputs",
    pipeline_config=config_dict,
    enable_persistence=True
)

# Check resume point
resume_stage = manager.get_resume_point()

# Update progress
manager.update_stage_progress(
    stage_name="sft",
    current_epoch=2,
    current_step=1000,
    metrics={"loss": 0.5}
)

# Complete stage
manager.complete_stage(
    stage_name="sft",
    model_path="./outputs/sft",
    tokenizer_path="./outputs/sft",
    final_metrics={"final_loss": 0.25}
)
```

### WandBLogger

```python
from lmpipeline.utils import WandBLogger

logger = WandBLogger(
    project_name="my-project",
    run_name="experiment-1"
)

# Initialize run
logger.init_run(config_dict, stage_name="sft")

# Log metrics
logger.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=100)

# Log stage completion
logger.log_stage_completion("sft", duration=3600, final_metrics={"loss": 0.25})
```

## Integration with Existing Code

The state persistence system is automatically integrated into:

- **Pipeline Class**: Handles state management and recovery
- **BaseStage**: Provides progress tracking methods
- **All Algorithm Stages**: Inherit state tracking capabilities

No changes required to existing stage implementations - the system works transparently.
