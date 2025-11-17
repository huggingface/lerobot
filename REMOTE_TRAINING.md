# Training LeRobot Models on Modal GPU Infrastructure

This guide explains how to use Modal's cloud GPU infrastructure to train LeRobot policies remotely.

**Implementation follows Modal best practices** from their [official training examples](https://modal.com/docs/examples/hp_sweep_gpt), including:

- ✅ Fast dependency installation with `uv_pip_install`
- ✅ Pre-importing heavy libraries for faster cold starts
- ✅ Explicit volume commits for data persistence
- ✅ Clear resource specification and timeout management

## Prerequisites

1. **Install Modal**:

   ```bash
   pip install modal
   ```

2. **Authenticate with Modal**:
   ```bash
   modal setup
   ```

## One-Time Setup

### 1. Create Modal Secrets

Modal Secrets securely store your API keys. Create them once:

```bash
# HuggingFace Token (for downloading models/datasets)
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>

# WandB API Key (for experiment tracking)
modal secret create wandb-secret WANDB_API_KEY=<your-wandb-key>
```

To get your tokens:

- **HuggingFace**: Visit https://huggingface.co/settings/tokens
- **WandB**: Visit https://wandb.ai/authorize

### 2. (Optional) Pre-upload Datasets to Modal Volume

For faster training runs, pre-upload datasets to Modal's persistent storage:

```bash
# Create the volume
modal volume create lerobot-datasets

# Upload a local dataset
modal volume put lerobot-datasets ./local-dataset-path /datasets/my-dataset
```

## Usage

### Basic Training with Modal

```bash
python lerobot/src/lerobot/scripts/lerobot_train.py --use-modal \
  --dataset.repo_id=lerobot/pusht \
  --policy=diffusion \
  --output_dir=/outputs/pusht-run-1
```

### GPU Configuration

The training runs on **A100 GPUs** by default. To use a different GPU type, modify the `gpu` parameter in the `@app.function` decorator in `lerobot_train.py` (line 462).

### Using Pre-uploaded Datasets

If you've pre-uploaded a dataset to Modal Volume:

```bash
python lerobot/src/lerobot/scripts/lerobot_train.py --use-modal \
  --dataset.root=/datasets/my-dataset \
  --dataset.repo_id=my-org/my-dataset \
  --policy=diffusion
```

## Retrieving Training Outputs

After training completes, checkpoints are stored in the Modal Volume `lerobot-outputs`.

### List Available Checkpoints

```bash
modal volume ls lerobot-outputs
```

### Download Checkpoints to Local Machine

```bash
# Download a specific checkpoint directory
modal volume get lerobot-outputs <checkpoint-dir> ./local-outputs/

# Example:
modal volume get lerobot-outputs pusht-run-1 ./checkpoints/pusht-run-1/
```

## Key Features

✅ **Network Access**: Automatically downloads models and datasets from HuggingFace Hub  
✅ **Persistent Storage**: Datasets and checkpoints stored in Modal Volumes  
✅ **Secure Secrets**: API keys managed via Modal Secrets  
✅ **Long-running Jobs**: 24-hour timeout for extended training runs  
✅ **High-Performance GPU**: Runs on A100 GPUs by default

## Volume Management

### View Volume Contents

```bash
# List datasets
modal volume ls lerobot-datasets

# List outputs/checkpoints
modal volume ls lerobot-outputs
```

### Clean Up Old Checkpoints

```bash
# Delete specific checkpoint
modal volume rm lerobot-outputs <checkpoint-path>
```

## Troubleshooting

### "Secret not found" Error

If you see an error about missing secrets, make sure you've created them:

```bash
modal secret list  # Check existing secrets
modal secret create huggingface-secret HF_TOKEN=<your-token>
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

### Missing Dependencies

The Modal image includes all core dependencies. If you need additional packages, modify the `.uv_pip_install()` calls in the image definition (around line 472 in `lerobot_train.py`):

```python
training_image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "torch>=2.2.1",
        # ... other packages ...
        "your-package>=1.0.0",  # Add your package here
    )
)
```

### Training Timeout

Default timeout is 24 hours. For longer training runs, modify the `timeout` parameter in the `@app.function` decorator (line 503):

```python
@app.function(
    gpu="A100",
    image=training_image,
    timeout=48 * HOURS,  # 48 hours
    ...
)
```

### Different GPU Type

To use a different GPU (e.g., H100, L4, T4), edit line 493 in `lerobot_train.py`:

```python
gpu="H100",  # Change from "A100" to your preferred GPU
```

### Slow Package Installation

The implementation uses `uv_pip_install` instead of regular `pip_install`, which is **significantly faster** (following [Modal's best practices](https://modal.com/docs/examples/hp_sweep_gpt)). Package installation should only take a few minutes on first run and is cached afterwards.

## Cost Considerations

Modal charges per second of GPU usage. To minimize costs:

1. **Pre-upload datasets** to avoid repeated downloads
2. **Choose appropriate GPU** - the default is A100; consider T4/L4 for smaller models by editing the code
3. **Monitor training** - stop jobs if they're not progressing
4. **Clean up volumes** - delete old checkpoints you no longer need

## Local vs Modal Training

| Feature      | Local Training                | Modal Training                            |
| ------------ | ----------------------------- | ----------------------------------------- |
| Command      | `python lerobot_train.py ...` | `python lerobot_train.py --use-modal ...` |
| GPU Required | Yes, on your machine          | No, uses Modal's GPUs                     |
| Setup        | Install CUDA, PyTorch locally | Just `modal setup`                        |
| Outputs      | Saved locally                 | Saved to Modal Volume                     |
| Cost         | Your hardware costs           | Pay per second of usage                   |

## Implementation Details

This integration follows Modal's recommended patterns for ML training:

### Fast Dependency Installation

Uses `uv_pip_install` instead of `pip_install` for **10-100x faster** package installation. Reference: [Modal Best Practices](https://modal.com/docs/examples/hp_sweep_gpt)

### Pre-warming Heavy Imports

Heavy libraries (torch, transformers) are pre-imported during image build for faster cold starts:

```python
with training_image.imports():
    import torch
    import accelerate
```

### Explicit Volume Management

Volumes are defined as variables and committed explicitly after training:

```python
outputs_volume = modal.Volume.from_name("lerobot-outputs")
# ... after training ...
outputs_volume.commit()
```

## Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal GPU Guide](https://modal.com/docs/guide/gpu)
- [Modal Training Examples](https://modal.com/docs/examples/hp_sweep_gpt)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
