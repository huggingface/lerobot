# K-Fold Cross Validation on Modal Labs

This implementation allows you to run k-fold cross validation for LeRobot ACT training on Modal Labs, leveraging parallel GPU compute for efficient training.

## Quick Start

### Prerequisites
- Python 3.10+
- Modal account (free tier available)
- (Optional) WandB account for experiment tracking

### Installation
```bash
# Install Modal
pip install modal

# Authenticate with Modal
modal token new
```

### Basic Usage

#### Option 1: Use the shell script (easiest)
```bash
# Make script executable
chmod +x run_kfold_modal.sh

# Run with default settings
./run_kfold_modal.sh

# Or with WandB logging
export WANDB_API_KEY=your_api_key
./run_kfold_modal.sh
```

#### Option 2: Use the Python script
```bash
# Run with default settings
python run_kfold_simple.py

# Or with WandB logging
export WANDB_API_KEY=your_api_key
python run_kfold_simple.py
```

#### Option 3: Manual steps
```bash
# 1. Generate k-fold splits
python kfold_split_generator.py \
    --dataset_repo_id=jackvial/merged_datasets_test_2 \
    --k=5

# 2. Run on Modal
modal run modal_kfold_training.py \
    --k=5 \
    --steps=100000 \
    --gpu-type=a10g

# 3. Download results
modal volume get lerobot-kfold-data outputs/ ./kfold_results/
```

## Configuration

Edit the configuration in `run_kfold_simple.py` or `run_kfold_modal.sh`:

```python
config = {
    "dataset_repo_id": "jackvial/merged_datasets_test_2",
    "k": 5,                    # Number of folds
    "steps": 100000,          # Training steps per fold
    "batch_size": 8,          # Batch size
    "learning_rate": 1e-5,    # Learning rate
    "gpu_type": "a10g",       # GPU type (t4, a10g, a100)
}
```

## GPU Options & Costs

| GPU Type | Performance | Cost/Hour | 5-Fold Total* |
|----------|------------|-----------|---------------|
| T4       | Good       | $0.59     | ~$15          |
| A10G     | Better     | $1.14     | ~$29          |
| A100     | Best       | $3.19     | ~$80          |

*Estimated for 100k steps at batch size 8

## File Structure

```
.
├── kfold_split_generator.py   # Generates k-fold splits
├── modal_kfold_training.py    # Modal app for parallel training
├── train_val.py              # Modified training script
├── run_kfold_simple.py       # Simple runner script
├── run_kfold_modal.sh        # Shell script runner
└── kfold_splits/             # Generated fold splits
    ├── fold_0.json
    ├── fold_1.json
    ├── ...
    └── kfold_summary.json
```

## Output Structure

Results are saved to Modal volume and can be downloaded:

```
kfold_results/
├── fold_0/
│   ├── checkpoints/
│   └── fold_split.json
├── fold_1/
│   ├── checkpoints/
│   └── fold_split.json
├── ...
└── kfold_summary.json    # Aggregated results
```

## Understanding Results

The `kfold_summary.json` contains:
- Mean and std metrics across all folds
- Per-fold validation results
- Success/failure status for each fold

Example output:
```json
{
  "num_successful_folds": 5,
  "metrics": {
    "mean": {
      "val_loss": 0.8234,
      "val_l1_loss": 0.4912
    },
    "std": {
      "val_loss": 0.0421,
      "val_l1_loss": 0.0234
    }
  }
}
```

## Advanced Usage

### Custom Training Parameters

Modify `modal_kfold_training.py`:
```python
training_config = {
    "policy.type": "act",
    "policy.chunk_size": 100,
    "policy.dim_model": 512,
    # Add more parameters...
}
```

### Different Dataset

```bash
python kfold_split_generator.py \
    --dataset_repo_id=your/dataset \
    --k=10
```

### Resume Failed Folds

The Modal app automatically retries failed folds. You can also manually rerun specific folds by modifying the `fold_ids` in `modal_kfold_training.py`.

## Monitoring

1. **Modal Dashboard**: Track job progress at https://modal.com
2. **WandB**: If configured, view training metrics at https://wandb.ai
3. **Logs**: Real-time logs in Modal dashboard

## Cost Optimization Tips

1. Start with T4 GPUs for testing
2. Use smaller `steps` for initial experiments
3. Run fewer folds (k=3) for quick validation
4. Monitor early stopping metrics to avoid overtraining

## Troubleshooting

### Modal Authentication Issues
```bash
modal token new
```

### Dataset Download Issues
- Ensure dataset is public or you're authenticated with HuggingFace
- Check Modal volume storage limits

### Out of Memory
- Reduce batch_size
- Use smaller GPU (counterintuitively, forces more efficient memory usage)

## Support

For issues specific to:
- LeRobot: https://github.com/huggingface/lerobot
- Modal: https://modal.com/docs
- This implementation: Create an issue with error logs 