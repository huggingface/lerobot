# K-Fold Cross Validation on Modal Labs - Implementation Plan

## Overview
This plan outlines how to run k-fold cross validation for ACT training on Modal Labs, enabling parallel training of multiple folds on separate GPUs.

## Architecture

### 1. **Data Preparation Phase**
- Create k-fold splits of your dataset
- Store splits in a shared location (Modal Volume or S3)
- Each fold gets a unique identifier

### 2. **Modal Infrastructure**
- GPU-enabled containers for training
- Shared volumes for dataset and checkpoints
- Parallel job orchestration

### 3. **Training Phase**
- Launch k parallel training jobs
- Each job trains on k-1 folds, validates on 1 fold
- Results saved to shared storage

### 4. **Aggregation Phase**
- Collect metrics from all folds
- Calculate mean and std deviation
- Select best model based on cross-validation performance

## Implementation Steps

### Step 1: Create K-Fold Split Generator

```python
# kfold_split_generator.py
import json
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

def create_kfold_splits(dataset_repo_id: str, k: int = 5, output_dir: Path = Path("kfold_splits"), seed: int = 42):
    """Generate k-fold splits and save to disk."""
    meta = LeRobotDatasetMetadata(dataset_repo_id)
    total_episodes = meta.total_episodes
    
    # Create episode indices
    all_episodes = list(range(total_episodes))
    
    # Shuffle for randomness
    np.random.seed(seed)
    np.random.shuffle(all_episodes)
    
    # Create k folds
    fold_size = total_episodes // k
    folds = []
    
    for i in range(k):
        start_idx = i * fold_size
        if i == k - 1:  # Last fold gets remaining episodes
            end_idx = total_episodes
        else:
            end_idx = (i + 1) * fold_size
        
        val_episodes = sorted(all_episodes[start_idx:end_idx])
        train_episodes = sorted([ep for ep in all_episodes if ep not in val_episodes])
        
        fold_data = {
            "fold_id": i,
            "dataset_repo_id": dataset_repo_id,
            "total_episodes": total_episodes,
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "num_train": len(train_episodes),
            "num_val": len(val_episodes),
            "k": k,
            "seed": seed
        }
        folds.append(fold_data)
        
        # Save individual fold
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"fold_{i}.json", 'w') as f:
            json.dump(fold_data, f, indent=2)
    
    # Save summary
    summary = {
        "dataset_repo_id": dataset_repo_id,
        "k": k,
        "seed": seed,
        "total_episodes": total_episodes,
        "folds": [f"fold_{i}.json" for i in range(k)]
    }
    
    with open(output_dir / "kfold_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Created {k}-fold splits in {output_dir}")
    return folds
```

### Step 2: Modal App Definition

```python
# modal_kfold_training.py
import modal
import json
from pathlib import Path

# Define the Modal app
app = modal.App("lerobot-kfold-training")

# Create a volume for shared data storage
volume = modal.Volume.from_name("lerobot-kfold-data", create_if_missing=True)

# Define the Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "opencv-python",
        "wandb",
        "huggingface_hub",
    )
    .run_commands("pip install git+https://github.com/huggingface/lerobot.git")
)

@app.function(
    image=image,
    gpu="a10g",  # or "t4", "a100", depending on needs
    volumes={"/data": volume},
    timeout=3600 * 8,  # 8 hour timeout
    retries=2,
)
def train_fold(
    fold_id: int,
    fold_data: dict,
    training_config: dict,
    wandb_api_key: str = None,
):
    """Train a single fold of k-fold cross validation."""
    import os
    import subprocess
    
    # Set up wandb if provided
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    
    # Create output directory for this fold
    output_dir = Path(f"/data/outputs/fold_{fold_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save fold data for the training script
    fold_file = output_dir / "fold_split.json"
    with open(fold_file, 'w') as f:
        json.dump(fold_data, f)
    
    # Build the training command
    cmd = [
        "python", "/app/train_val.py",
        f"--dataset.repo_id={fold_data['dataset_repo_id']}",
        f"--output_dir={output_dir}",
        f"--validation.enable=false",  # We handle train/val split manually
        f"--wandb.project=lerobot_kfold",
        f"--wandb.name=fold_{fold_id}",
        "--seed=1000",
    ]
    
    # Add user-specified training configs
    for key, value in training_config.items():
        cmd.append(f"--{key}={value}")
    
    # Add episode specification
    train_episodes_str = ','.join(map(str, fold_data['train_episodes']))
    cmd.append(f"--dataset.episodes=[{train_episodes_str}]")
    
    # Run training
    print(f"Starting training for fold {fold_id}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Training failed for fold {fold_id}: {result.stderr}")
        raise RuntimeError(f"Training failed for fold {fold_id}")
    
    # Run validation on the held-out fold
    val_result = run_validation_on_fold(
        output_dir, 
        fold_data['dataset_repo_id'],
        fold_data['val_episodes']
    )
    
    return {
        "fold_id": fold_id,
        "output_dir": str(output_dir),
        "val_result": val_result,
        "success": True
    }

def run_validation_on_fold(checkpoint_dir, dataset_repo_id, val_episodes):
    """Run validation on a trained model."""
    # Implementation to load model and run validation
    # This would use the LeRobot evaluation code
    # Return validation metrics
    pass

@app.function(
    image=image,
    volumes={"/data": volume},
)
def aggregate_results(results: list[dict]):
    """Aggregate results from all folds."""
    import numpy as np
    import pandas as pd
    
    # Extract validation metrics from all folds
    metrics = []
    for result in results:
        if result["success"]:
            metrics.append(result["val_result"])
    
    # Calculate mean and std for each metric
    df = pd.DataFrame(metrics)
    summary = {
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict(),
        "all_results": results
    }
    
    # Save summary
    output_file = Path("/data/outputs/kfold_summary.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("K-fold cross validation summary:")
    print(f"Mean metrics: {summary['mean']}")
    print(f"Std metrics: {summary['std']}")
    
    return summary

@app.local_entrypoint()
def main(
    dataset_repo_id: str = "jackvial/merged_datasets_test_2",
    k: int = 5,
    steps: int = 10000,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    wandb_api_key: str = None,
):
    """Main entry point for k-fold cross validation."""
    import concurrent.futures
    
    # First, generate k-fold splits locally
    from kfold_split_generator import create_kfold_splits
    folds = create_kfold_splits(dataset_repo_id, k=k)
    
    # Upload splits to Modal volume
    volume.put_directory("kfold_splits", "/data/kfold_splits")
    
    # Define training configuration
    training_config = {
        "policy.type": "act",
        "steps": steps,
        "batch_size": batch_size,
        "optimizer.lr": learning_rate,
        "log_freq": 100,
        "save_freq": 1000,
    }
    
    # Launch parallel training for all folds
    print(f"Launching {k} parallel training jobs...")
    
    # Use Modal's parallel execution
    results = []
    for fold_id, fold_data in enumerate(folds):
        result = train_fold.spawn(
            fold_id=fold_id,
            fold_data=fold_data,
            training_config=training_config,
            wandb_api_key=wandb_api_key
        )
        results.append(result)
    
    # Wait for all folds to complete
    completed_results = []
    for result in results:
        completed_results.append(result.get())
    
    # Aggregate results
    summary = aggregate_results.remote(completed_results)
    
    print(f"K-fold cross validation complete!")
    print(f"Results saved to Modal volume")
    
    return summary
```

### Step 3: Modified train_val.py for K-Fold

```python
# train_val_kfold.py - Key modifications
@dataclass
class KFoldConfig:
    """Configuration for k-fold cross validation."""
    enable_kfold: bool = False
    fold_id: int = -1
    fold_file: str = None  # Path to fold split JSON

# In the main training function:
if cfg.kfold.enable_kfold and cfg.kfold.fold_file:
    # Load fold data
    with open(cfg.kfold.fold_file, 'r') as f:
        fold_data = json.load(f)
    
    # Override dataset episodes with fold specification
    cfg.dataset.episodes = fold_data["train_episodes"]
    
    # Set up validation dataset with fold's val episodes
    val_episodes = fold_data["val_episodes"]
    # ... rest of validation setup
```

### Step 4: Local Execution Script

```python
# run_kfold_modal.py
#!/usr/bin/env python3
import modal
import os
from pathlib import Path

def run_kfold_training():
    """Run k-fold cross validation on Modal."""
    
    # Set your configuration
    config = {
        "dataset_repo_id": "jackvial/merged_datasets_test_2",
        "k": 5,
        "steps": 100000,
        "batch_size": 8,
        "learning_rate": 1e-5,
        "wandb_api_key": os.getenv("WANDB_API_KEY"),
    }
    
    # Run on Modal
    with modal.run():
        from modal_kfold_training import main
        results = main(**config)
    
    print("Training complete!")
    print(f"Results: {results}")

if __name__ == "__main__":
    run_kfold_training()
```

## Execution Plan

1. **Prepare Environment**
   ```bash
   pip install modal
   modal token new  # Set up Modal authentication
   ```

2. **Generate K-Fold Splits**
   ```bash
   python kfold_split_generator.py --dataset_repo_id=jackvial/merged_datasets_test_2 --k=5
   ```

3. **Deploy and Run on Modal**
   ```bash
   modal run modal_kfold_training.py
   ```

4. **Monitor Progress**
   - Check Modal dashboard for job status
   - Monitor WandB for training metrics
   - View logs in real-time

5. **Retrieve Results**
   ```bash
   modal volume get lerobot-kfold-data outputs/ ./local_results/
   ```

## Cost Optimization Tips

1. **GPU Selection**
   - T4: Cheapest, good for testing ($0.59/hour)
   - A10G: Better performance ($1.14/hour)
   - A100: Best performance ($3.19/hour)

2. **Parallel Execution**
   - Running 5 folds in parallel on T4s: ~$3/hour
   - Sequential would take 5x longer

3. **Checkpointing**
   - Save checkpoints to Modal volume
   - Resume from failures without restarting

## Advantages

1. **Parallel Execution**: Train all folds simultaneously
2. **Cost Effective**: Pay only for GPU time used
3. **Scalable**: Easy to run with different k values
4. **Reproducible**: All configurations saved
5. **Fault Tolerant**: Automatic retries on failures

## Next Steps

1. Copy `train_val.py` to Modal app directory
2. Implement the validation function
3. Add metric aggregation logic
4. Set up result visualization
5. Create a dashboard for comparing fold performances 