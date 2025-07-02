#!/usr/bin/env python3
"""Modal app for k-fold cross validation training of LeRobot models."""

import modal
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# Define the Modal app
app = modal.App("lerobot-kfold-training")

# Create volumes for shared data storage
volume = modal.Volume.from_name("lerobot-kfold-data", create_if_missing=True)
dataset_volume = modal.Volume.from_name("lerobot-datasets", create_if_missing=True)

# Define the Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", 
        "ffmpeg", 
        "libgl1-mesa-glx", 
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1"
    )
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "opencv-python",
        "wandb",
        "huggingface_hub",
        "matplotlib",
        "pandas",
    )
    .run_commands("pip install git+https://github.com/huggingface/lerobot.git")
    .copy_local_file("train_val.py", "/app/train_val.py")
)

@app.function(
    image=image,
    gpu="a10g",  # Options: "t4" ($0.59/hr), "a10g" ($1.14/hr), "a100" ($3.19/hr)
    volumes={
        "/data": volume,
        "/datasets": dataset_volume
    },
    timeout=3600 * 8,  # 8 hour timeout
    retries=1,
)
def train_fold(
    fold_id: int,
    fold_data: dict,
    training_config: dict,
    wandb_api_key: Optional[str] = None,
) -> dict:
    """Train a single fold of k-fold cross validation."""
    import subprocess
    import torch
    from huggingface_hub import snapshot_download
    
    # Set up wandb if provided
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    
    # Download dataset to local cache if not already present
    dataset_path = Path(f"/datasets/{fold_data['dataset_repo_id'].replace('/', '_')}")
    if not dataset_path.exists():
        print(f"Downloading dataset {fold_data['dataset_repo_id']}...")
        snapshot_download(
            repo_id=fold_data['dataset_repo_id'],
            repo_type="dataset",
            local_dir=dataset_path
        )
    
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
        f"--dataset.root=/datasets",
        f"--output_dir={output_dir}",
        f"--validation.enable=false",  # We handle train/val split manually
        f"--seed=1000",
    ]
    
    # Add wandb configuration if API key provided
    if wandb_api_key:
        cmd.extend([
            f"--wandb.enable=true",
            f"--wandb.project=lerobot_kfold",
            f"--wandb.name=fold_{fold_id}",
        ])
    
    # Add user-specified training configs
    for key, value in training_config.items():
        cmd.append(f"--{key}={value}")
    
    # Add episode specification
    train_episodes_str = ','.join(map(str, fold_data['train_episodes']))
    cmd.append(f"--dataset.episodes=[{train_episodes_str}]")
    
    # Run training
    print(f"Starting training for fold {fold_id}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Training failed for fold {fold_id}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Training failed for fold {fold_id}")
    
    print(f"Training completed for fold {fold_id}")
    
    # Run validation on the held-out fold
    val_result = run_validation_on_fold(
        output_dir, 
        fold_data['dataset_repo_id'],
        fold_data['val_episodes'],
        dataset_root="/datasets"
    )
    
    return {
        "fold_id": fold_id,
        "output_dir": str(output_dir),
        "val_result": val_result,
        "num_train_episodes": len(fold_data['train_episodes']),
        "num_val_episodes": len(fold_data['val_episodes']),
        "success": True
    }

def run_validation_on_fold(
    checkpoint_dir: Path, 
    dataset_repo_id: str, 
    val_episodes: List[int],
    dataset_root: str = "/datasets"
) -> dict:
    """Run validation on a trained model using the held-out fold."""
    import torch
    from lerobot.common.datasets.factory import make_dataset
    from lerobot.common.policies.factory import make_policy
    from lerobot.configs.default import DatasetConfig
    from train_val import TrainValPipelineConfig, run_validation
    
    # Find the latest checkpoint
    checkpoint_dirs = list(checkpoint_dir.glob("checkpoints/*/"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.name.split('_')[-1]))[-1]
    print(f"Loading checkpoint from {latest_checkpoint}")
    
    # Load the saved configuration
    with open(latest_checkpoint / "config.json", 'r') as f:
        saved_config = json.load(f)
    
    # Create validation dataset configuration
    val_dataset_config = DatasetConfig(
        repo_id=dataset_repo_id,
        root=dataset_root,
        episodes=val_episodes
    )
    
    # Create a minimal config for validation
    val_cfg = TrainValPipelineConfig(
        dataset=val_dataset_config,
        policy=None  # Will be loaded from checkpoint
    )
    
    # Load the policy
    policy = make_policy(saved_config['policy'], dataset_meta=None)
    
    # Load the model weights
    state_dict = torch.load(latest_checkpoint / "policy.pt", map_location="cpu")
    policy.load_state_dict(state_dict)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    
    # Create validation dataset
    val_dataset = make_dataset(val_cfg)
    
    # Run validation
    val_metrics = run_validation(
        policy=policy,
        val_dataset=val_dataset,
        device=device,
        batch_size=8,
        num_workers=4
    )
    
    print(f"Validation metrics: {val_metrics}")
    return val_metrics

@app.function(
    image=image,
    volumes={"/data": volume},
)
def aggregate_results(results: List[dict]) -> dict:
    """Aggregate results from all folds."""
    import numpy as np
    import pandas as pd
    
    print("Aggregating results from all folds...")
    
    # Extract validation metrics from all folds
    metrics_list = []
    successful_folds = []
    
    for result in results:
        if result["success"]:
            metrics_list.append(result["val_result"])
            successful_folds.append(result["fold_id"])
    
    if not metrics_list:
        raise ValueError("No successful folds to aggregate!")
    
    # Convert to DataFrame for easy aggregation
    df = pd.DataFrame(metrics_list)
    
    # Calculate mean and std for each metric
    summary = {
        "num_successful_folds": len(successful_folds),
        "successful_fold_ids": successful_folds,
        "metrics": {
            "mean": df.mean().to_dict(),
            "std": df.std().to_dict(),
            "min": df.min().to_dict(),
            "max": df.max().to_dict(),
        },
        "per_fold_results": metrics_list,
        "all_results": results
    }
    
    # Save summary
    output_file = Path("/data/outputs/kfold_summary.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("K-FOLD CROSS VALIDATION SUMMARY")
    print("="*50)
    print(f"Successful folds: {len(successful_folds)}/{len(results)}")
    print("\nMean metrics across folds:")
    for metric, value in summary["metrics"]["mean"].items():
        std = summary["metrics"]["std"][metric]
        print(f"  {metric}: {value:.4f} ± {std:.4f}")
    print("="*50)
    
    return summary

@app.local_entrypoint()
def main(
    dataset_repo_id: str = "jackvial/merged_datasets_test_2",
    k: int = 5,
    steps: int = 10000,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    gpu_type: str = "a10g",
    wandb_api_key: Optional[str] = None,
    local_splits_dir: str = "kfold_splits",
):
    """Main entry point for k-fold cross validation."""
    
    # Load pre-generated k-fold splits
    splits_dir = Path(local_splits_dir)
    if not splits_dir.exists():
        raise FileNotFoundError(
            f"K-fold splits not found in {splits_dir}. "
            f"Please run: python kfold_split_generator.py --dataset_repo_id={dataset_repo_id} --k={k}"
        )
    
    # Load fold data
    folds = []
    for i in range(k):
        with open(splits_dir / f"fold_{i}.json", 'r') as f:
            folds.append(json.load(f))
    
    # Upload splits to Modal volume
    print("Uploading k-fold splits to Modal volume...")
    volume.put_directory(local_splits_dir, "/data/kfold_splits")
    
    # Define training configuration
    training_config = {
        "policy.type": "act",
        "steps": steps,
        "batch_size": batch_size,
        "optimizer.lr": learning_rate,
        "log_freq": 100,
        "save_freq": 1000,
    }
    
    print(f"\nLaunching {k} parallel training jobs on {gpu_type} GPUs...")
    print(f"Training configuration: {training_config}")
    
    # Update GPU type for all functions
    train_fold.gpu = gpu_type
    
    # Launch parallel training for all folds using Modal's map
    fold_ids = list(range(k))
    fold_results = list(train_fold.map(
        fold_ids,
        [folds[i] for i in fold_ids],
        [training_config] * k,
        [wandb_api_key] * k
    ))
    
    # Aggregate results
    print("\nAggregating results from all folds...")
    summary = aggregate_results.remote(fold_results)
    
    print(f"\nK-fold cross validation complete!")
    print(f"Results saved to Modal volume at /data/outputs/kfold_summary.json")
    print(f"\nTo download results locally, run:")
    print(f"  modal volume get lerobot-kfold-data outputs/ ./kfold_results/")
    
    return summary

if __name__ == "__main__":
    # You can also run this directly with: modal run modal_kfold_training.py
    with modal.run():
        main() 