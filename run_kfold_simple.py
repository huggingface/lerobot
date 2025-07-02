#!/usr/bin/env python3
"""Simple script to run k-fold cross validation on Modal Labs."""

import os
import subprocess
import sys
from pathlib import Path

def main():
    # Configuration
    config = {
        "dataset_repo_id": "jackvial/merged_datasets_test_2",
        "k": 5,
        "steps": 100000,
        "batch_size": 8,
        "learning_rate": 1e-5,
        "gpu_type": "a10g",  # t4 ($0.59/hr), a10g ($1.14/hr), a100 ($3.19/hr)
    }
    
    print("K-Fold Cross Validation Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Step 1: Generate k-fold splits
    print("Step 1: Generating k-fold splits...")
    cmd = [
        sys.executable, "kfold_split_generator.py",
        f"--dataset_repo_id={config['dataset_repo_id']}",
        f"--k={config['k']}",
        "--output_dir=kfold_splits",
        "--seed=42"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Failed to generate k-fold splits")
        return 1
    
    # Step 2: Check Modal installation
    print("\nStep 2: Checking Modal installation...")
    try:
        import modal
    except ImportError:
        print("Installing Modal...")
        subprocess.run([sys.executable, "-m", "pip", "install", "modal"])
        import modal
    
    # Step 3: Run on Modal
    print("\nStep 3: Launching k-fold training on Modal...")
    
    # Build Modal command
    modal_cmd = [
        "modal", "run", "modal_kfold_training.py",
        f"--dataset-repo-id={config['dataset_repo_id']}",
        f"--k={config['k']}",
        f"--steps={config['steps']}",
        f"--batch-size={config['batch_size']}",
        f"--learning-rate={config['learning_rate']}",
        f"--gpu-type={config['gpu_type']}",
    ]
    
    # Add WandB API key if available
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        print("WandB API key detected, will log to WandB")
        modal_cmd.append(f"--wandb-api-key={wandb_key}")
    else:
        print("No WandB API key found, running without WandB logging")
    
    # Run Modal app
    result = subprocess.run(modal_cmd)
    
    if result.returncode == 0:
        print("\nK-fold training completed successfully!")
        
        # Step 4: Download results
        print("\nStep 4: Downloading results...")
        Path("kfold_results").mkdir(exist_ok=True)
        
        download_cmd = [
            "modal", "volume", "get",
            "lerobot-kfold-data", "outputs/",
            "./kfold_results/"
        ]
        subprocess.run(download_cmd)
        
        print("\nResults downloaded to ./kfold_results/")
        print("Summary file: ./kfold_results/kfold_summary.json")
        
        # Display summary if available
        summary_file = Path("kfold_results/kfold_summary.json")
        if summary_file.exists():
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            print("\nK-Fold Cross Validation Summary:")
            print(f"Successful folds: {summary['num_successful_folds']}")
            print("\nMean metrics:")
            for metric, value in summary['metrics']['mean'].items():
                std = summary['metrics']['std'][metric]
                print(f"  {metric}: {value:.4f} ± {std:.4f}")
    else:
        print("K-fold training failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 