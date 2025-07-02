#!/usr/bin/env python3
"""Generate k-fold splits for LeRobot datasets."""

import json
import numpy as np
from pathlib import Path
import argparse

def create_kfold_splits(
    dataset_repo_id: str, 
    k: int = 5, 
    output_dir: Path = Path("kfold_splits"), 
    seed: int = 42
):
    """Generate k-fold splits and save to disk."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except ImportError:
        print("Please install lerobot first: pip install lerobot")
        return None
    
    meta = LeRobotDatasetMetadata(dataset_repo_id)
    total_episodes = meta.total_episodes
    
    print(f"Dataset: {dataset_repo_id}")
    print(f"Total episodes: {total_episodes}")
    print(f"Creating {k}-fold splits...")
    
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
        
        print(f"Fold {i}: {len(train_episodes)} train, {len(val_episodes)} val episodes")
    
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
    
    print(f"\nCreated {k}-fold splits in {output_dir}/")
    print(f"Summary saved to {output_dir}/kfold_summary.json")
    
    return folds

def main():
    parser = argparse.ArgumentParser(description="Generate k-fold splits for LeRobot datasets")
    parser.add_argument("--dataset_repo_id", type=str, required=True, help="HuggingFace dataset repository ID")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--output_dir", type=str, default="kfold_splits", help="Output directory for splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    create_kfold_splits(
        dataset_repo_id=args.dataset_repo_id,
        k=args.k,
        output_dir=Path(args.output_dir),
        seed=args.seed
    )

if __name__ == "__main__":
    main() 