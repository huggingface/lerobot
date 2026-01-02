#!/usr/bin/env python3
"""
Exploration script for the LeRobot NLTuan/up-down dataset.
This script helps you understand:
1. Dataset metadata (FPS, size, etc.)
2. Available features (images, state, actions)
3. Natural language tasks included
4. Data structure and sample values
"""

import sys
from pathlib import Path
import torch

# Add the src directory to the python path
sys.path.append(str(Path(__file__).parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    dataset_repo_id = "NLTuan/up-down"
    
    print(f"--- Exploring Dataset: {dataset_repo_id} ---")
    
    # 1. Load Dataset
    # This will download metadata and potentially some data if not present
    try:
        # We force video_backend="pyav" because torchcodec requires system-level ffmpeg
        dataset = LeRobotDataset(dataset_repo_id, video_backend="pyav")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have 'datasets' and 'lerobot' installed in your environment.")
        return

    # 2. Basic Metadata
    print("\n[Metadata]")
    print(f"Total Episodes: {dataset.meta.total_episodes}")
    print(f"Total Frames:   {dataset.meta.total_frames}")
    print(f"FPS:            {dataset.meta.fps}")
    print(f"Robot Type:     {dataset.meta.robot_type}")

    # 3. Features
    print("\n[Features]")
    for key, feat in dataset.meta.features.items():
        shape = feat.get("shape", "N/A")
        dtype = feat.get("dtype", "N/A")
        print(f"- {key}: shape={shape}, dtype={dtype}")

    # 4. Tasks (Language Instructions)
    print("\n[Tasks/Instructions]")
    if dataset.meta.tasks is not None:
        for i, task in enumerate(dataset.meta.tasks.index):
            print(f"{i}. \"{task}\"")
    else:
        print("No explicit tasks found in metadata.")

    # 5. Sample Data (First Frame)
    print("\n[Sample Data - Frame 0]")
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"- {key}: tensor of shape {value.shape}")
        else:
            print(f"- {key}: {value}")

    print("\n--- Exploration Complete ---")
    print("Note: Images are stored as videos (.mp4) in the 'videos/' directory.")
    print("Metadata and vector data (state, action) are stored in .parquet files in 'data/'.")

if __name__ == "__main__":
    main()
