#!/usr/bin/env python

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def check_debug_dataset():
    """Check the debug dataset for potential issues."""

    print("🔍 Checking debug dataset...")

    # Load your debug dataset
    dataset = LeRobotDataset("a6047425318/green-marker-part2-ep0-debug")

    print(f"Dataset size: {len(dataset)} frames")
    print(f"Number of episodes: {dataset.num_episodes}")
    print(f"Total frames: {dataset.num_frames}")

    # Check first few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(
                    f"  {key}: shape={value.shape}, dtype={value.dtype}, min={value.min():.4f}, max={value.max():.4f}"
                )
            else:
                print(f"  {key}: {type(value)} - {value}")

    # Check for NaN values
    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"⚠️  WARNING: NaN values found in {key}")
            if torch.isinf(value).any():
                print(f"⚠️  WARNING: Inf values found in {key}")

    # Check action statistics
    if "action" in sample:
        actions = []
        for i in range(min(100, len(dataset))):
            actions.append(dataset[i]["action"])
        actions = torch.stack(actions)
        print("\nAction statistics:")
        print(f"  Shape: {actions.shape}")
        print(f"  Mean: {actions.mean(dim=0)}")
        print(f"  Std: {actions.std(dim=0)}")
        print(f"  Min: {actions.min(dim=0)[0]}")
        print(f"  Max: {actions.max(dim=0)[0]}")


if __name__ == "__main__":
    check_debug_dataset()
