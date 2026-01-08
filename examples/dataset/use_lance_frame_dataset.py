#!/usr/bin/env python
"""
Example: iterate frames with LanceFrameDataset and decode single frames by timestamp.

Prerequisites:
    pip install lance av pyarrow torch

Usage:
    python -m lerobot.examples.dataset.use_lance_frame_dataset --path /path/to/lerobot.lance
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.datasets.lance.lance_dataset import LanceFrameDataset


def main():
    parser = argparse.ArgumentParser(description="Example: iterate frames with LanceFrameDataset and decode single frames")
    parser.add_argument("--path", type=str, required=True, help="Lance dataset directory, e.g. /data/lerobot.lance")
    args = parser.parse_args()

    lance_path = Path(args.path)
    # Simple identity transform; in practice pass torchvision.transforms.v2 pipeline
    image_transforms = lambda x: x

    ds = LanceFrameDataset(lance_path, image_transforms=image_transforms)
    print(f"Total frames: {len(ds)}")

    # Take first 3 frames and print keys
    for i in range(min(3, len(ds))):
        item = ds[i]
        keys = list(item.keys())
        print(f"Sample {i}: keys={keys}")
        # To access an image tensor: item.get('observation.images.cam', None)
        # Pick one image key and perform a basic check
        cam_keys = [k for k in keys if k.startswith("observation.images.")]
        if cam_keys:
            img = item[cam_keys[0]]
            assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"
            print(f"  Image shape: {tuple(img.shape)} (C,H,W)")


if __name__ == "__main__":
    main()
