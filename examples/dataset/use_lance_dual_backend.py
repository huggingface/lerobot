#!/usr/bin/env python
"""
Example: use Lance dual tables (episodes + frames) for random frame and window sampling.

Prerequisites:
    pip install lance av pyarrow torch

Usage:
    python -m lerobot.examples.dataset.use_lance_dual_backend \
        --episodes /path/to/<repo>.episodes.lance \
        --frames /path/to/<repo>.frames.lance
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lance.lance_dual_dataset import LanceFramesTable


def main():
    parser = argparse.ArgumentParser(description="Example: iterate frames via Lance dual tables")
    parser.add_argument("--episodes", type=str, required=True, help="Episodes Lance directory")
    parser.add_argument("--frames", type=str, required=True, help="Frames Lance directory")
    args = parser.parse_args()

    ds = LanceFramesTable(Path(args.frames), Path(args.episodes), image_transforms=lambda x: x)
    print(f"Total frames: {len(ds)}")

    # Take first 3 frames
    for i in range(min(3, len(ds))):
        item = ds[i]
        keys = list(item.keys())
        print(f"Sample {i}: keys={keys}")

    # Sample a window of 5 frames starting at index 10 (if available)
    if len(ds) > 15:
        window = ds.sample_window(10, 5)
        print(f"Window length: {len(window)}; start frame_index={window[0]['frame_index'].item()} end frame_index={window[-1]['frame_index'].item()}")


if __name__ == "__main__":
    main()
