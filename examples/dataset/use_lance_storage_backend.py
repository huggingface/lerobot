#!/usr/bin/env python
"""
Example: use the Lance storage backend of LeRobotDataset (storage_backend='lance').

Steps:
1) First convert an existing v3.0 dataset root to Lance (one row per episode + video blobs).
   For example:
       python -m lerobot.examples.dataset.convert_to_lance_example \
           --root /path/to/v30_root \
           [--out /path/to/<root.name>.lance]  # optional, defaults to root/<root.name>.lance

2) Then iterate frames from the Lance table via LeRobotDataset(storage_backend='lance') (default Lance directory is root/<repo_id>.lance).

Prerequisites:
    pip install lance av pyarrow torch
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Example: use LeRobotDataset with the Lance storage backend")
    parser.add_argument("--root", type=str, required=True, help="v3.0 dataset root (contains meta/data/videos)")
    parser.add_argument("--repo-id", type=str, required=True, help="HF dataset repo id (used for the default Lance directory name)")
    parser.add_argument("--opt", action="append", default=[], help="Optional Lance storage options as key=value pairs (can be repeated), e.g., --opt s3.region=us-east-1 --opt read_params.timeout_ms=5000")
    args = parser.parse_args()

    root = Path(args.root)

    # Simple identity transform; in practice pass torchvision.transforms.v2 pipeline
    image_transforms = lambda x: x

    # Parse --opt key=value pairs into a dict
    def parse_opts(opt_list):
        opts = {}
        for kv in opt_list:
            if "=" in kv:
                k, v = kv.split("=", 1)
                opts[k] = v
        return opts

    storage_opts = parse_opts(args.opt)

    # Use storage_backend='lance' to read frames via Lance; default Lance directory is root/<repo_id>.lance
    # storage_backend_options is optional; pass only if you need to configure remote/object storage or custom reader params.
    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        image_transforms=image_transforms,
        storage_backend="lance",
        storage_backend_options=storage_opts or None,
    )

    print(f"Total frames: {len(ds)}")
    for i in range(min(3, len(ds))):
        item = ds[i]
        keys = list(item.keys())
        print(f"Sample {i}: keys={keys}")

        # Pick one image key to perform a basic check (if any)
        cam_keys = [k for k in keys if k.startswith("observation.images.")]
        if cam_keys:
            img = item[cam_keys[0]]
            try:
                import torch  # noqa: F401
                assert hasattr(img, "shape"), "Image should be a torch.Tensor"
                print(f"  Image shape: {tuple(img.shape)} (C,H,W)")
            except Exception:
                pass


if __name__ == "__main__":
    main()
