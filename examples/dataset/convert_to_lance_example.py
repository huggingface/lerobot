#!/usr/bin/env python
"""
Example: convert an existing v3.0 dataset directory to Lance (one row per episode + video blobs).

Usage:
    python -m lerobot.examples.dataset.convert_to_lance_example \
      --root /path/to/your/v30/dataset/root \
      [--out /path/to/output/<root.name>.lance]  # optional, defaults to root/<root.name>.lance

Dependencies:
    pip install lance av pyarrow
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lance.convert_dataset_v30_to_lance import convert_dataset_v30_to_lance


def main():
    parser = argparse.ArgumentParser(description="Example: convert a v3.0 dataset to Lance")
    parser.add_argument("--root", type=str, required=True, help="v3.0 dataset root")
    parser.add_argument("--out", type=str, default=None, help="output Lance dataset directory (default root/<root.name>.lance)")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else (root / f"{root.name}.lance")
    convert_dataset_v30_to_lance(root, out)


if __name__ == "__main__":
    main()
