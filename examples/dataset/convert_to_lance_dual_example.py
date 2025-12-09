#!/usr/bin/env python
"""
Example: convert v3.0 dataset to Lance dual tables (episodes + frames).

Usage:
    python -m lerobot.examples.dataset.convert_to_lance_dual_example \
      --root /path/to/your/v30/dataset/root \
      [--episodes-out /path/to/<repo>.episodes.lance] \
      [--frames-out   /path/to/<repo>.frames.lance]

Dependencies:
    pip install lance av pyarrow
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lance.convert_dataset_v30_to_lance_dual import convert_dataset_v30_to_lance_dual


def main():
    parser = argparse.ArgumentParser(description="Convert v3.0 dataset to Lance dual tables")
    parser.add_argument("--root", type=str, required=True, help="v3.0 dataset root")
    parser.add_argument("--episodes-out", type=str, default=None, help="episodes.lance output directory")
    parser.add_argument("--frames-out", type=str, default=None, help="frames.lance output directory")
    args = parser.parse_args()

    root = Path(args.root)
    episodes_out = Path(args.episodes_out) if args.episodes_out else (root / f"{root.name}.episodes.lance")
    frames_out = Path(args.frames_out) if args.frames_out else (root / f"{root.name}.frames.lance")
    convert_dataset_v30_to_lance_dual(root, episodes_out, frames_out)


if __name__ == "__main__":
    main()
