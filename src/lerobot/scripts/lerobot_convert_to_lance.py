#!/usr/bin/env python
"""
Convert existing LeRobotDataset Parquet data to parallel Lance files.

- Input: repo_id and optional root (defaults to HF_LEROBOT_HOME/{repo_id})
- Output: write .lance files alongside data/*/*.parquet (same basename, different extension)

Examples:
    python -m lerobot.scripts.lerobot_convert_to_lance --repo-id lerobot/pusht
    python -m lerobot.scripts.lerobot_convert_to_lance --repo-id lerobot/pusht --root /tmp/lerobot_cache

Install lance first:
    pip install lance
"""
from pathlib import Path
import argparse

import datasets

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import get_hf_features_from_features


def convert_parquet_chunk_to_lance(parquet_path: Path, hf_features: datasets.Features) -> None:
    try:
        import lance  # type: ignore
    except Exception as e:
        raise ImportError("The 'lance' package is required to write .lance files. Please `pip install lance` first.") from e

    # Read each parquet file into a HF Dataset, then write it as a Lance dataset
    ds = datasets.Dataset.from_parquet(str(parquet_path), features=hf_features, split="train")
    out_path = parquet_path.with_suffix(".lance")
    lance.write_dataset(ds, str(out_path))
    print(f"Wrote Lance dataset: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot parquet data to Lance files")
    parser.add_argument("--repo-id", type=str, required=True, help="HF dataset repo id, e.g. lerobot/pusht")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local root directory; defaults to ~/.cache/huggingface/lerobot/{repo_id}",
    )
    args = parser.parse_args()

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    data_dir = ds.root / "data"
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found: {data_dir}")

    hf_features = get_hf_features_from_features(ds.features)
    for p in parquet_files:
        convert_parquet_chunk_to_lance(p, hf_features)

    print("Done. You can now load with storage_backend='lance'.")


if __name__ == "__main__":
    main()
