#!/usr/bin/env python
"""
Example: Load LeRobotDataset using Lance as the storage backend and read a sample.

Usage:
- Install dependencies: `pip install lerobot datasets lance`
- The default Parquet layout remains unchanged; when storage_backend='lance'
  this example reads from `data/*/*.lance`.
- If no .lance files exist locally, run the converter:
  `python -m lerobot.scripts.lerobot_convert_to_lance --repo-id lerobot/pusht`

Note: Videos are still decoded from mp4 files; the storage backend does not affect video decoding.
"""
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    # Example 1: load from a local root (parquet→lance conversion recommended first)
    # root = Path("/your/local/cache/lerobot")
    # ds = LeRobotDataset(
    #     repo_id="lerobot/pusht",
    #     root=root / "lerobot/pusht",
    #     storage_backend="lance",
    # )

    # Example 2: load from the default cache root (~/.cache/huggingface/lerobot/{repo_id})
    ds = LeRobotDataset(
        repo_id="lerobot/pusht",
        storage_backend="lance",
    )

    print(ds)
    # Read a single sample
    item = ds[0]
    keys_preview = list(item.keys())[:10]
    print(f"Sample 0 keys: {keys_preview} ...")


if __name__ == "__main__":
    main()
