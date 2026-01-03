#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unify all tasks in a dataset to a single task (modifies in-place).

This script:
1. Loads a dataset
2. Sets all task_index to 0 and task description to "fold"
3. Updates tasks.parquet and task_index in data files (in-place, no copying)

Usage:
    python examples/openarms/unify_task.py --repo-id lerobot-data-collection/level1_rac1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DATA_DIR,
    write_info,
    write_tasks,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


# Single unified task
UNIFIED_TASK = "fold"


def unify_dataset_tasks(
    repo_id: str,
    root: Path | None = None,
    push_to_hub: bool = False,
) -> None:
    """Unify all tasks in a dataset to a single task (modifies in-place).

    Args:
        repo_id: Dataset repository ID.
        root: Optional root path for dataset.
        push_to_hub: Whether to push the result to HuggingFace Hub.
    """
    input_root = root if root else HF_LEROBOT_HOME / repo_id
    input_repo_id = repo_id

    logging.info(f"Loading metadata from {repo_id}")

    # Load source metadata
    src_meta = LeRobotDatasetMetadata(repo_id, root=input_root)

    logging.info(f"Source dataset: {src_meta.total_episodes} episodes, {src_meta.total_frames} frames")
    logging.info(f"Original tasks: {len(src_meta.tasks)}")

    # Modify in-place (input_root == output_root supported)
    data_dir = input_root / DATA_DIR

    # Process data files - set all task_index to 0
    logging.info("Processing data files (in-place)...")
    for parquet_file in tqdm(sorted(data_dir.rglob("*.parquet")), desc="Processing data"):
        df = pd.read_parquet(parquet_file)
        df["task_index"] = 0  # All tasks unified to index 0
        df.to_parquet(parquet_file)

    # Process episodes metadata - set all tasks to unified task
    logging.info("Processing episodes metadata (in-place)...")
    episodes_dir = input_root / "meta" / "episodes"
    if episodes_dir.exists():
        for parquet_file in tqdm(sorted(episodes_dir.rglob("*.parquet")), desc="Processing episodes"):
            df = pd.read_parquet(parquet_file)
            df["tasks"] = [[UNIFIED_TASK]] * len(df)  # All episodes get the unified task
            df.to_parquet(parquet_file)
    else:
        logging.warning(f"No episodes directory found at {episodes_dir}, skipping")

    # Update tasks.parquet with single task
    logging.info(f"Creating single task: {UNIFIED_TASK}")
    new_tasks = pd.DataFrame({"task_index": [0]}, index=[UNIFIED_TASK])
    write_tasks(new_tasks, input_root)

    # Update info.json
    new_info = src_meta.info.copy()
    new_info["total_tasks"] = 1
    write_info(new_info, input_root)

    logging.info(f"Dataset modified in-place at {input_root}")
    logging.info(f"Task: {UNIFIED_TASK}")

    if push_to_hub:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        logging.info(f"Pushing {input_repo_id} to hub")
        dataset = LeRobotDataset(input_repo_id, root=input_root)
        dataset.push_to_hub(private=True)
        logging.info("Push complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Unify all tasks in a dataset to a single task 'fold' (modifies in-place)."
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repository ID",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional root path (defaults to HF_LEROBOT_HOME/repo_id)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push result to HuggingFace Hub",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    unify_dataset_tasks(
        repo_id=args.repo_id,
        root=args.root,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
