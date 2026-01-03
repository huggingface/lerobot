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
Unify all tasks in a dataset to a single task.

This script:
1. Loads a dataset
2. Sets all task_index to 0 and task description to "fold"
3. Updates tasks.parquet and task_index in data files

Usage:
    python examples/openarms/unify_task.py \
        --input-repo-id lerobot-data-collection/level1_rac1 \
        --output-repo-id lerobot-data-collection/level1_rac1
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DATA_DIR,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.utils.constants import HF_LEROBOT_HOME


# Single unified task
UNIFIED_TASK = "fold"


def unify_dataset_tasks(
    input_repo_id: str,
    output_repo_id: str,
    input_root: Path | None = None,
    output_root: Path | None = None,
    push_to_hub: bool = False,
) -> None:
    """Unify all tasks in a dataset to a single task.

    Args:
        input_repo_id: Source dataset repository ID.
        output_repo_id: Output dataset repository ID.
        input_root: Optional root path for input dataset.
        output_root: Optional root path for output dataset.
        push_to_hub: Whether to push the result to HuggingFace Hub.
    """
    logging.info(f"Loading metadata from {input_repo_id}")

    input_root = input_root if input_root else HF_LEROBOT_HOME / input_repo_id
    output_root = output_root if output_root else HF_LEROBOT_HOME / output_repo_id

    # Load source metadata
    src_meta = LeRobotDatasetMetadata(input_repo_id, root=input_root)

    logging.info(f"Source dataset: {src_meta.total_episodes} episodes, {src_meta.total_frames} frames")
    logging.info(f"Original tasks: {len(src_meta.tasks)}")

    # Create output directory
    if output_root.exists():
        logging.warning(f"Output directory {output_root} exists, removing it")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    # Copy videos directory (no changes needed)
    src_videos = input_root / "videos"
    if src_videos.exists():
        logging.info("Copying videos...")
        shutil.copytree(src_videos, output_root / "videos")

    # Process data files - set all task_index to 0
    logging.info("Processing data files...")
    src_data_dir = input_root / DATA_DIR
    dst_data_dir = output_root / DATA_DIR
    dst_data_dir.mkdir(parents=True, exist_ok=True)

    for src_parquet in tqdm(sorted(src_data_dir.rglob("*.parquet")), desc="Processing data"):
        rel_path = src_parquet.relative_to(input_root)
        dst_parquet = output_root / rel_path
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src_parquet)
        df["task_index"] = 0  # All tasks unified to index 0
        df.to_parquet(dst_parquet)

    # Process episodes metadata - set all tasks to unified task
    logging.info("Processing episodes metadata...")
    src_episodes_dir = input_root / "meta" / "episodes"
    dst_episodes_dir = output_root / "meta" / "episodes"
    dst_episodes_dir.mkdir(parents=True, exist_ok=True)

    for src_parquet in tqdm(sorted(src_episodes_dir.rglob("*.parquet")), desc="Processing episodes"):
        rel_path = src_parquet.relative_to(src_episodes_dir)
        dst_parquet = dst_episodes_dir / rel_path
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src_parquet)
        df["tasks"] = [[UNIFIED_TASK]] * len(df)  # All episodes get the unified task
        df.to_parquet(dst_parquet)

    # Create new tasks.parquet with single task
    logging.info(f"Creating single task: {UNIFIED_TASK}")
    new_tasks = pd.DataFrame({"task_index": [0]}, index=[UNIFIED_TASK])
    write_tasks(new_tasks, output_root)

    # Update info.json
    new_info = src_meta.info.copy()
    new_info["total_tasks"] = 1
    write_info(new_info, output_root)

    # Copy stats.json (unchanged)
    if src_meta.stats:
        write_stats(src_meta.stats, output_root)

    logging.info(f"Dataset saved to {output_root}")
    logging.info(f"Task: {UNIFIED_TASK}")

    if push_to_hub:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        logging.info(f"Pushing {output_repo_id} to hub")
        dataset = LeRobotDataset(output_repo_id, root=output_root)
        dataset.push_to_hub(private=True)
        logging.info("Push complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Unify all tasks in a dataset to a single task 'fold'."
    )

    parser.add_argument(
        "--input-repo-id",
        type=str,
        default="lerobot-data-collection/full_folding_2025-11-30",
        help="Input dataset repository ID",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default="lerobot-data-collection/folding_2025-11-30",
        help="Output dataset repository ID",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional input root path (defaults to HF_LEROBOT_HOME/input_repo_id)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root path (defaults to HF_LEROBOT_HOME/output_repo_id)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push result to HuggingFace Hub",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    unify_dataset_tasks(
        input_repo_id=args.input_repo_id,
        output_repo_id=args.output_repo_id,
        input_root=args.input_root,
        output_root=args.output_root,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
