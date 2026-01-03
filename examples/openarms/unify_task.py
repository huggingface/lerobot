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
Unify/remap tasks in a dataset based on shirt ID.

This script:
1. Loads a dataset with shirt_id feature
2. Assigns tasks based on shirt ID:
   - Shirt IDs 0XX (starting with 0): "Fold the T-shirt properly"
   - Shirt IDs 1XX, 2XX, etc.: "Layout the t-shirt on the table in an organized manner, then fold the t-shirt properly"
3. Updates tasks.parquet and task_index in data files

Usage:
    python unify_tasks.py \
        --input-repo-id lerobot-data-collection/full_folding_2025-11-30 \
        --output-repo-id lerobot-data-collection/single_task_folding_2025-11-30
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


# Task definitions based on shirt ID
TASK_FOLD_ONLY = "Fold the T-shirt properly"
TASK_LAYOUT_AND_FOLD = "Layout the t-shirt on the table in an organized manner, then fold the t-shirt properly"


def get_task_for_shirt_id(shirt_id: int) -> tuple[str, int]:
    """Get the task string and index based on shirt ID.

    Args:
        shirt_id: The shirt ID (e.g., 2, 112, 219)

    Returns:
        Tuple of (task_string, task_index)
        - Shirt IDs 0-99 (0XX): task_index=0, fold only
        - Shirt IDs 100+ (1XX, 2XX, ...): task_index=1, layout and fold
    """
    if shirt_id < 100:
        return TASK_FOLD_ONLY, 0
    return TASK_LAYOUT_AND_FOLD, 1


def unify_dataset_tasks(
    input_repo_id: str,
    output_repo_id: str,
    input_root: Path | None = None,
    output_root: Path | None = None,
    push_to_hub: bool = False,
) -> None:
    """Remap tasks in a dataset based on shirt ID.

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

    # Check if shirt_id feature exists
    if "shirt_id" not in src_meta.features:
        raise ValueError(
            "Dataset does not have 'shirt_id' feature. "
            "Please add it first using the add_features function."
        )

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

    # Process data files - update task_index based on shirt_id
    logging.info("Processing data files...")
    src_data_dir = input_root / DATA_DIR
    dst_data_dir = output_root / DATA_DIR
    dst_data_dir.mkdir(parents=True, exist_ok=True)

    # Track which tasks are used
    tasks_used = set()

    for src_parquet in tqdm(sorted(src_data_dir.rglob("*.parquet")), desc="Processing data"):
        rel_path = src_parquet.relative_to(input_root)
        dst_parquet = output_root / rel_path
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src_parquet)

        # Get shirt_id and compute task_index for each row
        if "shirt_id" in df.columns:
            # shirt_id might be shape (1,) array or scalar
            def extract_shirt_id(val):
                if hasattr(val, "__len__") and len(val) == 1:
                    return int(val[0])
                return int(val)

            df["task_index"] = df["shirt_id"].apply(
                lambda x: get_task_for_shirt_id(extract_shirt_id(x))[1]
            )

            # Track which tasks are used
            unique_shirt_ids = df["shirt_id"].apply(extract_shirt_id).unique()
            for sid in unique_shirt_ids:
                task_str, _ = get_task_for_shirt_id(sid)
                tasks_used.add(task_str)
        else:
            logging.warning(f"No shirt_id column in {src_parquet}, setting task_index=0")
            df["task_index"] = 0
            tasks_used.add(TASK_FOLD_ONLY)

        df.to_parquet(dst_parquet)

    # Process episodes metadata - update task references
    logging.info("Processing episodes metadata...")
    src_episodes_dir = input_root / "meta" / "episodes"
    dst_episodes_dir = output_root / "meta" / "episodes"
    dst_episodes_dir.mkdir(parents=True, exist_ok=True)

    # Build episode to shirt_id mapping by reading first frame of each episode
    episode_shirt_ids = {}
    for src_parquet in sorted(src_data_dir.rglob("*.parquet")):
        df = pd.read_parquet(src_parquet)
        if "shirt_id" in df.columns and "episode_index" in df.columns:
            for ep_idx in df["episode_index"].unique():
                if ep_idx not in episode_shirt_ids:
                    ep_data = df[df["episode_index"] == ep_idx].iloc[0]
                    shirt_val = ep_data["shirt_id"]
                    if hasattr(shirt_val, "__len__") and len(shirt_val) == 1:
                        episode_shirt_ids[int(ep_idx)] = int(shirt_val[0])
                    else:
                        episode_shirt_ids[int(ep_idx)] = int(shirt_val)

    for src_parquet in tqdm(sorted(src_episodes_dir.rglob("*.parquet")), desc="Processing episodes"):
        rel_path = src_parquet.relative_to(src_episodes_dir)
        dst_parquet = dst_episodes_dir / rel_path
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(src_parquet)

        # Update tasks column based on episode's shirt_id
        new_tasks_col = []
        for idx, row in df.iterrows():
            ep_idx = int(row["episode_index"])
            shirt_id = episode_shirt_ids.get(ep_idx, 0)
            task_str, _ = get_task_for_shirt_id(shirt_id)
            new_tasks_col.append([task_str])

        df["tasks"] = new_tasks_col
        df.to_parquet(dst_parquet)

    # Create new tasks.parquet with the tasks that are actually used
    logging.info(f"Creating tasks: {tasks_used}")
    task_list = sorted(tasks_used)  # Sort for consistent ordering
    # Ensure TASK_FOLD_ONLY is index 0 and TASK_LAYOUT_AND_FOLD is index 1
    if TASK_FOLD_ONLY in task_list and TASK_LAYOUT_AND_FOLD in task_list:
        task_list = [TASK_FOLD_ONLY, TASK_LAYOUT_AND_FOLD]
    elif TASK_FOLD_ONLY in task_list:
        task_list = [TASK_FOLD_ONLY]
    elif TASK_LAYOUT_AND_FOLD in task_list:
        # If only layout task is used, it should still be index 1 for consistency
        # But we need index 0 to exist, so include both
        task_list = [TASK_FOLD_ONLY, TASK_LAYOUT_AND_FOLD]

    new_tasks = pd.DataFrame(
        {"task_index": list(range(len(task_list)))},
        index=task_list
    )
    write_tasks(new_tasks, output_root)

    # Update info.json
    new_info = src_meta.info.copy()
    new_info["total_tasks"] = len(task_list)
    write_info(new_info, output_root)

    # Copy stats.json (unchanged)
    if src_meta.stats:
        write_stats(src_meta.stats, output_root)

    logging.info(f"Dataset saved to {output_root}")
    logging.info(f"Tasks: {task_list}")

    if push_to_hub:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        logging.info(f"Pushing {output_repo_id} to hub")
        dataset = LeRobotDataset(output_repo_id, root=output_root)
        dataset.push_to_hub(private=True)
        logging.info("Push complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Remap tasks in a dataset based on shirt ID. "
        "Shirt IDs 0-99 get 'Fold the T-shirt properly', "
        "Shirt IDs 100+ get 'Layout and fold' task."
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
