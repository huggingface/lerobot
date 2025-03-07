#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import argparse
import logging
import shutil
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import datasets

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    TASKS_PATH,
    append_jsonlines,
    embed_images,
    get_episode_data_index,
    hf_transform_to_torch,
    write_episode,
    write_episode_stats,
    write_info,
)
from lerobot.common.utils.utils import init_logging


def remove_episodes(
    dataset: LeRobotDataset,
    episodes_to_remove: list[int],
    backup: str | Path | bool = False,
) -> LeRobotDataset:
    """
    Removes specified episodes from a LeRobotDataset and updates all metadata and files accordingly.

    Args:
        dataset: The LeRobotDataset to modify
        episodes_to_remove: List of episode indices to remove
        backup: Controls backup behavior:
                   - False: No backup is created
                   - True: Create backup at default location next to dataset
                   - str/Path: Create backup at the specified location

    Returns:
        Updated LeRobotDataset with specified episodes removed
    """
    if not episodes_to_remove:
        return dataset

    if not all(0 <= ep_idx < dataset.meta.total_episodes for ep_idx in episodes_to_remove):
        raise ValueError("Episodes to remove must be valid episode indices in the dataset")

    # Get mapping from old episode indices to new (post-removal) indices
    remaining_episode_indices = [i for i in range(dataset.meta.total_episodes) if i not in episodes_to_remove]
    old_to_new = {old: new for new, old in enumerate(remaining_episode_indices)}

    # Step 1: Filter out removed episodes and update episode indices in hf_dataset
    new_hf_dataset = _filter_hf_dataset(dataset, episodes_to_remove, old_to_new)

    # Step 2: Update metadata
    new_meta = deepcopy(dataset.meta)
    new_meta.info["total_episodes"] = len(old_to_new)
    new_meta.info["total_frames"] = len(new_hf_dataset)
    new_meta.info["total_chunks"] = (
        (new_meta.info["total_episodes"] - 1) // new_meta.chunks_size + 1
        if new_meta.info["total_episodes"] > 0
        else 0
    )
    new_episodes = _create_new_episodes_dict(dataset, old_to_new)
    new_meta.episodes = new_episodes
    old_to_new_task, new_tasks, new_task_to_task_index = _build_new_tasks_mapping(dataset, new_episodes)
    new_meta.tasks = new_tasks
    new_meta.task_to_task_index = new_task_to_task_index
    new_meta.info["total_tasks"] = len(new_tasks)

    new_episodes_stats = {}
    for old_idx, new_idx in old_to_new.items():
        if old_idx in dataset.meta.episodes_stats:
            new_episodes_stats[new_idx] = deepcopy(dataset.meta.episodes_stats[old_idx])

    new_meta.episodes_stats = new_episodes_stats
    new_meta.stats = aggregate_stats(list(new_episodes_stats.values()))

    new_meta.info["total_videos"] = (
        len(old_to_new) * len(dataset.meta.video_keys) if dataset.meta.video_keys else 0
    )

    if "splits" in new_meta.info:
        new_meta.info["splits"] = {"train": f"0:{new_meta.info['total_episodes']}"}

    # Step 3: Update task indices in new hf dataset
    def update_task_indices(batch):
        batch["task_index"] = [old_to_new_task[idx.item()] for idx in batch["task_index"]]
        return batch

    new_hf_dataset = new_hf_dataset.map(update_task_indices, batched=True)

    # Now, we have all the updated metadata and dataset files.
    # We will first try to write them all to a temporary dir, and if successful,
    # we will replace the original dataset with the updated one.
    # This is to avoid corrupting the original dataset in case of any errors.
    # Optionally, the original dataset can be backed up before replacing it,
    # using the backup argument.
    temp_dir = tempfile.mkdtemp(prefix="lerobot_dataset_temp_")
    temp_root = Path(temp_dir)

    try:
        _write_new_dataset_files(
            dataset,
            old_to_new,
            new_hf_dataset,
            new_meta,
            new_episodes,
            new_tasks,
            new_episodes_stats,
            temp_root,
        )

        _replace_folder(dataset.root, temp_root, backup)

        # Reload the dataset with the updated files
        updated_dataset = LeRobotDataset(
            repo_id=dataset.repo_id,
            root=dataset.root,
            episodes=None,  # Load all episodes
            image_transforms=dataset.image_transforms,
            delta_timestamps=dataset.delta_timestamps,
            tolerance_s=dataset.tolerance_s,
            revision=dataset.revision,
            download_videos=False,  # No need to download, we just saved them
            video_backend=dataset.video_backend,
        )

        return updated_dataset

    except Exception as e:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise RuntimeError(f"Error during dataset reorganization: {str(e)}") from e
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


def _create_new_episodes_dict(dataset: LeRobotDataset, old_to_new: dict[int, int]) -> dict:
    """
    Extract the episodes that are still used and build a new episodes dictionary.
    """
    new_episodes = {}
    for old_idx, new_idx in old_to_new.items():
        if old_idx in dataset.meta.episodes:
            ep_data = deepcopy(dataset.meta.episodes[old_idx])
            ep_data["episode_index"] = new_idx
            new_episodes[new_idx] = ep_data
    return new_episodes


def _build_new_tasks_mapping(
    dataset: LeRobotDataset, new_episodes: dict
) -> tuple[dict[int, int], dict[int, str], dict[str, int]]:
    """
    Determine which tasks are still used and build new mapping structures.
    """
    used_task_indices = set()
    for ep_data in new_episodes.values():
        if "tasks" in ep_data:
            for task in ep_data["tasks"]:
                task_idx = dataset.meta.get_task_index(task)
                if task_idx is not None:
                    used_task_indices.add(task_idx)

    old_to_new_task = {}
    new_tasks = {}
    new_task_to_task_index = {}

    for new_idx, old_idx in enumerate(sorted(used_task_indices)):
        if old_idx in dataset.meta.tasks:
            task = dataset.meta.tasks[old_idx]
            new_tasks[new_idx] = task
            new_task_to_task_index[task] = new_idx
            old_to_new_task[old_idx] = new_idx

    return old_to_new_task, new_tasks, new_task_to_task_index


def _write_new_dataset_files(
    dataset: LeRobotDataset,
    old_to_new: dict[int, int],
    new_hf_dataset: datasets.Dataset,
    new_meta: LeRobotDatasetMetadata,
    new_episodes: dict,
    new_tasks: dict[int, str],
    new_episodes_stats: dict[int, dict],
    temp_root: Path,
):
    """
    Write the updated metadata and dataset files to a temporary directory.
    """
    new_episode_data_index = get_episode_data_index(new_episodes)

    (temp_root / "meta").mkdir(parents=True, exist_ok=True)
    (temp_root / "data").mkdir(parents=True, exist_ok=True)
    if dataset.meta.video_keys:
        (temp_root / "videos").mkdir(parents=True, exist_ok=True)

    write_info(new_meta.info, temp_root)

    for ep_data in new_episodes.values():
        write_episode(ep_data, temp_root)

    for task_idx, task in new_tasks.items():
        task_dict = {
            "task_index": task_idx,
            "task": task,
        }
        append_jsonlines(task_dict, temp_root / TASKS_PATH)

    for ep_idx, ep_stats in new_episodes_stats.items():
        write_episode_stats(ep_idx, ep_stats, temp_root)

    for old_idx, new_idx in old_to_new.items():
        new_chunk = new_meta.get_episode_chunk(new_idx)
        (temp_root / "data" / f"chunk-{new_chunk:03d}").mkdir(parents=True, exist_ok=True)

        from_idx = new_episode_data_index["from"][new_idx].item()
        to_idx = new_episode_data_index["to"][new_idx].item()

        ep_slice = new_hf_dataset[from_idx:to_idx]
        ep_data_path = temp_root / f"data/chunk-{new_chunk:03d}/episode_{new_idx:06d}.parquet"
        ep_dataset = datasets.Dataset.from_dict(
            {k: ep_slice[k] for k in ep_slice},
            features=new_hf_dataset.features,
            split="train",
        )

        ep_dataset = embed_images(ep_dataset)
        ep_dataset.set_transform(hf_transform_to_torch)
        ep_dataset.to_parquet(ep_data_path)

        if dataset.meta.video_keys:
            for vid_key in dataset.meta.video_keys:
                old_video_path = dataset.root / dataset.meta.get_video_file_path(old_idx, vid_key)
                if old_video_path.exists():
                    new_video_dir = temp_root / f"videos/chunk-{new_chunk:03d}/{vid_key}"
                    new_video_dir.mkdir(parents=True, exist_ok=True)
                    new_video_path = new_video_dir / f"episode_{new_idx:06d}.mp4"
                    shutil.copy2(old_video_path, new_video_path)


def _replace_folder(target_dir: Path, source_dir: Path, backup: str | Path | bool = False):
    """
    Replace a directory with another one, optionally creating a backup of the original.

    Args:
        target_dir: Path to the directory to replace
        source_dir: Path to the new directory to replace it with
        backup: Controls backup behavior:
                   - False: No backup is created
                   - True: Create backup at default location next to dataset
                   - str/Path: Create backup at the specified location
    """
    if target_dir.exists():
        if backup:
            backup_path = (
                Path(backup)
                if isinstance(backup, (str, Path))
                else target_dir.parent / f"{target_dir.name}_backup_{int(time.time())}"
            )

            if backup_path.resolve() == target_dir.resolve() or backup_path.resolve().is_relative_to(
                target_dir.resolve()
            ):
                raise ValueError(
                    f"Backup directory '{backup_path}' cannot be inside the dataset "
                    f"directory '{target_dir}' as this would cause infinite recursion"
                )

            backup_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Creating backup at: {backup_path}")
            shutil.copytree(target_dir, backup_path)

        # Remove the target directory's contents
        for item in target_dir.glob("*"):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    # Replace the contents of the target directory with the contents of the source directory
    for item in source_dir.glob("*"):
        if item.is_dir():
            shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target_dir / item.name)


def _filter_hf_dataset(
    dataset: LeRobotDataset, episodes_to_remove: list[int], old_to_new: dict[int, int]
) -> datasets.Dataset:
    """
    Filter a dataset to remove specified episodes and update episode indices.
    """

    def create_filter_and_map_fn():
        # We use a closure here to keep track of the global frame index across batches in the mapping
        current_global_idx = 0

        def filter_and_map_fn(batch, indices):
            nonlocal current_global_idx
            keep_mask = [ep_idx not in episodes_to_remove for ep_idx in batch["episode_index"]]

            if not any(keep_mask):
                return {k: [] for k in batch}

            filtered_batch = {
                k: [v for v, keep in zip(batch[k], keep_mask, strict=True) if keep] for k in batch
            }
            filtered_batch["episode_index"] = [
                old_to_new[ep_idx.item()] for ep_idx in filtered_batch["episode_index"]
            ]
            filtered_batch["index"] = list(range(current_global_idx, current_global_idx + sum(keep_mask)))
            current_global_idx += sum(keep_mask)

            return filtered_batch

        return filter_and_map_fn

    return dataset.hf_dataset.map(
        function=create_filter_and_map_fn(),
        batched=True,
        with_indices=True,
    )


def _parse_episodes_list(episodes_str: str) -> list[int]:
    """
    Parse a string of episode indices, ranges, and comma-separated lists into a list of integers.
    """
    episodes = []
    for ep in episodes_str.split(","):
        if "-" in ep:
            start, end = ep.split("-")
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(ep))
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Remove episodes from a LeRobot dataset")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally. By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=str,
        required=True,
        help="Episodes to remove. Can be a single index, comma-separated indices, or ranges (e.g., '1-5,7,10-12')",
    )
    parser.add_argument(
        "-b",
        "--backup",
        nargs="?",
        const=True,
        default=False,
        help="Create a backup before modifying the dataset. Without a value, creates a backup in the default location. "
        "With a value, either 'true'/'false' or a path to store the backup.",
    )
    args = parser.parse_args()

    # Parse the backup argument
    backup_value = args.backup
    if isinstance(backup_value, str):
        if backup_value.lower() == "true":
            backup_value = True
        elif backup_value.lower() == "false":
            backup_value = False
        # Otherwise, it's treated as a path

    # Parse episodes to remove
    episodes_to_remove = _parse_episodes_list(args.episodes)
    if not episodes_to_remove:
        logging.warning("No episodes specified to remove")
        sys.exit(0)

    # Load the dataset
    logging.info(f"Loading dataset '{args.repo_id}'...")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    logging.info(f"Dataset has {dataset.meta.total_episodes} episodes")

    # Modify the dataset
    logging.info(f"Removing {len(set(episodes_to_remove))} episodes: {sorted(set(episodes_to_remove))}")
    updated_dataset = remove_episodes(
        dataset=dataset,
        episodes_to_remove=episodes_to_remove,
        backup=backup_value,
    )
    logging.info(
        f"Successfully removed episodes. Dataset now has {updated_dataset.meta.total_episodes} episodes."
    )


if __name__ == "__main__":
    init_logging()
    main()
