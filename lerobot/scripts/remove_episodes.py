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
import contextlib
import logging
import shutil
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RevisionNotFoundError

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    INFO_PATH,
    TASKS_PATH,
    append_jsonlines,
    create_lerobot_dataset_card,
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

    if not all(ep_idx in dataset.meta.episodes for ep_idx in episodes_to_remove):
        raise ValueError("Episodes to remove must be valid episode indices in the dataset")

    # Calculate the new metadata
    new_meta = deepcopy(dataset.meta)
    new_meta.info["total_episodes"] -= len(episodes_to_remove)
    new_meta.info["total_frames"] -= sum(
        dataset.meta.episodes[ep_idx]["length"] for ep_idx in episodes_to_remove
    )

    for ep_idx in episodes_to_remove:
        new_meta.episodes.pop(ep_idx)
        new_meta.episodes_stats.pop(ep_idx)
    new_meta.stats = aggregate_stats(list(new_meta.episodes_stats.values()))

    tasks = {task for ep in new_meta.episodes.values() if "tasks" in ep for task in ep["tasks"]}
    new_meta.tasks = {new_meta.get_task_index(task): task for task in tasks}
    new_meta.task_to_task_index = {task: idx for idx, task in new_meta.tasks.items()}
    new_meta.info["total_tasks"] = len(new_meta.tasks)

    new_meta.info["total_videos"] = (
        (new_meta.info["total_episodes"]) * len(dataset.meta.video_keys) if dataset.meta.video_keys else 0
    )

    if "splits" in new_meta.info:
        new_meta.info["splits"] = {"train": f"0:{new_meta.info['total_episodes']}"}

    # Now that the metadata is recalculated, we update the dataset files by
    # removing the files related to the specified episodes. We perform a safe
    # update such that if an error occurs, any changes are rolled back and the
    # dataset files are left in its original state. Optionally, a non-temporary
    # full backup can be made so that we also have the dataset in its original state.
    if backup:
        backup_path = (
            Path(backup)
            if isinstance(backup, (str, Path))
            else dataset.root.parent / f"{dataset.root.name}_backup_{int(time.time())}"
        )
        _backup_folder(dataset.root, backup_path)

    _update_dataset_files(
        new_meta,
        episodes_to_remove,
    )

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


def _move_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dest)


def _update_dataset_files(new_meta: LeRobotDatasetMetadata, episodes_to_remove: list[int]):
    """Update dataset files.

    This function performs a safe update for dataset files. It moves modified or removed
    episode files to a temporary directory. Once all changes are made, the temporary
    directory is deleted. If an error occurs during the update, all changes are rolled
    back and the original dataset files are restored.

    Args:
         new_meta (LeRobotDatasetMetadata): Updated metadata object containing the new
              dataset state after removing episodes
         episodes_to_remove (list[int]): List of episode indices to remove from the dataset

    Raises:
         Exception: If any operation fails, rolls back all changes and re-raises the original exception
    """
    with tempfile.TemporaryDirectory(prefix="lerobot_backup_temp_") as backup_path:
        backup_dir = Path(backup_path)

        # Init empty containers s.t. they are guaranteed to exist in the except block
        metadata_files = {}
        rel_data_paths = []
        rel_video_paths = []

        try:
            # Step 1: Update metadata files
            metadata_files = {
                INFO_PATH: lambda: write_info(new_meta.info, new_meta.root),
                EPISODES_PATH: lambda: [
                    write_episode(ep, new_meta.root) for ep in new_meta.episodes.values()
                ],
                TASKS_PATH: lambda: [
                    append_jsonlines({"task_index": idx, "task": task}, new_meta.root / TASKS_PATH)
                    for idx, task in new_meta.tasks.items()
                ],
                EPISODES_STATS_PATH: lambda: [
                    write_episode_stats(idx, stats, new_meta.root)
                    for idx, stats in new_meta.episodes_stats.items()
                ],
            }
            for file_path, update_func in metadata_files.items():
                _move_file(new_meta.root / file_path, backup_dir / file_path)
                update_func()

            # Step 2: Update data and video
            rel_data_paths = [new_meta.get_data_file_path(ep_idx) for ep_idx in episodes_to_remove]
            rel_video_paths = [
                new_meta.get_video_file_path(ep_idx, vid_key)
                for ep_idx in episodes_to_remove
                for vid_key in new_meta.video_keys
            ]
            for rel_path in rel_data_paths + rel_video_paths:
                if (new_meta.root / rel_path).exists():
                    _move_file(new_meta.root / rel_path, backup_dir / rel_path)

        except Exception as e:
            logging.error(f"Error updating dataset files: {str(e)}. Rolling back changes.")

            # Restore metadata files
            for file_path in metadata_files:
                if (backup_dir / file_path).exists():
                    _move_file(backup_dir / file_path, new_meta.root / file_path)

            # Restore data and video files
            for rel_file_path in rel_data_paths + rel_video_paths:
                if (backup_dir / rel_file_path).exists():
                    _move_file(backup_dir / rel_file_path, new_meta.root / rel_file_path)

            raise e


def _backup_folder(target_dir: Path, backup_path: Path) -> None:
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


def _delete_hub_file(hub_api: HfApi, repo_id: str, file_path: str, branch: str):
    try:
        with contextlib.suppress(RevisionNotFoundError):
            if hub_api.file_exists(
                repo_id,
                file_path,
                repo_type="dataset",
                revision=branch,
            ):
                hub_api.delete_file(
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=branch,
                )
                logging.info(f"Deleted '{file_path}' from branch '{branch}'")
    except Exception as e:
        logging.error(f"Error removing file '{file_path}' from the hub: {str(e)}")


def _remove_episodes_from_hub(
    updated_dataset: LeRobotDataset, episodes_to_remove: list[int], branch: str | None = None
):
    """Remove episodes from the hub repository at a specific revision."""
    hub_api = HfApi()

    try:
        for ep_idx in episodes_to_remove:
            data_path = str(updated_dataset.meta.get_data_file_path(ep_idx))
            _delete_hub_file(hub_api, updated_dataset.repo_id, data_path, branch)

            for vid_key in updated_dataset.meta.video_keys:
                video_path = str(updated_dataset.meta.get_video_file_path(ep_idx, vid_key))
                _delete_hub_file(hub_api, updated_dataset.repo_id, video_path, branch)

        logging.info(f"Successfully removed episode files from Hub on branch '{branch}'")

    except RevisionNotFoundError:
        logging.error(f"Branch '{branch}' not found in repository '{updated_dataset.repo_id}'")
    except Exception as e:
        logging.error(f"Error during Hub operations: {str(e)}")


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
    parser.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload to Hugging Face hub.",
    )
    parser.add_argument(
        "--private",
        type=int,
        default=0,
        help="If set, the repository on the Hub will be private",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="List of tags to apply to the dataset on the Hub",
    )
    parser.add_argument("--license", type=str, default=None, help="License to use for the dataset on the Hub")
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

    target_revision_tag = dataset.revision
    push_branch = "main"
    logging.info(
        f"Dataset loaded using revision tag: {target_revision_tag}. Changes will be pushed to 'main' and this tag will be updated."
    )

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

    if args.push_to_hub:
        logging.info("Pushing dataset to hub...")

        updated_dataset.push_to_hub(
            tags=args.tags,
            private=bool(args.private),
            license=args.license,
            branch=push_branch,
            tag_version=False,  # Disable automatic tagging here, we'll do it manually later
        )
        updated_card = create_lerobot_dataset_card(
            tags=args.tags, dataset_info=updated_dataset.meta.info, license=args.license
        )
        updated_card.push_to_hub(repo_id=updated_dataset.repo_id, repo_type="dataset", revision=push_branch)
        _remove_episodes_from_hub(updated_dataset, episodes_to_remove, branch=push_branch)

        logging.info(
            f"Updating tag '{target_revision_tag}' to point to the latest commit on branch '{push_branch}'..."
        )
        hub_api = HfApi()
        try:
            # Delete the old tag first if it exists
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(updated_dataset.repo_id, tag=target_revision_tag, repo_type="dataset")
                logging.info(f"Deleted existing tag '{target_revision_tag}'.")

            # Create the new tag pointing to the head of the push branch
            hub_api.create_tag(
                updated_dataset.repo_id,
                tag=target_revision_tag,
                revision=push_branch,
                repo_type="dataset",
            )
            logging.info(
                f"Successfully created tag '{target_revision_tag}' pointing to branch '{push_branch}'."
            )

        except Exception as e:
            logging.error(f"Error during tag update for '{target_revision_tag}': {str(e)}")

        logging.info("Dataset pushed to hub.")


if __name__ == "__main__":
    init_logging()
    main()
