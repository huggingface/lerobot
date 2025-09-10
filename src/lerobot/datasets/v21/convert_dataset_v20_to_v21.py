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

"""
This script will help you download any LeRobot dataset from the hub, convert it to the latest format, and
upload it to your own repository. It will:

- Download the dataset from any source repository
- Generate per-episodes stats and writes them in `episodes_stats.jsonl`
- Update codebase_version in `info.json` to the latest version
- Create proper version tags
- Push the converted dataset to your specified destination repository

Usage:

```bash
python -m lerobot.datasets.v21.convert_dataset_v20_to_v21 \
    --source-repo-id=IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --dest-repo-id=your-username/libero_spatial_converted \
    --episodes=0,1,2,3,4
```

"""

import argparse
import logging

from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, write_info
from lerobot.datasets.v21.convert_stats import convert_stats

V20 = "v2.0"
V21 = "v2.1"


class SuppressWarnings:
    def __enter__(self):
        self.previous_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.previous_level)


def convert_dataset(
    source_repo_id: str,
    dest_repo_id: str | None = None,
    episodes: str | None = None,
    branch: str | None = None,
    num_workers: int = 4,
    force_cache_sync: bool = True,
):
    """
    Download a dataset from source_repo_id, convert it, and upload to dest_repo_id.

    Args:
        source_repo_id: Source repository to download from
        dest_repo_id: Destination repository to upload to (defaults to source_repo_id)
        episodes: Comma-separated list of episode indices to include (e.g. "0,1,2,3")
        branch: Branch to upload to
        num_workers: Number of workers for stats computation
        force_cache_sync: Whether to force cache synchronization
    """
    if dest_repo_id is None:
        dest_repo_id = source_repo_id

    # Parse episodes list if provided
    episode_list = None
    if episodes:
        try:
            episode_list = [int(ep.strip()) for ep in episodes.split(",")]
            print(f"Loading episodes: {episode_list}")
        except ValueError as e:
            raise ValueError(
                f"Invalid episodes format '{episodes}'. Use comma-separated integers like '0,1,2,3'"
            ) from e

    print(f"Downloading dataset from: {source_repo_id}")

    # Try to load the dataset with different approaches to handle versioning issues
    dataset = None
    load_attempts = [
        {"revision": None},  # Try latest first
        {"revision": V20},  # Try v2.0
        {"revision": "main"},  # Try main branch
    ]

    for attempt in load_attempts:
        try:
            print(f"Attempting to load with revision: {attempt['revision']}")
            with SuppressWarnings():
                dataset = LeRobotDataset(
                    source_repo_id, episodes=episode_list, force_cache_sync=force_cache_sync, **attempt
                )
            print("Successfully loaded dataset!")
            break
        except Exception as e:
            print(f"Failed with revision {attempt['revision']}: {e}")
            continue

    if dataset is None:
        raise RuntimeError(f"Could not load dataset {source_repo_id} with any revision")

    # Clean up old stats if present
    if (dataset.root / EPISODES_STATS_PATH).is_file():
        (dataset.root / EPISODES_STATS_PATH).unlink()
        print("Removed existing episodes_stats.jsonl")

    print("Converting stats to new format...")
    convert_stats(dataset, num_workers=num_workers)

    # Update dataset info
    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(dataset.meta.info, dataset.root)
    print(f"Updated codebase_version to {CODEBASE_VERSION}")

    # Change repo_id for destination if different
    if dest_repo_id != source_repo_id:
        print(f"Changing repository from {source_repo_id} to {dest_repo_id}")
        dataset.repo_id = dest_repo_id

    print(f"Pushing converted dataset to: {dest_repo_id}")
    dataset.push_to_hub(branch=branch, tag_version=False)

    # Clean up old stats.json file locally and on hub
    if (dataset.root / STATS_PATH).is_file():
        (dataset.root / STATS_PATH).unlink()
        print("Removed local stats.json file")

    hub_api = HfApi()
    try:
        if hub_api.file_exists(
            repo_id=dest_repo_id, filename=STATS_PATH, revision=branch, repo_type="dataset"
        ):
            hub_api.delete_file(
                path_in_repo=STATS_PATH, repo_id=dest_repo_id, revision=branch, repo_type="dataset"
            )
            print("Removed stats.json from hub")
    except Exception as e:
        print(f"Warning: Could not remove stats.json from hub: {e}")

    # Create version tag
    try:
        hub_api.create_tag(dest_repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")
        print(f"Created tag {CODEBASE_VERSION} for {dest_repo_id}")
    except Exception as e:
        print(f"Warning: Could not create tag: {e}")

    print(f"✅ Successfully converted and uploaded dataset to {dest_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, convert, and re-upload LeRobot datasets with proper versioning"
    )
    parser.add_argument(
        "--source-repo-id",
        type=str,
        required=True,
        help="Source repository identifier to download from (e.g. 'IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot')",
    )
    parser.add_argument(
        "--dest-repo-id",
        type=str,
        default=None,
        help="Destination repository identifier to upload to. Defaults to source-repo-id if not specified.",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated list of episode indices to include (e.g. '0,1,2,3,4'). If not specified, all episodes are included.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing stats compute. Defaults to 4.",
    )
    parser.add_argument(
        "--no-cache-sync",
        action="store_true",
        help="Skip forcing cache synchronization (faster but may use cached data)",
    )

    args = parser.parse_args()

    # Convert args to match function signature
    convert_args = {
        "source_repo_id": args.source_repo_id,
        "dest_repo_id": args.dest_repo_id,
        "episodes": args.episodes,
        "branch": args.branch,
        "num_workers": args.num_workers,
        "force_cache_sync": not args.no_cache_sync,
    }

    convert_dataset(**convert_args)
