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
This script converts a LeRobot dataset already pushed to the Hub from codebase version 2.0 to 2.1.
It downloads metadata from a SOURCE dataset repo, computes/validates per-episode stats, updates
the codebase version in `info.json`, and uploads the result to a DESTINATION dataset repo.
It will:

- Generate per-episodes stats and writes them in `episodes_stats.jsonl`
- Check consistency between these new stats and the old ones.
- Remove the deprecated `stats.json`.
- Update codebase_version in `info.json`.
- Push this new version to the destination repo/branch and tag it with the current codebase version.

Usage:

```bash
python -m lerobot.datasets.v21.convert_dataset_v20_to_v21 \
    --source-repo-id=namespace/source_dataset \
    --dest-repo-id=namespace/destination_dataset \
    --branch=main
```

"""

import argparse
import logging

from huggingface_hub import HfApi

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info
from lerobot.datasets.v21.convert_stats import check_aggregate_stats, convert_stats

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
    dest_repo_id: str,
    branch: str | None = None,
    num_workers: int = 4,
):
    # Download metadata from the source repo at v2.0
    with SuppressWarnings():
        dataset = LeRobotDataset(source_repo_id, revision=V20, force_cache_sync=True)

    # Ensure we recompute fresh episodes stats
    if (dataset.root / EPISODES_STATS_PATH).is_file():
        (dataset.root / EPISODES_STATS_PATH).unlink()

    # Compute and validate stats
    convert_stats(dataset, num_workers=num_workers)
    ref_stats = load_stats(dataset.root)
    check_aggregate_stats(dataset, ref_stats)

    # Update codebase version in info.json
    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(dataset.meta.info, dataset.root)

    # Remove deprecated stats.json locally so it won't be uploaded
    if (dataset.root / STATS_PATH).is_file():
        (dataset.root / STATS_PATH).unlink()

    # Push only meta/ to destination repo
    hub_api = HfApi()
    hub_api.create_repo(repo_id=dest_repo_id, private=False, repo_type="dataset", exist_ok=True)
    if branch:
        hub_api.create_branch(repo_id=dest_repo_id, branch=branch, repo_type="dataset", exist_ok=True)

    hub_api.upload_folder(
        repo_id=dest_repo_id,
        folder_path=str(dataset.root),
        repo_type="dataset",
        revision=branch,
        allow_patterns="meta/",
    )

    # Ensure old stats.json is deleted on destination
    if hub_api.file_exists(repo_id=dest_repo_id, filename=STATS_PATH, revision=branch, repo_type="dataset"):
        hub_api.delete_file(path_in_repo=STATS_PATH, repo_id=dest_repo_id, revision=branch, repo_type="dataset")

    # Tag destination with current codebase version
    hub_api.create_tag(dest_repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-repo-id",
        type=str,
        required=True,
        help="Source dataset repo id to download from (must be v2.0).",
    )
    parser.add_argument(
        "--dest-repo-id",
        type=str,
        required=True,
        help="Destination dataset repo id to upload the converted metadata to.",
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

    args = parser.parse_args()
    convert_dataset(**vars(args))
