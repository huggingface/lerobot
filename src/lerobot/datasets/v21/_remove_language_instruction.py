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

import logging
import traceback
from pathlib import Path

from datasets import get_dataset_config_info
from huggingface_hub import HfApi

from lerobot import available_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import INFO_PATH, write_info
from lerobot.datasets.v21.convert_dataset_v20_to_v21 import V20, SuppressWarnings

LOCAL_DIR = Path("data/")

hub_api = HfApi()


def fix_dataset(repo_id: str) -> str:
    if not hub_api.revision_exists(repo_id, V20, repo_type="dataset"):
        return f"{repo_id}: skipped (not in {V20})."

    dataset_info = get_dataset_config_info(repo_id, "default")
    with SuppressWarnings():
        lerobot_metadata = LeRobotDatasetMetadata(repo_id, revision=V20, force_cache_sync=True)

    meta_features = {key for key, ft in lerobot_metadata.features.items() if ft["dtype"] != "video"}
    parquet_features = set(dataset_info.features)

    diff_parquet_meta = parquet_features - meta_features
    diff_meta_parquet = meta_features - parquet_features

    if diff_parquet_meta:
        raise ValueError(f"In parquet not in info.json: {parquet_features - meta_features}")

    if not diff_meta_parquet:
        return f"{repo_id}: skipped (no diff)"

    if diff_meta_parquet:
        logging.warning(f"In info.json not in parquet: {meta_features - parquet_features}")
        assert diff_meta_parquet == {"language_instruction"}
        lerobot_metadata.features.pop("language_instruction")
        write_info(lerobot_metadata.info, lerobot_metadata.root)
        commit_info = hub_api.upload_file(
            path_or_fileobj=lerobot_metadata.root / INFO_PATH,
            path_in_repo=INFO_PATH,
            repo_id=repo_id,
            repo_type="dataset",
            revision=V20,
            commit_message="Remove 'language_instruction'",
            create_pr=True,
        )
        return f"{repo_id}: success - PR: {commit_info.pr_url}"


def batch_fix():
    status = {}
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOCAL_DIR / "fix_features_v20.txt"
    for num, repo_id in enumerate(available_datasets):
        print(f"\nConverting {repo_id} ({num}/{len(available_datasets)})")
        print("---------------------------------------------------------")
        try:
            status = fix_dataset(repo_id)
        except Exception:
            status = f"{repo_id}: failed\n    {traceback.format_exc()}"

        logging.info(status)
        with open(logfile, "a") as file:
            file.write(status + "\n")


if __name__ == "__main__":
    batch_fix()
