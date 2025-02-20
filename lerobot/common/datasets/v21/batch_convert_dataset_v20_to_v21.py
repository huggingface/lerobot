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

"""
This script is for internal use to convert all datasets under the 'lerobot' hub user account to v2.1.
"""

import traceback
from pathlib import Path

from huggingface_hub import HfApi

from lerobot import available_datasets
from lerobot.common.datasets.v21.convert_dataset_v20_to_v21 import V21, convert_dataset

LOCAL_DIR = Path("data/")


def batch_convert():
    status = {}
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOCAL_DIR / "conversion_log_v21.txt"
    hub_api = HfApi()
    for num, repo_id in enumerate(available_datasets):
        print(f"\nConverting {repo_id} ({num}/{len(available_datasets)})")
        print("---------------------------------------------------------")
        try:
            if hub_api.revision_exists(repo_id, V21, repo_type="dataset"):
                status = f"{repo_id}: success (already in {V21})."
            else:
                convert_dataset(repo_id)
                status = f"{repo_id}: success."
        except Exception:
            status = f"{repo_id}: failed\n    {traceback.format_exc()}"

        with open(logfile, "a") as file:
            file.write(status + "\n")


if __name__ == "__main__":
    batch_convert()
