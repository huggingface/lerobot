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
This file contains download scripts for raw datasets.

Example of usage:
```
python lerobot/common/datasets/push_dataset_to_hub/_download_raw.py \
--raw-dir data/cadene/pusht_raw \
--repo-id cadene/pusht_raw
```
"""

import argparse
import logging
import warnings
from pathlib import Path

from huggingface_hub import snapshot_download


def download_raw(raw_dir: Path, repo_id: str):
    # Check repo_id is well formated
    if len(repo_id.split("/")) != 2:
        raise ValueError(
            f"`repo_id` is expected to contain a community or user id `/` the name of the dataset (e.g. 'lerobot/pusht'), but contains '{repo_id}'."
        )
    user_id, dataset_id = repo_id.split("/")

    if not dataset_id.endswith("_raw"):
        warnings.warn(
            f"`dataset_id` ({dataset_id}) doesn't end with '_raw' (e.g. 'lerobot/pusht_raw'). Following this naming convention by renaming your repository is advised, but not mandatory.",
            stacklevel=1,
        )

    raw_dir = Path(raw_dir)
    # Send warning if raw_dir isn't well formated
    if raw_dir.parts[-2] != user_id or raw_dir.parts[-1] != dataset_id:
        warnings.warn(
            f"`raw_dir` ({raw_dir}) doesn't contain a community or user id `/` the name of the dataset that match the `repo_id` (e.g. 'data/lerobot/pusht_raw'). Following this naming convention is advised, but not mandatory.",
            stacklevel=1,
        )
    raw_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Start downloading from huggingface.co/{user_id} for {dataset_id}")
    snapshot_download(f"{repo_id}", repo_type="dataset", local_dir=raw_dir)
    logging.info(f"Finish downloading from huggingface.co/{user_id} for {dataset_id}")


def download_all_raw_datasets():
    data_dir = Path("data")
    repo_ids = [
        "cadene/pusht_image_raw",
        "cadene/xarm_lift_medium_image_raw",
        "cadene/xarm_lift_medium_replay_image_raw",
        "cadene/xarm_push_medium_image_raw",
        "cadene/xarm_push_medium_replay_image_raw",
        "cadene/aloha_sim_insertion_human_image_raw",
        "cadene/aloha_sim_insertion_scripted_image_raw",
        "cadene/aloha_sim_transfer_cube_human_image_raw",
        "cadene/aloha_sim_transfer_cube_scripted_image_raw",
        "cadene/pusht_raw",
        "cadene/xarm_lift_medium_raw",
        "cadene/xarm_lift_medium_replay_raw",
        "cadene/xarm_push_medium_raw",
        "cadene/xarm_push_medium_replay_raw",
        "cadene/aloha_sim_insertion_human_raw",
        "cadene/aloha_sim_insertion_scripted_raw",
        "cadene/aloha_sim_transfer_cube_human_raw",
        "cadene/aloha_sim_transfer_cube_scripted_raw",
        "cadene/aloha_mobile_cabinet_raw",
        "cadene/aloha_mobile_chair_raw",
        "cadene/aloha_mobile_elevator_raw",
        "cadene/aloha_mobile_shrimp_raw",
        "cadene/aloha_mobile_wash_pan_raw",
        "cadene/aloha_mobile_wipe_wine_raw",
        "cadene/aloha_static_battery_raw",
        "cadene/aloha_static_candy_raw",
        "cadene/aloha_static_coffee_raw",
        "cadene/aloha_static_coffee_new_raw",
        "cadene/aloha_static_cups_open_raw",
        "cadene/aloha_static_fork_pick_up_raw",
        "cadene/aloha_static_pingpong_test_raw",
        "cadene/aloha_static_pro_pencil_raw",
        "cadene/aloha_static_screw_driver_raw",
        "cadene/aloha_static_tape_raw",
        "cadene/aloha_static_thread_velcro_raw",
        "cadene/aloha_static_towel_raw",
        "cadene/aloha_static_vinh_cup_raw",
        "cadene/aloha_static_vinh_cup_left_raw",
        "cadene/aloha_static_ziploc_slide_raw",
        "cadene/umi_cup_in_the_wild_raw",
    ]
    for repo_id in repo_ids:
        raw_dir = data_dir / repo_id
        download_raw(raw_dir, repo_id)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input raw datasets (e.g. `data/aloha_mobile_chair_raw` or `data/pusht_raw).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset (e.g. `lerobot/pusht_raw`, `cadene/aloha_sim_insertion_human_raw`).",
    )
    args = parser.parse_args()
    download_raw(**vars(args))


if __name__ == "__main__":
    main()
