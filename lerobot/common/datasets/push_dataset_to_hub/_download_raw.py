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
--raw-dir data/lerobot-raw/pusht_raw \
--repo-id lerobot-raw/pusht_raw
```
"""

import argparse
import logging
import warnings
from pathlib import Path

from huggingface_hub import snapshot_download

from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id

# {raw_repo_id: raw_format}
AVAILABLE_RAW_REPO_IDS = {
    "lerobot-raw/aloha_mobile_cabinet_raw": "aloha_hdf5",
    "lerobot-raw/aloha_mobile_chair_raw": "aloha_hdf5",
    "lerobot-raw/aloha_mobile_elevator_raw": "aloha_hdf5",
    "lerobot-raw/aloha_mobile_shrimp_raw": "aloha_hdf5",
    "lerobot-raw/aloha_mobile_wash_pan_raw": "aloha_hdf5",
    "lerobot-raw/aloha_mobile_wipe_wine_raw": "aloha_hdf5",
    "lerobot-raw/aloha_sim_insertion_human_raw": "aloha_hdf5",
    "lerobot-raw/aloha_sim_insertion_scripted_raw": "aloha_hdf5",
    "lerobot-raw/aloha_sim_transfer_cube_human_raw": "aloha_hdf5",
    "lerobot-raw/aloha_sim_transfer_cube_scripted_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_battery_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_candy_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_coffee_new_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_coffee_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_cups_open_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_fork_pick_up_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_pingpong_test_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_pro_pencil_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_screw_driver_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_tape_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_thread_velcro_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_towel_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_vinh_cup_left_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_vinh_cup_raw": "aloha_hdf5",
    "lerobot-raw/aloha_static_ziploc_slide_raw": "aloha_hdf5",
    "lerobot-raw/pusht_raw": "pusht_zarr",
    "lerobot-raw/umi_cup_in_the_wild_raw": "umi_zarr",
    "lerobot-raw/unitreeh1_fold_clothes_raw": "aloha_hdf5",
    "lerobot-raw/unitreeh1_rearrange_objects_raw": "aloha_hdf5",
    "lerobot-raw/unitreeh1_two_robot_greeting_raw": "aloha_hdf5",
    "lerobot-raw/unitreeh1_warehouse_raw": "aloha_hdf5",
    "lerobot-raw/xarm_lift_medium_raw": "xarm_pkl",
    "lerobot-raw/xarm_lift_medium_replay_raw": "xarm_pkl",
    "lerobot-raw/xarm_push_medium_raw": "xarm_pkl",
    "lerobot-raw/xarm_push_medium_replay_raw": "xarm_pkl",
}


def download_raw(raw_dir: Path, repo_id: str):
    check_repo_id(repo_id)
    user_id, dataset_id = repo_id.split("/")

    if not dataset_id.endswith("_raw"):
        warnings.warn(
            f"""`dataset_id` ({dataset_id}) doesn't end with '_raw' (e.g. 'lerobot/pusht_raw'). Following this
             naming convention by renaming your repository is advised, but not mandatory.""",
            stacklevel=1,
        )

    # Send warning if raw_dir isn't well formated
    if raw_dir.parts[-2] != user_id or raw_dir.parts[-1] != dataset_id:
        warnings.warn(
            f"""`raw_dir` ({raw_dir}) doesn't contain a community or user id `/` the name of the dataset that
             match the `repo_id` (e.g. 'data/lerobot/pusht_raw'). Following this naming convention is advised,
             but not mandatory.""",
            stacklevel=1,
        )
    raw_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Start downloading from huggingface.co/{user_id} for {dataset_id}")
    snapshot_download(repo_id, repo_type="dataset", local_dir=raw_dir)
    logging.info(f"Finish downloading from huggingface.co/{user_id} for {dataset_id}")


def download_all_raw_datasets(data_dir: Path | None = None):
    if data_dir is None:
        data_dir = Path("data")
    for repo_id in AVAILABLE_RAW_REPO_IDS:
        raw_dir = data_dir / repo_id
        download_raw(raw_dir, repo_id)


def main():
    parser = argparse.ArgumentParser(
        description=f"""A script to download raw datasets from Hugging Face hub to a local directory. Here is a
            non exhaustive list of available repositories to use in `--repo-id`: {AVAILABLE_RAW_REPO_IDS}""",
    )

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
        help="""Repositery identifier on Hugging Face: a community or a user name `/` the name of
        the dataset (e.g. `lerobot/pusht_raw`, `cadene/aloha_sim_insertion_human_raw`).""",
    )
    args = parser.parse_args()
    download_raw(**vars(args))


if __name__ == "__main__":
    main()
