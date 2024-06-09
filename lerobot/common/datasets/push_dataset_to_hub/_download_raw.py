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
"""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download


def download_raw(raw_dir: Path, dataset_id: str):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Start downloading from huggingface.co/cadene for {dataset_id}")
    snapshot_download(f"cadene/{dataset_id}_raw", repo_type="dataset", local_dir=raw_dir)
    logging.info(f"Finish downloading from huggingface.co/cadene for {dataset_id}")


if __name__ == "__main__":
    data_dir = Path("data")
    dataset_ids = [
        "pusht_image",
        "xarm_lift_medium_image",
        "xarm_lift_medium_replay_image",
        "xarm_push_medium_image",
        "xarm_push_medium_replay_image",
        "aloha_sim_insertion_human_image",
        "aloha_sim_insertion_scripted_image",
        "aloha_sim_transfer_cube_human_image",
        "aloha_sim_transfer_cube_scripted_image",
        "pusht",
        "xarm_lift_medium",
        "xarm_lift_medium_replay",
        "xarm_push_medium",
        "xarm_push_medium_replay",
        "aloha_sim_insertion_human",
        "aloha_sim_insertion_scripted",
        "aloha_sim_transfer_cube_human",
        "aloha_sim_transfer_cube_scripted",
        "aloha_mobile_cabinet",
        "aloha_mobile_chair",
        "aloha_mobile_elevator",
        "aloha_mobile_shrimp",
        "aloha_mobile_wash_pan",
        "aloha_mobile_wipe_wine",
        "aloha_static_battery",
        "aloha_static_candy",
        "aloha_static_coffee",
        "aloha_static_coffee_new",
        "aloha_static_cups_open",
        "aloha_static_fork_pick_up",
        "aloha_static_pingpong_test",
        "aloha_static_pro_pencil",
        "aloha_static_screw_driver",
        "aloha_static_tape",
        "aloha_static_thread_velcro",
        "aloha_static_towel",
        "aloha_static_vinh_cup",
        "aloha_static_vinh_cup_left",
        "aloha_static_ziploc_slide",
        "umi_cup_in_the_wild",
    ]
    for dataset_id in dataset_ids:
        raw_dir = data_dir / f"{dataset_id}_raw"
        download_raw(raw_dir, dataset_id)
