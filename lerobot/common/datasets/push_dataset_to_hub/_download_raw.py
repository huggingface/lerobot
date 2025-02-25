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
    "lerobot-raw/umi_cup_in_the_wild_raw": "umi_zarr",
    "lerobot-raw/pusht_raw": "pusht_zarr",
    "lerobot-raw/unitreeh1_fold_clothes_raw": "aloha_hdf5",
    "lerobot-raw/unitreeh1_rearrange_objects_raw": "aloha_hdf5",
    "lerobot-raw/unitreeh1_two_robot_greeting_raw": "aloha_hdf5",
    "lerobot-raw/unitreeh1_warehouse_raw": "aloha_hdf5",
    "lerobot-raw/xarm_lift_medium_raw": "xarm_pkl",
    "lerobot-raw/xarm_lift_medium_replay_raw": "xarm_pkl",
    "lerobot-raw/xarm_push_medium_raw": "xarm_pkl",
    "lerobot-raw/xarm_push_medium_replay_raw": "xarm_pkl",
    "lerobot-raw/fractal20220817_data_raw": "openx_rlds.fractal20220817_data",
    "lerobot-raw/kuka_raw": "openx_rlds.kuka",
    "lerobot-raw/bridge_openx_raw": "openx_rlds.bridge_openx",
    "lerobot-raw/taco_play_raw": "openx_rlds.taco_play",
    "lerobot-raw/jaco_play_raw": "openx_rlds.jaco_play",
    "lerobot-raw/berkeley_cable_routing_raw": "openx_rlds.berkeley_cable_routing",
    "lerobot-raw/roboturk_raw": "openx_rlds.roboturk",
    "lerobot-raw/nyu_door_opening_surprising_effectiveness_raw": "openx_rlds.nyu_door_opening_surprising_effectiveness",
    "lerobot-raw/viola_raw": "openx_rlds.viola",
    "lerobot-raw/berkeley_autolab_ur5_raw": "openx_rlds.berkeley_autolab_ur5",
    "lerobot-raw/toto_raw": "openx_rlds.toto",
    "lerobot-raw/language_table_raw": "openx_rlds.language_table",
    "lerobot-raw/columbia_cairlab_pusht_real_raw": "openx_rlds.columbia_cairlab_pusht_real",
    "lerobot-raw/stanford_kuka_multimodal_dataset_raw": "openx_rlds.stanford_kuka_multimodal_dataset",
    "lerobot-raw/nyu_rot_dataset_raw": "openx_rlds.nyu_rot_dataset",
    "lerobot-raw/io_ai_tech_raw": "openx_rlds.io_ai_tech",
    "lerobot-raw/stanford_hydra_dataset_raw": "openx_rlds.stanford_hydra_dataset",
    "lerobot-raw/austin_buds_dataset_raw": "openx_rlds.austin_buds_dataset",
    "lerobot-raw/nyu_franka_play_dataset_raw": "openx_rlds.nyu_franka_play_dataset",
    "lerobot-raw/maniskill_dataset_raw": "openx_rlds.maniskill_dataset",
    "lerobot-raw/furniture_bench_dataset_raw": "openx_rlds.furniture_bench_dataset",
    "lerobot-raw/cmu_franka_exploration_dataset_raw": "openx_rlds.cmu_franka_exploration_dataset",
    "lerobot-raw/ucsd_kitchen_dataset_raw": "openx_rlds.ucsd_kitchen_dataset",
    "lerobot-raw/ucsd_pick_and_place_dataset_raw": "openx_rlds.ucsd_pick_and_place_dataset",
    "lerobot-raw/spoc_raw": "openx_rlds.spoc",
    "lerobot-raw/austin_sailor_dataset_raw": "openx_rlds.austin_sailor_dataset",
    "lerobot-raw/austin_sirius_dataset_raw": "openx_rlds.austin_sirius_dataset",
    "lerobot-raw/bc_z_raw": "openx_rlds.bc_z",
    "lerobot-raw/utokyo_pr2_opening_fridge_raw": "openx_rlds.utokyo_pr2_opening_fridge",
    "lerobot-raw/utokyo_pr2_tabletop_manipulation_raw": "openx_rlds.utokyo_pr2_tabletop_manipulation",
    "lerobot-raw/utokyo_xarm_pick_and_place_raw": "openx_rlds.utokyo_xarm_pick_and_place",
    "lerobot-raw/utokyo_xarm_bimanual_raw": "openx_rlds.utokyo_xarm_bimanual",
    "lerobot-raw/utokyo_saytap_raw": "openx_rlds.utokyo_saytap",
    "lerobot-raw/robo_net_raw": "openx_rlds.robo_net",
    "lerobot-raw/robo_set_raw": "openx_rlds.robo_set",
    "lerobot-raw/berkeley_mvp_raw": "openx_rlds.berkeley_mvp",
    "lerobot-raw/berkeley_rpt_raw": "openx_rlds.berkeley_rpt",
    "lerobot-raw/kaist_nonprehensile_raw": "openx_rlds.kaist_nonprehensile",
    "lerobot-raw/stanford_mask_vit_raw": "openx_rlds.stanford_mask_vit",
    "lerobot-raw/tokyo_u_lsmo_raw": "openx_rlds.tokyo_u_lsmo",
    "lerobot-raw/dlr_sara_pour_raw": "openx_rlds.dlr_sara_pour",
    "lerobot-raw/dlr_sara_grid_clamp_raw": "openx_rlds.dlr_sara_grid_clamp",
    "lerobot-raw/dlr_edan_shared_control_raw": "openx_rlds.dlr_edan_shared_control",
    "lerobot-raw/asu_table_top_raw": "openx_rlds.asu_table_top",
    "lerobot-raw/stanford_robocook_raw": "openx_rlds.stanford_robocook",
    "lerobot-raw/imperialcollege_sawyer_wrist_cam_raw": "openx_rlds.imperialcollege_sawyer_wrist_cam",
    "lerobot-raw/iamlab_cmu_pickup_insert_raw": "openx_rlds.iamlab_cmu_pickup_insert",
    "lerobot-raw/uiuc_d3field_raw": "openx_rlds.uiuc_d3field",
    "lerobot-raw/utaustin_mutex_raw": "openx_rlds.utaustin_mutex",
    "lerobot-raw/berkeley_fanuc_manipulation_raw": "openx_rlds.berkeley_fanuc_manipulation",
    "lerobot-raw/cmu_playing_with_food_raw": "openx_rlds.cmu_playing_with_food",
    "lerobot-raw/cmu_play_fusion_raw": "openx_rlds.cmu_play_fusion",
    "lerobot-raw/cmu_stretch_raw": "openx_rlds.cmu_stretch",
    "lerobot-raw/berkeley_gnm_recon_raw": "openx_rlds.berkeley_gnm_recon",
    "lerobot-raw/berkeley_gnm_cory_hall_raw": "openx_rlds.berkeley_gnm_cory_hall",
    "lerobot-raw/berkeley_gnm_sac_son_raw": "openx_rlds.berkeley_gnm_sac_son",
    "lerobot-raw/droid_raw": "openx_rlds.droid",
    "lerobot-raw/droid_100_raw": "openx_rlds.droid100",
    "lerobot-raw/fmb_raw": "openx_rlds.fmb",
    "lerobot-raw/dobbe_raw": "openx_rlds.dobbe",
    "lerobot-raw/usc_cloth_sim_raw": "openx_rlds.usc_cloth_sim",
    "lerobot-raw/plex_robosuite_raw": "openx_rlds.plex_robosuite",
    "lerobot-raw/conq_hose_manipulation_raw": "openx_rlds.conq_hose_manipulation",
    "lerobot-raw/vima_raw": "openx_rlds.vima",
    "lerobot-raw/robot_vqa_raw": "openx_rlds.robot_vqa",
    "lerobot-raw/mimic_play_raw": "openx_rlds.mimic_play",
    "lerobot-raw/tidybot_raw": "openx_rlds.tidybot",
    "lerobot-raw/eth_agent_affordances_raw": "openx_rlds.eth_agent_affordances",
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

    # Send warning if raw_dir isn't well formatted
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
            non exhaustive list of available repositories to use in `--repo-id`: {list(AVAILABLE_RAW_REPO_IDS.keys())}""",
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
