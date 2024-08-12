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
    "lerobot-raw/fractal20220817_data": "openx_rlds.fractal20220817_data",
    "lerobot-raw/kuka": "openx_rlds.kuka",
    "lerobot-raw/bridge_openx": "openx_rlds.bridge_openx",
    "lerobot-raw/taco_play": "openx_rlds.taco_play",
    "lerobot-raw/jaco_play": "openx_rlds.jaco_play",
    "lerobot-raw/berkeley_cable_routing": "openx_rlds.berkeley_cable_routing",
    "lerobot-raw/roboturk": "openx_rlds.roboturk",
    "lerobot-raw/nyu_door_opening_surprising_effectiveness": "openx_rlds.nyu_door_opening_surprising_effectiveness",
    "lerobot-raw/viola": "openx_rlds.viola",
    "lerobot-raw/berkeley_autolab_ur5": "openx_rlds.berkeley_autolab_ur5",
    "lerobot-raw/toto": "openx_rlds.toto",
    "lerobot-raw/language_table": "openx_rlds.language_table",
    "lerobot-raw/columbia_cairlab_pusht_real": "openx_rlds.columbia_cairlab_pusht_real",
    "lerobot-raw/stanford_kuka_multimodal_dataset_converted_externally_to_rlds": "openx_rlds.stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
    "lerobot-raw/nyu_rot_dataset_converted_externally_to_rlds": "openx_rlds.nyu_rot_dataset_converted_externally_to_rlds",
    "lerobot-raw/io_ai_tech": "openx_rlds.io_ai_tech",
    "lerobot-raw/stanford_hydra_dataset_converted_externally_to_rlds": "openx_rlds.stanford_hydra_dataset_converted_externally_to_rlds",
    "lerobot-raw/austin_buds_dataset_converted_externally_to_rlds": "openx_rlds.austin_buds_dataset_converted_externally_to_rlds",
    "lerobot-raw/nyu_franka_play_dataset_converted_externally_to_rlds": "openx_rlds.nyu_franka_play_dataset_converted_externally_to_rlds",
    "lerobot-raw/maniskill_dataset_converted_externally_to_rlds": "openx_rlds.maniskill_dataset_converted_externally_to_rlds",
    "lerobot-raw/furniture_bench_dataset_converted_externally_to_rlds": "openx_rlds.furniture_bench_dataset_converted_externally_to_rlds",
    "lerobot-raw/cmu_franka_exploration_dataset_converted_externally_to_rlds": "openx_rlds.cmu_franka_exploration_dataset_converted_externally_to_rlds",
    "lerobot-raw/ucsd_kitchen_dataset_converted_externally_to_rlds": "openx_rlds.ucsd_kitchen_dataset_converted_externally_to_rlds",
    "lerobot-raw/ucsd_pick_and_place_dataset_converted_externally_to_rlds": "openx_rlds.ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "lerobot-raw/spoc": "openx_rlds.spoc",
    "lerobot-raw/austin_sailor_dataset_converted_externally_to_rlds": "openx_rlds.austin_sailor_dataset_converted_externally_to_rlds",
    "lerobot-raw/austin_sirius_dataset_converted_externally_to_rlds": "openx_rlds.austin_sirius_dataset_converted_externally_to_rlds",
    "lerobot-raw/bc_z": "openx_rlds.bc_z",
    "lerobot-raw/utokyo_pr2_opening_fridge_converted_externally_to_rlds": "openx_rlds.utokyo_pr2_opening_fridge_converted_externally_to_rlds",
    "lerobot-raw/utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": "openx_rlds.utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "lerobot-raw/utokyo_xarm_pick_and_place_converted_externally_to_rlds": "openx_rlds.utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    "lerobot-raw/utokyo_xarm_bimanual_converted_externally_to_rlds": "openx_rlds.utokyo_xarm_bimanual_converted_externally_to_rlds",
    "lerobot-raw/robo_net": "openx_rlds.robo_net",
    "lerobot-raw/robo_set": "openx_rlds.robo_set",
    "lerobot-raw/berkeley_mvp_converted_externally_to_rlds": "openx_rlds.berkeley_mvp_converted_externally_to_rlds",
    "lerobot-raw/berkeley_rpt_converted_externally_to_rlds": "openx_rlds.berkeley_rpt_converted_externally_to_rlds",
    "lerobot-raw/kaist_nonprehensile_converted_externally_to_rlds": "openx_rlds.kaist_nonprehensile_converted_externally_to_rlds",
    "lerobot-raw/stanford_mask_vit_converted_externally_to_rlds": "openx_rlds.stanford_mask_vit_converted_externally_to_rlds",
    "lerobot-raw/tokyo_u_lsmo_converted_externally_to_rlds": "openx_rlds.tokyo_u_lsmo_converted_externally_to_rlds",
    "lerobot-raw/dlr_sara_pour_converted_externally_to_rlds": "openx_rlds.dlr_sara_pour_converted_externally_to_rlds",
    "lerobot-raw/dlr_sara_grid_clamp_converted_externally_to_rlds": "openx_rlds.dlr_sara_grid_clamp_converted_externally_to_rlds",
    "lerobot-raw/dlr_edan_shared_control_converted_externally_to_rlds": "openx_rlds.dlr_edan_shared_control_converted_externally_to_rlds",
    "lerobot-raw/asu_table_top_converted_externally_to_rlds": "openx_rlds.asu_table_top_converted_externally_to_rlds",
    "lerobot-raw/stanford_robocook_converted_externally_to_rlds": "openx_rlds.stanford_robocook_converted_externally_to_rlds",
    "lerobot-raw/imperialcollege_sawyer_wrist_cam": "openx_rlds.imperialcollege_sawyer_wrist_cam",
    "lerobot-raw/iamlab_cmu_pickup_insert_converted_externally_to_rlds": "openx_rlds.iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "lerobot-raw/uiuc_d3field": "openx_rlds.uiuc_d3field",
    "lerobot-raw/utaustin_mutex": "openx_rlds.utaustin_mutex",
    "lerobot-raw/berkeley_fanuc_manipulation": "openx_rlds.berkeley_fanuc_manipulation",
    "lerobot-raw/cmu_playing_with_food": "openx_rlds.cmu_playing_with_food",
    "lerobot-raw/cmu_play_fusion": "openx_rlds.cmu_play_fusion",
    "lerobot-raw/cmu_stretch": "openx_rlds.cmu_stretch",
    "lerobot-raw/berkeley_gnm_recon": "openx_rlds.berkeley_gnm_recon",
    "lerobot-raw/berkeley_gnm_cory_hall": "openx_rlds.berkeley_gnm_cory_hall",
    "lerobot-raw/berkeley_gnm_sac_son": "openx_rlds.berkeley_gnm_sac_son",
    "lerobot-raw/droid": "openx_rlds.droid",
    "lerobot-raw/droid100": "openx_rlds.droid100",
    "lerobot-raw/fmb": "openx_rlds.fmb",
    "lerobot-raw/dobbe": "openx_rlds.dobbe",
    "lerobot-raw/usc_cloth_sim_converted_externally_to_rlds": "openx_rlds.usc_cloth_sim_converted_externally_to_rlds",
    "lerobot-raw/plex_robosuite": "openx_rlds.plex_robosuite",
    "lerobot-raw/conq_hose_manipulation": "openx_rlds.conq_hose_manipulation",
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
