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
This file contains lists of available environments, dataset and policies to reflect the current state of LeRobot library.
We do not want to import all the dependencies, but instead we keep it lightweight to ensure fast access to these variables.

Example:
    ```python
        import lerobot
        print(lerobot.available_envs)
        print(lerobot.available_tasks_per_env)
        print(lerobot.available_datasets)
        print(lerobot.available_datasets_per_env)
        print(lerobot.available_real_world_datasets)
        print(lerobot.available_policies)
        print(lerobot.available_policies_per_env)
        print(lerobot.available_robots)
        print(lerobot.available_cameras)
        print(lerobot.available_motors)
    ```

When implementing a new dataset loadable with LeRobotDataset follow these steps:
- Update `available_datasets_per_env` in `lerobot/__init__.py`

When implementing a new environment (e.g. `gym_aloha`), follow these steps:
- Update `available_tasks_per_env` and `available_datasets_per_env` in `lerobot/__init__.py`

When implementing a new policy class (e.g. `DiffusionPolicy`) follow these steps:
- Update `available_policies` and `available_policies_per_env`, in `lerobot/__init__.py`
- Set the required `name` class attribute.
- Update variables in `tests/test_available.py` by importing your new Policy class
"""

import itertools

from lerobot.__version__ import __version__  # noqa: F401

# TODO(rcadene): Improve policies and envs. As of now, an item in `available_policies`
# refers to a yaml file AND a modeling name. Same for `available_envs` which refers to
# a yaml file AND a environment name. The difference should be more obvious.
available_tasks_per_env = {
    "aloha": [
        "AlohaInsertion-v0",
        "AlohaTransferCube-v0",
    ],
    "pusht": ["PushT-v0"],
    "xarm": ["XarmLift-v0"],
}
available_envs = list(available_tasks_per_env.keys())

available_datasets_per_env = {
    "aloha": [
        "lerobot/aloha_sim_insertion_human",
        "lerobot/aloha_sim_insertion_scripted",
        "lerobot/aloha_sim_transfer_cube_human",
        "lerobot/aloha_sim_transfer_cube_scripted",
        "lerobot/aloha_sim_insertion_human_image",
        "lerobot/aloha_sim_insertion_scripted_image",
        "lerobot/aloha_sim_transfer_cube_human_image",
        "lerobot/aloha_sim_transfer_cube_scripted_image",
    ],
    # TODO(alexander-soare): Add "lerobot/pusht_keypoints". Right now we can't because this is too tightly
    # coupled with tests.
    "pusht": ["lerobot/pusht", "lerobot/pusht_image"],
    "xarm": [
        "lerobot/xarm_lift_medium",
        "lerobot/xarm_lift_medium_replay",
        "lerobot/xarm_push_medium",
        "lerobot/xarm_push_medium_replay",
        "lerobot/xarm_lift_medium_image",
        "lerobot/xarm_lift_medium_replay_image",
        "lerobot/xarm_push_medium_image",
        "lerobot/xarm_push_medium_replay_image",
    ],
}

available_real_world_datasets = [
    "lerobot/aloha_mobile_cabinet",
    "lerobot/aloha_mobile_chair",
    "lerobot/aloha_mobile_elevator",
    "lerobot/aloha_mobile_shrimp",
    "lerobot/aloha_mobile_wash_pan",
    "lerobot/aloha_mobile_wipe_wine",
    "lerobot/aloha_static_battery",
    "lerobot/aloha_static_candy",
    "lerobot/aloha_static_coffee",
    "lerobot/aloha_static_coffee_new",
    "lerobot/aloha_static_cups_open",
    "lerobot/aloha_static_fork_pick_up",
    "lerobot/aloha_static_pingpong_test",
    "lerobot/aloha_static_pro_pencil",
    "lerobot/aloha_static_screw_driver",
    "lerobot/aloha_static_tape",
    "lerobot/aloha_static_thread_velcro",
    "lerobot/aloha_static_towel",
    "lerobot/aloha_static_vinh_cup",
    "lerobot/aloha_static_vinh_cup_left",
    "lerobot/aloha_static_ziploc_slide",
    "lerobot/umi_cup_in_the_wild",
    "lerobot/unitreeh1_fold_clothes",
    "lerobot/unitreeh1_rearrange_objects",
    "lerobot/unitreeh1_two_robot_greeting",
    "lerobot/unitreeh1_warehouse",
    "lerobot/nyu_rot_dataset",
    "lerobot/utokyo_saytap",
    "lerobot/imperialcollege_sawyer_wrist_cam",
    "lerobot/utokyo_xarm_bimanual",
    "lerobot/tokyo_u_lsmo",
    "lerobot/utokyo_pr2_opening_fridge",
    "lerobot/cmu_franka_exploration_dataset",
    "lerobot/cmu_stretch",
    "lerobot/asu_table_top",
    "lerobot/utokyo_pr2_tabletop_manipulation",
    "lerobot/utokyo_xarm_pick_and_place",
    "lerobot/ucsd_kitchen_dataset",
    "lerobot/austin_buds_dataset",
    "lerobot/dlr_sara_grid_clamp",
    "lerobot/conq_hose_manipulation",
    "lerobot/columbia_cairlab_pusht_real",
    "lerobot/dlr_sara_pour",
    "lerobot/dlr_edan_shared_control",
    "lerobot/ucsd_pick_and_place_dataset",
    "lerobot/berkeley_cable_routing",
    "lerobot/nyu_franka_play_dataset",
    "lerobot/austin_sirius_dataset",
    "lerobot/cmu_play_fusion",
    "lerobot/berkeley_gnm_sac_son",
    "lerobot/nyu_door_opening_surprising_effectiveness",
    "lerobot/berkeley_fanuc_manipulation",
    "lerobot/jaco_play",
    "lerobot/viola",
    "lerobot/kaist_nonprehensile",
    "lerobot/berkeley_mvp",
    "lerobot/uiuc_d3field",
    "lerobot/berkeley_gnm_recon",
    "lerobot/austin_sailor_dataset",
    "lerobot/utaustin_mutex",
    "lerobot/roboturk",
    "lerobot/stanford_hydra_dataset",
    "lerobot/berkeley_autolab_ur5",
    "lerobot/stanford_robocook",
    "lerobot/toto",
    "lerobot/fmb",
    "lerobot/droid_100",
    "lerobot/berkeley_rpt",
    "lerobot/stanford_kuka_multimodal_dataset",
    "lerobot/iamlab_cmu_pickup_insert",
    "lerobot/taco_play",
    "lerobot/berkeley_gnm_cory_hall",
    "lerobot/usc_cloth_sim",
]

available_datasets = sorted(
    set(itertools.chain(*available_datasets_per_env.values(), available_real_world_datasets))
)

# lists all available policies from `lerobot/common/policies`
available_policies = [
    "act",
    "diffusion",
    "tdmpc",
    "vqbet",
]

# lists all available robots from `lerobot/common/robot_devices/robots`
available_robots = [
    "koch",
    "koch_bimanual",
    "aloha",
    "so100",
    "moss",
]

# lists all available cameras from `lerobot/common/robot_devices/cameras`
available_cameras = [
    "opencv",
    "intelrealsense",
]

# lists all available motors from `lerobot/common/robot_devices/motors`
available_motors = [
    "dynamixel",
    "feetech",
]

# keys and values refer to yaml files
available_policies_per_env = {
    "aloha": ["act"],
    "pusht": ["diffusion", "vqbet"],
    "xarm": ["tdmpc"],
    "koch_real": ["act_koch_real"],
    "aloha_real": ["act_aloha_real"],
}

env_task_pairs = [(env, task) for env, tasks in available_tasks_per_env.items() for task in tasks]
env_dataset_pairs = [
    (env, dataset) for env, datasets in available_datasets_per_env.items() for dataset in datasets
]
env_dataset_policy_triplets = [
    (env, dataset, policy)
    for env, datasets in available_datasets_per_env.items()
    for dataset in datasets
    for policy in available_policies_per_env[env]
]
