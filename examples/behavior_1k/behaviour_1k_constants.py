#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from collections import OrderedDict

import numpy as np
import torch as th

ROBOT_TYPE = "R1Pro"
FPS = 30

ROBOT_CAMERA_NAMES = {
    "A1": {
        "external": "external::external_camera",
        "wrist": "external::wrist_camera",
    },
    "R1Pro": {
        "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
        "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
        "head": "robot_r1::robot_r1:zed_link:Camera:0",
    },
}

# Camera resolutions and corresponding intrinstics
HEAD_RESOLUTION = (720, 720)
WRIST_RESOLUTION = (480, 480)
# TODO: Fix A1
CAMERA_INTRINSICS = {
    "A1": {
        "external": np.array(
            [[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 240x240
        "wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 240x240
    },
    "R1Pro": {
        "head": np.array(
            [[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 720x720
        "left_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 480x480
        "right_wrist": np.array(
            [[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),  # 480x480
    },
}


# Dataset features for BEHAVIOR-1K LeRobotDataset v3.0
BEHAVIOR_DATASET_FEATURES = {
    # Actions
    "action": {
        "dtype": "float32",
        "shape": (23,),  # 23-dimensional action space for R1Pro
        "names": None,
    },
    # Proprioception
    "observation.state": {
        "dtype": "float32",
        "shape": (256,),  # Full proprioception state
        "names": None,
    },
    # Camera relative poses
    "observation.cam_rel_poses": {
        "dtype": "float32",
        "shape": (21,),  # 3 cameras * 7 (pos + quat)
        "names": None,
    },
    # Task information
    "observation.task_info": {
        "dtype": "float32",
        "shape": (None,),  # Variable size
        "names": None,
    },
    # RGB images
    "observation.images.rgb.head": {
        "dtype": "video",
        "shape": [720, 720, 3],
        "names": ["height", "width", "channels"],
    },
    "observation.images.rgb.left_wrist": {
        "dtype": "video",
        "shape": [480, 480, 3],
        "names": ["height", "width", "channels"],
    },
    "observation.images.rgb.right_wrist": {
        "dtype": "video",
        "shape": [480, 480, 3],
        "names": ["height", "width", "channels"],
    },
    # Depth images
    "observation.images.depth.head": {
        "dtype": "video",
        "shape": [720, 720, 1],
        "names": ["height", "width", "channels"],
    },
    "observation.images.depth.left_wrist": {
        "dtype": "video",
        "shape": [480, 480, 1],
        "names": ["height", "width", "channels"],
    },
    "observation.images.depth.right_wrist": {
        "dtype": "video",
        "shape": [480, 480, 1],
        "names": ["height", "width", "channels"],
    },
    # Segmentation instance ID images
    "observation.images.seg_instance_id.head": {
        "dtype": "video",
        "shape": [720, 720, 1],
        "names": ["height", "width", "channels"],
    },
    "observation.images.seg_instance_id.left_wrist": {
        "dtype": "video",
        "shape": [480, 480, 1],
        "names": ["height", "width", "channels"],
    },
    "observation.images.seg_instance_id.right_wrist": {
        "dtype": "video",
        "shape": [480, 480, 1],
        "names": ["height", "width", "channels"],
    },
}


# Action indices
ACTION_QPOS_INDICES = {
    "A1": OrderedDict(
        {
            "arm": np.s_[0:6],
            "gripper": np.s_[6:7],
        }
    ),
    "R1Pro": OrderedDict(
        {
            "base": np.s_[0:3],
            "torso": np.s_[3:7],
            "left_arm": np.s_[7:14],
            "left_gripper": np.s_[14:15],
            "right_arm": np.s_[15:22],
            "right_gripper": np.s_[22:23],
        }
    ),
}


# Proprioception configuration
PROPRIOCEPTION_INDICES = {
    "A1": OrderedDict(
        {
            "joint_qpos": np.s_[0:8],
            "joint_qpos_sin": np.s_[8:16],
            "joint_qpos_cos": np.s_[16:24],
            "joint_qvel": np.s_[24:32],
            "joint_qeffort": np.s_[32:40],
            "eef_0_pos": np.s_[40:43],
            "eef_0_quat": np.s_[43:47],
            "grasp_0": np.s_[47:48],
            "gripper_0_qpos": np.s_[48:50],
            "gripper_0_qvel": np.s_[50:52],
        }
    ),
    "R1Pro": OrderedDict(
        {
            "joint_qpos": np.s_[
                0:28
            ],  # Full robot joint positions, the first 6 are base joints, which is NOT allowed in standard track
            "joint_qpos_sin": np.s_[
                28:56
            ],  # Full robot joint positions, the first 6 are base joints, which is NOT allowed in standard track
            "joint_qpos_cos": np.s_[
                56:84
            ],  # Full robot joint positions, the first 6 are base joints, which is NOT allowed in standard track
            "joint_qvel": np.s_[84:112],
            "joint_qeffort": np.s_[112:140],
            "robot_pos": np.s_[140:143],  # Global pos, this is NOT allowed in standard track
            "robot_ori_cos": np.s_[143:146],  # Global ori, this is NOT allowed in standard track
            "robot_ori_sin": np.s_[146:149],  # Global ori, this is NOT allowed in standard track
            "robot_2d_ori": np.s_[149:150],  # 2D global ori, this is NOT allowed in standard track
            "robot_2d_ori_cos": np.s_[150:151],  # 2D global ori, this is NOT allowed in standard track
            "robot_2d_ori_sin": np.s_[151:152],  # 2D global ori, this is NOT allowed in standard track
            "robot_lin_vel": np.s_[152:155],
            "robot_ang_vel": np.s_[155:158],
            "arm_left_qpos": np.s_[158:165],
            "arm_left_qpos_sin": np.s_[165:172],
            "arm_left_qpos_cos": np.s_[172:179],
            "arm_left_qvel": np.s_[179:186],
            "eef_left_pos": np.s_[186:189],
            "eef_left_quat": np.s_[189:193],
            "gripper_left_qpos": np.s_[193:195],
            "gripper_left_qvel": np.s_[195:197],
            "arm_right_qpos": np.s_[197:204],
            "arm_right_qpos_sin": np.s_[204:211],
            "arm_right_qpos_cos": np.s_[211:218],
            "arm_right_qvel": np.s_[218:225],
            "eef_right_pos": np.s_[225:228],
            "eef_right_quat": np.s_[228:232],
            "gripper_right_qpos": np.s_[232:234],
            "gripper_right_qvel": np.s_[234:236],
            "trunk_qpos": np.s_[236:240],
            "trunk_qvel": np.s_[240:244],
            "base_qpos": np.s_[244:247],  # Base joint position, this is NOT allowed in standard track
            "base_qpos_sin": np.s_[247:250],  # Base joint position, this is NOT allowed in standard track
            "base_qpos_cos": np.s_[250:253],  # Base joint position, this is NOT allowed in standard track
            "base_qvel": np.s_[253:256],
        }
    ),
}

# Proprioception indices
PROPRIO_QPOS_INDICES = {
    "A1": OrderedDict(
        {
            "arm": np.s_[0:6],
            "gripper": np.s_[6:8],
        }
    ),
    "R1Pro": OrderedDict(
        {
            "torso": np.s_[6:10],
            "left_arm": np.s_[10:24:2],
            "right_arm": np.s_[11:24:2],
            "left_gripper": np.s_[24:26],
            "right_gripper": np.s_[26:28],
        }
    ),
}


# Joint limits (lower, upper)
JOINT_RANGE = {
    "A1": {
        "arm": (
            th.tensor([-2.8798, 0.0, -3.3161, -2.8798, -1.6581, -2.8798], dtype=th.float32),
            th.tensor([2.8798, 3.1415, 0.0, 2.8798, 1.6581, 2.8798], dtype=th.float32),
        ),
        "gripper": (th.tensor([0.00], dtype=th.float32), th.tensor([0.03], dtype=th.float32)),
    },
    "R1Pro": {
        "base": (
            th.tensor([-0.75, -0.75, -1.0], dtype=th.float32),
            th.tensor([0.75, 0.75, 1.0], dtype=th.float32),
        ),
        "torso": (
            th.tensor([-1.1345, -2.7925, -1.8326, -3.0543], dtype=th.float32),
            th.tensor([1.8326, 2.5307, 1.5708, 3.0543], dtype=th.float32),
        ),
        "left_arm": (
            th.tensor([-4.4506, -0.1745, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708], dtype=th.float32),
            th.tensor([1.3090, 3.1416, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708], dtype=th.float32),
        ),
        "left_gripper": (th.tensor([0.00], dtype=th.float32), th.tensor([0.05], dtype=th.float32)),
        "right_arm": (
            th.tensor([-4.4506, -3.1416, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708], dtype=th.float32),
            th.tensor([1.3090, 0.1745, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708], dtype=th.float32),
        ),
        "right_gripper": (th.tensor([0.00], dtype=th.float32), th.tensor([0.05], dtype=th.float32)),
    },
}


EEF_POSITION_RANGE = {
    "A1": {
        "0": (th.tensor([0.0, -0.7, 0.0], dtype=th.float32), th.tensor([0.7, 0.7, 0.7], dtype=th.float32)),
    },
    "R1Pro": {
        "left": (
            th.tensor([0.0, -0.65, 0.0], dtype=th.float32),
            th.tensor([0.65, 0.65, 2.5], dtype=th.float32),
        ),
        "right": (
            th.tensor([0.0, -0.65, 0.0], dtype=th.float32),
            th.tensor([0.65, 0.65, 2.5], dtype=th.float32),
        ),
    },
}


TASK_NAMES_TO_INDICES = {
    # B10
    "turning_on_radio": 0,
    "picking_up_trash": 1,
    "putting_away_Halloween_decorations": 2,
    "cleaning_up_plates_and_food": 3,
    "can_meat": 4,
    "setting_mousetraps": 5,
    "hiding_Easter_eggs": 6,
    "picking_up_toys": 7,
    "rearranging_kitchen_furniture": 8,
    "putting_up_Christmas_decorations_inside": 9,
    # B20
    "set_up_a_coffee_station_in_your_kitchen": 10,
    "putting_dishes_away_after_cleaning": 11,
    "preparing_lunch_box": 12,
    "loading_the_car": 13,
    "carrying_in_groceries": 14,
    "bringing_in_wood": 15,
    "moving_boxes_to_storage": 16,
    "bringing_water": 17,
    "tidying_bedroom": 18,
    "outfit_a_basic_toolbox": 19,
    # B30
    "sorting_vegetables": 20,
    "collecting_childrens_toys": 21,
    "putting_shoes_on_rack": 22,
    "boxing_books_up_for_storage": 23,
    "storing_food": 24,
    "clearing_food_from_table_into_fridge": 25,
    "assembling_gift_baskets": 26,
    "sorting_household_items": 27,
    "getting_organized_for_work": 28,
    "clean_up_your_desk": 29,
    # B40
    "setting_the_fire": 30,
    "clean_boxing_gloves": 31,
    "wash_a_baseball_cap": 32,
    "wash_dog_toys": 33,
    "hanging_pictures": 34,
    "attach_a_camera_to_a_tripod": 35,
    "clean_a_patio": 36,
    "clean_a_trumpet": 37,
    "spraying_for_bugs": 38,
    "spraying_fruit_trees": 39,
    # B50
    "make_microwave_popcorn": 40,
    "cook_cabbage": 41,
    "chop_an_onion": 42,
    "slicing_vegetables": 43,
    "chopping_wood": 44,
    "cook_hot_dogs": 45,
    "cook_bacon": 46,
    "freeze_pies": 47,
    "canning_food": 48,
    "make_pizza": 49,
}
TASK_INDICES_TO_NAMES = {v: k for k, v in TASK_NAMES_TO_INDICES.items()}
