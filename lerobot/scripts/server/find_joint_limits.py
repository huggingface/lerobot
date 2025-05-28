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

import argparse
import time

import cv2
import numpy as np

from lerobot.common.robot_devices.control_utils import is_headless
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.configs import parser
from lerobot.scripts.server.kinematics import RobotKinematics


def find_joint_bounds(
    robot,
    control_time_s=30,
    display_cameras=False,
):
    if not robot.is_connected:
        robot.connect()

    start_episode_t = time.perf_counter()
    pos_list = []
    while True:
        observation, action = robot.teleop_step(record_data=True)

        # Wait for 5 seconds to stabilize the robot initial position
        if time.perf_counter() - start_episode_t < 5:
            continue

        pos_list.append(robot.follower_arms["main"].read("Present_Position"))

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if time.perf_counter() - start_episode_t > control_time_s:
            max = np.max(np.stack(pos_list), 0)
            min = np.min(np.stack(pos_list), 0)
            print(f"Max angle position per joint {max}")
            print(f"Min angle position per joint {min}")
            break


def find_ee_bounds(
    robot,
    control_time_s=30,
    display_cameras=False,
):
    if not robot.is_connected:
        robot.connect()

    start_episode_t = time.perf_counter()
    ee_list = []
    while True:
        observation, action = robot.teleop_step(record_data=True)

        # Wait for 5 seconds to stabilize the robot initial position
        if time.perf_counter() - start_episode_t < 5:
            continue

        kinematics = RobotKinematics(robot.robot_type)
        joint_positions = robot.follower_arms["main"].read("Present_Position")
        print(f"Joint positions: {joint_positions}")
        ee_list.append(kinematics.fk_gripper_tip(joint_positions)[:3, 3])

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if time.perf_counter() - start_episode_t > control_time_s:
            max = np.max(np.stack(ee_list), 0)
            min = np.min(np.stack(ee_list), 0)
            print(f"Max ee position {max}")
            print(f"Min ee position {min}")
            break


if __name__ == "__main__":
    # Create argparse for script-specific arguments
    parser = argparse.ArgumentParser(add_help=False)  # Set add_help=False to avoid conflict
    parser.add_argument(
        "--mode",
        type=str,
        default="joint",
        choices=["joint", "ee"],
        help="Mode to run the script in. Can be 'joint' or 'ee'.",
    )
    parser.add_argument(
        "--control-time-s",
        type=int,
        default=30,
        help="Time step to use for control.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="so100",
        help="Robot type (so100, koch, aloha, etc.)",
    )

    # Only parse known args, leaving robot config args for Hydra if used
    args = parser.parse_args()

    # Create robot with the appropriate config
    robot_config = RobotConfig.get_choice_class(args.robot_type)(mock=False)
    robot = make_robot_from_config(robot_config)

    if args.mode == "joint":
        find_joint_bounds(robot, args.control_time_s)
    elif args.mode == "ee":
        find_ee_bounds(robot, args.control_time_s)

    if robot.is_connected:
        robot.disconnect()
