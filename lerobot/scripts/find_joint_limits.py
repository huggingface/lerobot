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

"""
Simple script to control a robot from teleoperation.

Example:

```shell
python -m lerobot.scripts.server.find_joint_limits \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue
```
"""

import time
from dataclasses import dataclass

import draccus
import numpy as np

from lerobot.common.model.kinematics import RobotKinematics
from lerobot.common.robots import (  # noqa: F401
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
)
from lerobot.common.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    gamepad,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
)


@dataclass
class FindJointLimitsConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second. By default, no limit.
    fps: int | None = None
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


@draccus.wrap()
def find_joint_and_ee_bounds(cfg: FindJointLimitsConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    start_episode_t = time.perf_counter()
    robot_type = cfg.robot.type.split("_")[0]
    kinematics = RobotKinematics(robot_type=robot_type)
    control_time_s = 10

    # Initialize min/max values
    observation = robot.get_observation()
    joint_positions = np.array([observation[f"{key}.pos"] for key in robot.bus.motors])
    ee_pos = kinematics.forward_kinematics(joint_positions, frame="gripper_tip")[:3, 3]

    max_pos = joint_positions.copy()
    min_pos = joint_positions.copy()
    max_ee = ee_pos.copy()
    min_ee = ee_pos.copy()

    while True:
        action = teleop.get_action()
        robot.send_action(action)

        observation = robot.get_observation()
        joint_positions = np.array([observation[f"{key}.pos"] for key in robot.bus.motors])
        ee_pos = kinematics.forward_kinematics(joint_positions, frame="gripper_tip")[:3, 3]

        # Skip initial warmup period
        if (time.perf_counter() - start_episode_t) < 5:
            continue

        # Update min/max values
        max_ee = np.maximum(max_ee, ee_pos)
        min_ee = np.minimum(min_ee, ee_pos)
        max_pos = np.maximum(max_pos, joint_positions)
        min_pos = np.minimum(min_pos, joint_positions)

        if time.perf_counter() - start_episode_t > control_time_s:
            print(f"Max ee position {np.round(max_ee, 4).tolist()}")
            print(f"Min ee position {np.round(min_ee, 4).tolist()}")
            print(f"Max joint pos position {np.round(max_pos, 4).tolist()}")
            print(f"Min joint pos position {np.round(min_pos, 4).tolist()}")
            break


if __name__ == "__main__":
    find_joint_and_ee_bounds()
