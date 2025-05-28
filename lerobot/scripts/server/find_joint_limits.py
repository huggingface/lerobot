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

from lerobot.common.model.kinematics_utils import RobotKinematics
from lerobot.common.robots import (  # noqa: F401
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so100_follower_end_effector,
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

    urdf_path: str = "/Users/michel_aractingi/code/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"


@draccus.wrap()
def find_joint_and_ee_bounds(cfg: FindJointLimitsConfig):
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    start_episode_t = time.perf_counter()
    ee_list = []
    pos_list = []
    kinematics = RobotKinematics(cfg.urdf_path)
    control_time_s = 10
    while True:
        action = teleop.get_action()
        robot.send_action(action)

        joint_positions = robot.bus.sync_read("Present_Position")
        joint_positions = np.array([joint_positions[key] for key in joint_positions])
        ee_pos, _, _ = kinematics.forward_kinematics(joint_positions * np.pi / 180)
        ee_list.append(ee_pos.copy())
        pos_list.append(joint_positions)

        if time.perf_counter() - start_episode_t > control_time_s:
            max_ee = np.max(np.stack(ee_list), 0)
            min_ee = np.min(np.stack(ee_list), 0)
            max_pos = np.max(np.stack(pos_list), 0)
            min_pos = np.min(np.stack(pos_list), 0)
            print(f"Max ee position {max_ee}")
            print(f"Min ee position {min_ee}")
            print(f"Max joint pos position {max_pos}")
            print(f"Min joint pos position {min_pos}")
            break


if __name__ == "__main__":
    find_joint_and_ee_bounds()
