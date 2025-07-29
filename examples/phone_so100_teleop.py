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
# See the License for the specif

import time

from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerEndEffectorConfig
from lerobot.robots.so100_follower.so100_follower_end_effector import SO100FollowerEndEffector
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone import Phone

robot_config = SO100FollowerEndEffectorConfig(
    port="/dev/tty.usbmodem58760434471",
    id="so100_follower_end_effector",
    urdf_path="./src/lerobot/teleoperators/sim/so101_new_calib.urdf",  # Path to your robot URDF
    end_effector_bounds={
        "min": [-1.0, -1.0, -1.0],  # min x, y, z
        "max": [1.0, 1.0, 1.0],  # max x, y, z
    },
    end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5, "roll": 1.0, "pitch": 1.0, "yaw": 1.0},
)

# TODO(pepijn): Add LeKiwi example

teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)

robot = SO100FollowerEndEffector(robot_config)
teleop_device = Phone(teleop_config)
robot.connect()
teleop_device.connect()

print("Starting teleop loop. Move your phone to teleoperate the robot.")

# TODO(pepijn): Test Android
while True:
    teleop_cmd = teleop_device.get_action()
    if not teleop_cmd:
        time.sleep(0.005)
        continue

    # Absolute targets now
    gripper_cmd = float(teleop_cmd["a3"])
    action = {
        "enabled": teleop_cmd["enabled"],
        "target_x": teleop_cmd["target_x"],
        "target_y": teleop_cmd["target_y"],
        "target_z": teleop_cmd["target_z"],
        "target_qx": teleop_cmd["target_qx"],
        "target_qy": teleop_cmd["target_qy"],
        "target_qz": teleop_cmd["target_qz"],
        "target_qw": teleop_cmd["target_qw"],
        "gripper": gripper_cmd,
    }

    # print one line with x,y,z and quaternion
    print(
        "\r"
        f"x={teleop_cmd['target_x']:.3f}, "
        f"y={teleop_cmd['target_y']:.3f}, "
        f"z={teleop_cmd['target_z']:.3f}, "
        f"qx={teleop_cmd['target_qx']:.3f}, "
        f"qy={teleop_cmd['target_qy']:.3f}, "
        f"qz={teleop_cmd['target_qz']:.3f}, "
        f"qw={teleop_cmd['target_qw']:.3f}",
        end="",
        flush=True,
    )

    robot.send_action(action)
    time.sleep(0.01)
