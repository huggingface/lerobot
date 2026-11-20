# !/usr/bin/env python

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

import time

from lerobot.common.teleop_setup import make_mobile_teleop
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main() -> None:
    robot_config = LeKiwiClientConfig(remote_ip="172.18.134.136", id="my_lekiwi")
    # Set this to None for keyboard-only control.
    teleop_arm_config = SO100LeaderConfig(port="/dev/tty.usbmodem585A0077581", id="my_awesome_leader_arm")
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(teleop_arm_config) if teleop_arm_config is not None else None
    keyboard = KeyboardTeleop(keyboard_config)
    teleop, action_processor = make_mobile_teleop(robot, base=keyboard, arm=leader_arm)

    # Start the LeKiwi host before connecting (see the LeKiwi guide).
    try:
        robot.connect()
        teleop.connect()
        init_rerun(session_name="lekiwi_teleop")

        print("Starting teleop loop...")
        while keyboard.is_connected:
            t0 = time.perf_counter()

            observation = robot.get_observation()

            raw_action = teleop.get_action(observation=observation)
            action = action_processor((raw_action, observation))

            _ = robot.send_action(action)

            log_rerun_data(observation=observation, action=action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if robot.is_connected:
                robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        finally:
            try:
                if teleop.is_connected:
                    teleop.disconnect()
            finally:
                if robot.is_connected:
                    robot.disconnect()


if __name__ == "__main__":
    main()
