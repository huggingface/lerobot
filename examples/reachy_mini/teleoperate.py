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

import logging
import time

from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.robots.reachy_mini import ReachyMini, ReachyMiniConfig
from lerobot.teleoperators.keyboard import KeyboardReachyMiniTeleop, KeyboardReachyMiniTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30

def main():
    # Configure Reachy Mini (use 'localhost' if running on the robot or connecting via daemon)
    robot_config = ReachyMiniConfig(ip_address="localhost")
    robot = ReachyMini(robot_config)

    # Configure Keyboard Teleop
    teleop_config = KeyboardReachyMiniTeleopConfig()
    teleop = KeyboardReachyMiniTeleop(teleop_config)

    # Load the IK processor from the robots/reachy_mini/processor.json
    # This pipeline handles Cartesian (EE) control
    robot_action_processor = RobotProcessorPipeline.from_pretrained(
        "src/lerobot/robots/reachy_mini", 
        config_filename="processor.json"
    )

    # Connect to hardware
    robot.connect()
    teleop.connect()

    print("Teleoperation started. Use WASD/QE/UpDown to control the head.")
    print("Press ESC to quit.")

    try:
        while teleop.is_connected:
            t0 = time.perf_counter()

            # 1. Get Current Observation (needed for IK guess)
            observation = robot.get_observation()

            # 2. Get Keyboard Action (EE Intent: ee.wy, ee.wx, etc.)
            raw_action = teleop.get_action()

            # 3. Process Action through IK (converts EE Intent -> Joint Positions)
            # ReachyInverseKinematicsEEToJoints expects a dict with ee.* and observations
            processed_action = robot_action_processor((raw_action, observation))

            # 4. Send processed action to robot
            robot.send_action(processed_action)

            # Control loop frequency
            dt_s = time.perf_counter() - t0
            precise_sleep(max(1.0 / FPS - dt_s, 0.0))
            
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        robot.disconnect()

if __name__ == "__main__":
    main()
