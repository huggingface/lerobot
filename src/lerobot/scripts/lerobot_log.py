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
Script to log motor positions and gripper torque.

Example:

```shell
python lerobot_log.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --freq=500
```
"""

import logging
import time
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.utils.import_utils import register_third_party_plugins


@dataclass
class LogConfig:
    robot: RobotConfig
    # Frequency of logging in milliseconds
    freq: int = 1000


@parser.wrap()
def log_motors(cfg: LogConfig) -> None:
    register_third_party_plugins()
    
    # Create and connect to robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    print("Connected to robot. Starting logging...")
    print("Motor positions and gripper torque will be logged every {} ms".format(cfg.freq))
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Read all motor positions
            positions = robot.bus.sync_read("Present_Position")
            
            # Read gripper torque (current)
            try:
                gripper_torque = robot.bus.read("Present_Current", "gripper")
            except KeyError:
                # If no gripper motor named "gripper", try to find it
                gripper_torque = "N/A"
                for motor_name in robot.bus.motors:
                    if "gripper" in motor_name.lower():
                        try:
                            gripper_torque = robot.bus.read("Present_Current", motor_name)
                            break
                        except:
                            pass
            
            # Format positions as comma-separated values
            pos_str = ", ".join(f"{name}: {pos:.2f}" for name, pos in positions.items())
            
            # Print in single line
            print(f"Positions: {pos_str} | Gripper Torque: {gripper_torque}")
            
            # Sleep for specified duration (convert ms to seconds)
            time.sleep(cfg.freq / 1000.0)
            
    except KeyboardInterrupt:
        print("\nStopping logging...")
    finally:
        robot.disconnect()
        print("Disconnected from robot")


def main():
    log_motors()


if __name__ == "__main__":
    main()