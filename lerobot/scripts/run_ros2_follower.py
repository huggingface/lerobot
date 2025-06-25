#!/usr/bin/env python3
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

import sys
import logging
from pprint import pformat
from dataclasses import asdict, dataclass

from lerobot.common.config import parse_and_overload, parser
from lerobot.common.robot_devices.control_configs import ControlPipelineConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.ros2_teleop import ROS2TeleopConfig, run_ros2_follower
from lerobot.common.logging_utils import init_logging

@dataclass
class ROS2FollowerConfig:
    """Configuration for ROS2 follower teleoperation."""
    node_name: str = "lerobot_teleop"
    topic_name: str = "joint_states"
    use_best_effort_qos: bool = True

@parser.wrap()
def run_follower(cfg: ControlPipelineConfig, ros2_config: ROS2FollowerConfig = None):
    """Run a robot as a ROS2 follower that subscribes to joint states."""
    init_logging()
    
    if ros2_config is None:
        ros2_config = ROS2FollowerConfig()
    
    teleop_config = ROS2TeleopConfig(
        node_name=ros2_config.node_name,
        topic_name=ros2_config.topic_name,
        use_best_effort_qos=ros2_config.use_best_effort_qos
    )
    
    logging.info("ROS2 Follower Configuration:")
    logging.info(pformat(asdict(teleop_config)))
    logging.info("Robot Configuration:")
    logging.info(pformat(asdict(cfg.robot)))
    
    robot = make_robot_from_config(cfg.robot)
    
    try:
        run_ros2_follower(robot, teleop_config)
    except KeyboardInterrupt:
        logging.info("Follower node stopped by user")
    finally:
        if robot.is_connected:
            robot.disconnect()

if __name__ == "__main__":
    run_follower()
