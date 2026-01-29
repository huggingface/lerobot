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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("BrewieConfig")
@dataclass
class BrewieConfig(RobotConfig):
    
    # ROS connection parameters
    master_ip: str = "192.168.20.21"
    master_port: int = 9090
    
    # Servo configuration
    servo_ids: list[int] = field(default_factory=lambda: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    servo_duration: float = 0.1  # Duration for servo movements
     
    # Servo mapping (ID -> joint name)
    servo_mapping: dict[int, str] = field(default_factory=lambda: {
        13: "left_shoulder_pan",
        14: "right_shoulder_pan", 
        15: "left_shoulder_lift",
        16: "right_shoulder_lift",
        17: "left_forearm_roll",
        18: "right_forearm_roll",
        19: "left_forearm_pitch",
        20: "right_forearm_pitch",
        21: "left_gripper",
        22: "right_gripper",
        23: "head_pan",      # head rotation
        24: "head_tilt"      # head tilt
    })
    
    # ROS topics and services
    position_service: str = "/ros_robot_controller/bus_servo/get_position"
    set_position_topic: str = "/ros_robot_controller/bus_servo/set_position"
    camera_topic: str = "/camera/image_raw/compressed"
    
    # Additional sensor topics
    joy_topic: str = "/joy"  # Joystick data
    imu_topic: str = "/imu"  # IMU data (gyroscope, accelerometer)
    
    # Safety parameters
    max_relative_target: float | None = None  # Maximum relative movement per step
    disable_torque_on_disconnect: bool = True  # Disable torque when disconnecting
    
    # cameras Will use?
    cameras: dict[str, CameraConfig] = field(default_factory=dict)