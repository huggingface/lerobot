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
Robot hardware specification - source of truth for all servos and their properties.

This module defines the complete robot hardware configuration, including all servos,
their roles, operating modes, and control parameters. The hardware definition is
independent of how servos are physically connected to control buses.

This design allows the robot to run on various numbers of buses:
- 4-bus layout: Each role (left_arm, right_arm, head, base) gets its own bus
- 2-bus layout: Mixed roles on fewer buses (e.g., left_arm+head on bus1, right_arm+base on bus2)
- Custom layouts: Any combination of roles on any number of buses

The bus mapping is configured in this same module, making the system
flexible for different hardware setups while maintaining the same robot capabilities.
"""

from dataclasses import dataclass
from typing import Dict, List
from lerobot.motors import MotorNormMode


@dataclass
class RobotPart:
    """Definition of a single robot servo/motor."""
    id: int
    model: str
    role: str  # "left_arm", "right_arm", "head", "base"
    norm_mode: MotorNormMode
    operating_mode: str = "position"  # "position" or "velocity"
    pid_p: int = 16
    pid_i: int = 0
    pid_d: int = 43


# Source of truth: All robot servos defined once
ROBOT_PARTS = {
    # Left arm servos
    "left_arm_shoulder_pan": RobotPart(1, "sts3215", "left_arm", MotorNormMode.DEGREES),
    "left_arm_shoulder_lift": RobotPart(2, "sts3215", "left_arm", MotorNormMode.DEGREES),
    "left_arm_elbow_flex": RobotPart(3, "sts3215", "left_arm", MotorNormMode.DEGREES),
    "left_arm_wrist_flex": RobotPart(4, "sts3215", "left_arm", MotorNormMode.DEGREES),
    "left_arm_wrist_roll": RobotPart(5, "sts3215", "left_arm", MotorNormMode.DEGREES),
    "left_arm_gripper": RobotPart(6, "sts3215", "left_arm", MotorNormMode.RANGE_0_100),
    
    # Right arm servos
    "right_arm_shoulder_pan": RobotPart(1, "sts3215", "right_arm", MotorNormMode.DEGREES),
    "right_arm_shoulder_lift": RobotPart(2, "sts3215", "right_arm", MotorNormMode.DEGREES),
    "right_arm_elbow_flex": RobotPart(3, "sts3215", "right_arm", MotorNormMode.DEGREES),
    "right_arm_wrist_flex": RobotPart(4, "sts3215", "right_arm", MotorNormMode.DEGREES),
    "right_arm_wrist_roll": RobotPart(5, "sts3215", "right_arm", MotorNormMode.DEGREES),
    "right_arm_gripper": RobotPart(6, "sts3215", "right_arm", MotorNormMode.RANGE_0_100),
    
    # Head servos
    "head_motor_1": RobotPart(7, "sts3215", "head", MotorNormMode.DEGREES),
    "head_motor_2": RobotPart(8, "sts3215", "head", MotorNormMode.DEGREES),
    
    # Base servos (wheels)
    "base_left_wheel": RobotPart(9, "sts3215", "base", MotorNormMode.RANGE_M100_100, "velocity"),
    "base_back_wheel": RobotPart(10, "sts3215", "base", MotorNormMode.RANGE_M100_100, "velocity"),
    "base_right_wheel": RobotPart(11, "sts3215", "base", MotorNormMode.RANGE_M100_100, "velocity"),
}

DEFAULT_IDS_BY_LAYOUT = {
    "4_bus": {
        # Left arm bus
        "left_arm_shoulder_pan": 1,
        "left_arm_shoulder_lift": 2,
        "left_arm_elbow_flex": 3,
        "left_arm_wrist_flex": 4,
        "left_arm_wrist_roll": 5,
        "left_arm_gripper": 6,
        # Right arm bus
        "right_arm_shoulder_pan": 1,
        "right_arm_shoulder_lift": 2,
        "right_arm_elbow_flex": 3,
        "right_arm_wrist_flex": 4,
        "right_arm_wrist_roll": 5,
        "right_arm_gripper": 6,
        # Head bus
        "head_motor_1": 1,
        "head_motor_2": 2,
        # Base bus
        "base_left_wheel": 1,
        "base_back_wheel": 2,
        "base_right_wheel": 3,
    },
    "2_bus": {name: part.id for name, part in ROBOT_PARTS.items()},
}


@dataclass
class BusMapping:
    """Defines which robot parts go on which bus."""
    bus_name: str
    port_key: str  # key in config.ports dict
    motor_patterns: List[str]  # patterns like ["left_arm_*", "head_*"]


# 4-bus mapping: each role gets its own bus
BUS_MAPPING_4_BUS = {
    "left_arm_bus": BusMapping("left_arm_bus", "left_arm", ["left_arm_*"]),
    "right_arm_bus": BusMapping("right_arm_bus", "right_arm", ["right_arm_*"]),
    "head_bus": BusMapping("head_bus", "head", ["head_*"]),
    "base_bus": BusMapping("base_bus", "base", ["base_*"]),
}

# 2-bus mapping: mixed roles on fewer buses
BUS_MAPPING_2_BUS = {
    "board_A": BusMapping("board_A", "left_arm", ["left_arm_*", "head_*"]),
    "board_B": BusMapping("board_B", "right_arm", ["right_arm_*", "base_*"]),
}

# Available bus mappings
BUS_MAPPINGS = {
    "4_bus": BUS_MAPPING_4_BUS,
    "2_bus": BUS_MAPPING_2_BUS,
}
