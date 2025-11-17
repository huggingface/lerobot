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

from dataclasses import dataclass, field
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.robots.sourccey.sourccey.sourccey.config_sourccey import sourccey_motor_models


@TeleoperatorConfig.register_subclass("sourccey_teleop")
@dataclass
class PhoneTeleoperatorSourcceyConfig(TeleoperatorConfig):
    """Configuration for Sourccey phone teleoperation."""
    
    # Which arm the phone controls: "left" or "right"
    arm_side: str = "left"

    # gRPC server settings
    grpc_port: int = 8765  # Default port to match phone app
    grpc_timeout: float = 100.0
    
    # Robot model paths - same as SO100
    urdf_path: str = "lerobot/robots/sourccey/sourccey_v2beta/model/Arm.urdf"
    mesh_path: str | None = "lerobot/robots/sourccey/sourccey_v2beta/model/meshes"
    calibration_path_left: str = "lerobot/robots/sourccey/sourccey/sourccey/left_arm_default_calibration.json"
    calibration_path_right: str = "lerobot/robots/sourccey/sourccey/sourccey/right_arm_default_calibration.json"
    motor_models: dict[str, str] = field(default_factory=sourccey_motor_models)
    
    # IK solver settings - same as SO100
    target_link_name: str = "Feetech-Servo-Motor-v1-5"
    # Left rest pose (radians)
    rest_pose: tuple[float, ...] = (
        -0.864068,   # shoulder_pan  (-49.506903 deg)
        2.095329,    # shoulder_lift (100.0 deg)
        -2.205474,   # elbow_flex    (-97.716150 deg)
        0.093922,    # wrist_flex    (5.381376 deg)
        0.014914,    # wrist_roll    (0.854701 deg)
        1.738416,    # gripper       (99.603960 -> used as 0-100)
    )   # Always in radians - initial robot positions for IK solver

    # Right rest pose (radians)
    rest_pose_right: tuple[float, ...] = (
        0.640044,    # right_shoulder_pan  (36.671576 deg)
        -2.474699,   # right_shoulder_lift (-141.809571 deg)
        2.518931,    # right_elbow_flex    (144.292517 deg)
        -0.235352,   # right_wrist_flex    (-13.484163 deg)
        0.020884,    # right_wrist_roll    (1.196581 deg)
        0.004511,    # right_gripper       (0.258398 deg)
    )

    # Phone mapping settings
    rotation_sensitivity: float = 1.0
    sensitivity_normal: float = 0.5
    sensitivity_precision: float = 0.2
    # Global mapping gain multiplier for both translation and rotation
    mapping_gain: float = 2.0
    
    # Initial robot pose (when connecting phone) - same as SO100
    initial_position: tuple[float, ...] = (0.0, -0.17, 0.237)  # meters
    initial_wxyz: tuple[float, ...] = (0, 0, 1, 0)  # quaternion (w,x,y,z)
    # Right-arm initial
    initial_position_right: tuple[float, ...] = (
        0.09376381640512954,
        -0.17794639170766768,
        0.2820500723608793,
    )
    initial_wxyz_right: tuple[float, ...] = (0, 0, 1, 0)

    # Joint offsets for calibration (degrees)
    joint_offsets_deg: dict[str, float] = field(default_factory=lambda: {
        "shoulder_pan": 0.0,  # Will be set based on arm_side: -30.0 for left, 30.0 for right
        "shoulder_lift": 0.0,
        "elbow_flex": 0.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
    })

    # Output controls
    # If True, the teleop will emit both arms' keys. The non-controlled arm will be set to rest.
    emit_both_arms: bool = True
    # Whether incoming observations from the host are already in degrees (True) or normalized [-100,100] (False).
    observation_uses_degrees: bool = False
    
    # Visualization settings
    enable_visualization: bool = True
    viser_port: int = 8080
    
    # Gripper settings - same as SO100
    gripper_min_pos: float = 0.0    # Gripper closed position (0% slider)
    gripper_max_pos: float = 50.0   # Gripper open position (100% slider)

    # Remove duplicate joint_offsets_deg - already defined above on line 78

    # Base control via phone (optional)
    enable_base_from_phone: bool = True
    base_scale_x: float = 1.0
    base_scale_y: float = 1.0
    base_scale_theta: float = 1.0
    # Allow base to run when teleop is inactive or resetting
    base_allow_when_inactive: bool = True
    base_allow_when_resetting: bool = True