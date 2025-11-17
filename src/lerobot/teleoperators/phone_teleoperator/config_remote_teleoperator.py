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

from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("phone")
@dataclass
class PhoneTeleoperatorConfig(TeleoperatorConfig):
    """Configuration for phone teleoperation."""
    
    # gRPC server settings
    grpc_port: int = 8765  # Default port to match phone app
    grpc_timeout: float = 100.0
    
    # Robot model paths - can be set via command line or will be auto-detected for SO100
    urdf_path: str = "src/lerobot/robots/so100_follower/model/so100.urdf"
    mesh_path: str = "src/lerobot/robots/so100_follower/model/assets"
    
    # IK solver settings
    # target_link_name: str = "Fixed_Jaw"
    target_link_name: str = "gripper"
    # rest_pose: tuple[float, ...] = (0.008439, 0.289993, -0.043729, -0.121981, 0.128119, 0.020447)  # Always in radians - initial robot positions for IK solver
    
    rest_pose: tuple[float, ...] = (0.017499, -1.661131, 1.659391, 1.130985, 0.004688, 0.010240)  # Always in radians - initial robot positions for IK solver

    # Phone mapping settings
    rotation_sensitivity: float = 1.0
    sensitivity_normal: float = 0.5
    sensitivity_precision: float = 0.2
    # Global mapping gain multiplier for both translation and rotation
    mapping_gain: float = 2.0
    
    # Initial robot pose (when connecting phone)
    # initial_position: tuple[float, ...] = (0.0, -0.17, 0.237)  # meters
    # initial_wxyz: tuple[float, ...] = (0, 0, 1, 0)  # quaternion (w,x,y,z)
    # initial_position: tuple[float, ...] = (0.0, -0.17, 0.237)  # meters
    # initial_wxyz: tuple[float, ...] = (0, 0, 1, 0)  # quaternion (w,x,y,z)
    initial_position: tuple[float, ...] = (0.0, -0.17, 0.237)  # meters
    initial_wxyz: tuple[float, ...] = (0, 0, 1, 0)  # quaternion (w,x,y,z)

    # Visualization settings
    enable_visualization: bool = True
    viser_port: int = 8080
    
    # Gripper settings
    gripper_min_pos: float = 0.0    # Gripper closed position (0% slider)
    gripper_max_pos: float = 50.0   # Gripper open position (100% slider) - matches SO100 default
    
    # Safety settings
    max_relative_target: Optional[float] = None 