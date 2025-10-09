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
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("so101_mujoco")
@dataclass
class SO101MujocoConfig(RobotConfig):
    """Configuration for SO-101 in MuJoCo simulation.

    This robot uses the orient_down.py control logic with:
    - Keyboard teleoperation
    - Jacobian-based XYZ control at 180 Hz
    - Wrist-flex for vertical orientation maintenance
    - Gravity compensation
    - Position-based action recording at 30 Hz
    """

    # MuJoCo model path
    xml_path: Path = Path("gym-hil/gym_hil/assets/SO101/pick_scene.xml")

    # Control frequencies (all exact multiples for timing accuracy)
    record_fps: int = 30       # Recording/dataset frequency
    control_fps: int = 180     # Internal control loop frequency (6 × 30)
    physics_fps: int = 360     # MuJoCo physics frequency (2 × 180)

    # Camera configuration
    camera_width: int = 640
    camera_height: int = 480
    camera_names: list[str] = field(default_factory=lambda: ["top", "front", "wrist"])  # Multiple cameras
    show_collision_geometry: bool = False  # Show collision geometry (convex hulls) in camera renders

    # Control parameters (from orient_down.py)
    lin_speed: float = 0.04         # Linear velocity for XYZ control (m/s)
    yaw_speed: float = 1.20         # Wrist roll rate (rad/s)
    grip_speed: float = 0.7         # Gripper speed

    # Wrist tilt control gains
    ori_gain: float = 6.0           # Orientation correction gain
    tilt_deadzone: float = 0.03     # Radians, ignore small errors
    tilt_wmax: float = 6.0          # Max angular velocity for tilt

    # Jacobian solve damping
    lambda_pos: float = 1.0e-2      # DLS for XYZ positioning
    lambda_tilt: float = 1.0e-4     # DLS for wrist tilt

    # Rate limiting and smoothing
    vel_limit: float = 0.5          # Joint velocity limit for arm (rad/s)
    vel_limit_wrist: float = 8.0    # Joint velocity limit for wrist (rad/s)
    smooth_dq: float = 0.30         # Smoothing factor for arm joints
    smooth_dq_wrist: float = 0.08   # Smoothing factor for wrist joints

    # Gravity compensation
    wrist_gff_gain: float = 0.5     # Gravity feedforward gain for wrist

    # Safety
    table_z: float = 0.0            # Table height
    clearance: float = 0.07         # Minimum clearance above table (m)

    # End-effector site name
    ee_site_name: str = "wrist_site"

    # Tool axis (wrist_site frame, -Y points down when vertical)
    tool_axis_site: list[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])
