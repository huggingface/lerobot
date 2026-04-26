#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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


@dataclass
class NEOKeyboardEEConfig:
    """Configuration for keyboard end-effector teleoperation + IK on NERO."""

    enabled: bool = True
    urdf_path: str | None = None
    target_frame_name: str = "gripper"
    joint_names: list[str] = field(default_factory=lambda: [f"joint{i}" for i in range(1, 8)])
    max_linear_step_m: float = 0.01
    max_angular_step_rad: float = 0.12
    gripper_delta_per_step: float = 2.0
    gripper_min: float = 0.0
    gripper_max: float = 100.0
    position_bounds_min: list[float] | None = None
    position_bounds_max: list[float] | None = None


@dataclass
class NEOFollowerConfig:
    """Base configuration class for NERO Follower robots."""

    # CAN channel (e.g. "can0" on Linux, "0" on Windows)
    channel: str = "can0"

    # Communication interface ("socketcan" on Linux, "agx_cando" on Windows, "slcan" on macOS)
    interface: str = "socketcan"

    # Firmware version: "default" (<=1.10) or "v111" (>=1.11)
    # NOTE: field name matches pyAgxArm SDK's spelling ("firmeware")
    firmeware_version: str = "default"

    # Auto set motion mode when switching between move commands
    auto_set_motion_mode: bool = True

    # Enable software joint angle limits
    enable_joint_limits: bool = True

    # Speed percentage (0-100)
    speed_percent: int = 50

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety.
    max_relative_target: float | dict[str, float] | None = None

    # Disable torque on disconnect
    disable_torque_on_disconnect: bool = True

    # Cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Optional processor settings to support keyboard end-effector teleoperation.
    keyboard_ee: NEOKeyboardEEConfig = field(default_factory=NEOKeyboardEEConfig)


@RobotConfig.register_subclass("nero_follower")
@dataclass
class NEOFollowerRobotConfig(RobotConfig, NEOFollowerConfig):
    pass
