#!/usr/bin/env python

# Example teleoperation command for full XLeRobot with leader arms plus gamepad:
#
# lerobot-teleoperate \
#     --robot.type=xlerobot \
#     --robot.base_type=lekiwi_base \
#     --robot.arms='{
#         "left_arm_port": "/dev/ttyACM2",
#         "right_arm_port": "/dev/ttyACM3",
#         "id": "follower"
#     }' \
#     --robot.base='{
#         "port": "/dev/ttyACM4",
#         "wheel_radius_m": 0.05,
#         "base_radius_m": 0.125
#     }' \
#     --robot.mount='{
#         "port": "/dev/ttyACM5",
#         "pan_motor_id": 0,
#         "tilt_motor_id": 1,
#         "motor_model": "sts3215",
#         "pan_key": "mount_pan.pos",
#         "tilt_key": "mount_tilt.pos",
#         "max_pan_speed_dps": 60.0,
#         "max_tilt_speed_dps": 45.0,
#         "pan_range": [-90.0, 90.0],
#         "tilt_range": [-30.0, 60.0]
#     }' \
#     --robot.cameras='{
#         "top":   {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 30}
#     }' \
#     --teleop.type=xlerobot_leader_gamepad \
#     --teleop.base_type=lekiwi_base_gamepad \
#     --teleop.arms='{
#         "left_arm_port": "/dev/ttyACM0",
#         "right_arm_port": "/dev/ttyACM1",
#         "id": "leader"
#     }' \
#     --teleop.base='{
#         "joystick_index": 0,
#         "max_speed_mps": 0.8,
#         "deadzone": 0.15,
#         "yaw_speed_deg": 45
#     }' \
#     --teleop.mount='{
#         "joystick_index": 0,
#         "deadzone": 0.15,
#         "polling_fps": 50,
#         "max_pan_speed_dps": 60.0,
#         "max_tilt_speed_dps": 45.0,
#         "pan_axis": 3,
#         "tilt_axis": 4,
#         "invert_pan": false,
#         "invert_tilt": true,
#         "pan_range": [-90.0, 90.0],
#         "tilt_range": [-30.0, 60.0]
#     }' \
#     --display_data=true

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

from __future__ import annotations

from contextlib import suppress
from functools import cached_property
from typing import Any

from ..bi_so101_leader.bi_so101_leader import BiSO101Leader
from ..biwheel_gamepad.teleop_biwheel_gamepad import BiwheelGamepadTeleop
from ..lekiwi_base_gamepad.teleop_lekiwi_base_gamepad import LeKiwiBaseTeleop
from ..teleoperator import Teleoperator
from ..xlerobot_mount_gamepad.teleop import XLeRobotMountGamepadTeleop
from .config import XLeRobotLeaderGamepadConfig


class XLeRobotLeaderGamepad(Teleoperator):
    """Composite teleoperator for XLeRobot with leader arms and gamepad control.

    This teleoperator combines three input methods:
    - BiSO101Leader: Leader arms for controlling follower arms
    - LeKiwiBaseTeleop: Xbox gamepad left stick for base movement
    - XLeRobotMountGamepadTeleop: Xbox gamepad right stick for mount pan/tilt

    All three inputs are merged into a single action dictionary that controls
    the complete XLeRobot system.
    """

    config_class = XLeRobotLeaderGamepadConfig
    name = "xlerobot_leader_gamepad"

    def __init__(self, config: XLeRobotLeaderGamepadConfig):
        self.config = config
        super().__init__(config)
        self.arm_teleop = BiSO101Leader(config.arms_config) if config.arms_config else None
        self.base_teleop = self._build_base_teleop()
        self.mount_teleop = self._build_mount_teleop()

    def _build_base_teleop(self) -> Teleoperator | None:
        base_config = getattr(self.config, "base_config", None)
        if base_config is None:
            return None
        base_type = getattr(self.config, "base_type", None) or XLeRobotLeaderGamepadConfig.BASE_TYPE_LEKIWI
        if base_type == XLeRobotLeaderGamepadConfig.BASE_TYPE_LEKIWI:
            return LeKiwiBaseTeleop(base_config)
        if base_type == XLeRobotLeaderGamepadConfig.BASE_TYPE_BIWHEEL:
            return BiwheelGamepadTeleop(base_config)
        raise ValueError(f"Unsupported base teleoperator type: {base_type}")

    def _build_mount_teleop(self) -> Teleoperator | None:
        mount_config = getattr(self.config, "mount_config", None)
        if mount_config is None:
            return None
        return XLeRobotMountGamepadTeleop(mount_config)

    def _iter_active_teleops(self) -> tuple[Teleoperator, ...]:
        return tuple(tp for tp in (self.arm_teleop, self.base_teleop, self.mount_teleop) if tp is not None)

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for teleop in self._iter_active_teleops():
            features.update(teleop.action_features)
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for teleop in self._iter_active_teleops():
            features.update(teleop.feedback_features)
        return features

    @property
    def is_connected(self) -> bool:
        return all(teleop.is_connected for teleop in self._iter_active_teleops())

    def connect(self, calibrate: bool = True) -> None:
        for teleop in self._iter_active_teleops():
            teleop.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        for teleop in self._iter_active_teleops():
            teleop.disconnect()

    def calibrate(self) -> None:
        for teleop in self._iter_active_teleops():
            teleop.calibrate()

    def configure(self) -> None:
        for teleop in self._iter_active_teleops():
            teleop.configure()

    def on_observation(self, robot_obs: dict[str, Any]) -> None:
        if self.mount_teleop and hasattr(self.mount_teleop, "on_observation"):
            with suppress(Exception):
                self.mount_teleop.on_observation(robot_obs)

    def get_action(self) -> dict[str, float]:
        action: dict[str, float] = {}
        for teleop in self._iter_active_teleops():
            action.update(teleop.get_action())
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        for teleop in self._iter_active_teleops():
            teleop.send_feedback(feedback)

    @property
    def is_calibrated(self) -> bool:
        return all(getattr(teleop, "is_calibrated", True) for teleop in self._iter_active_teleops())
