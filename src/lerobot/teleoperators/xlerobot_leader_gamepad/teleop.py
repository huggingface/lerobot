#!/usr/bin/env python

# Example teleoperation command for full XLeRobot with leader arms plus gamepad:
#
# lerobot-teleoperate \
#     --robot.type=xlerobot \
#     --robot.base_type=lekiwi_base \
#     --robot.arms='{\
#         "left_arm_port": "/dev/ttyACM2",\
#         "right_arm_port": "/dev/ttyACM3",\
#         "id": "follower"\
#     }' \
#     --robot.base='{\
#         "port": "/dev/ttyACM4",\
#         "wheel_radius_m": 0.05,\
#         "base_radius_m": 0.125\
#     }' \
#     --robot.mount='{\
#         "port": "/dev/ttyACM5",\
#         "pan_motor_id": 0,\
#         "tilt_motor_id": 1,\
#         "motor_model": "sts3215",\
#         "pan_key": "mount_pan.pos",\
#         "tilt_key": "mount_tilt.pos",\
#         "max_pan_speed_dps": 60.0,\
#         "max_tilt_speed_dps": 45.0,\
#         "pan_range": [-90.0, 90.0],\
#         "tilt_range": [-30.0, 60.0]\
#     }' \
#     --robot.cameras='{\
#         "left":  {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 15},\
#         "right": {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 15},\
#         "top":   {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 15}\
#     }' \
#     --teleop.type=xlerobot_leader_gamepad \
#     --teleop.base_type=lekiwi_base_gamepad \
#     --teleop.arms='{\
#         "left_arm_port": "/dev/ttyACM0",\
#         "right_arm_port": "/dev/ttyACM1",\
#         "id": "leader"\
#     }' \
#     --teleop.base='{\
#         "joystick_index": 0,\
#         "max_speed_mps": 0.8,\
#         "deadzone": 0.15,\
#         "yaw_speed_deg": 45\
#     }' \
#     --teleop.mount='{\
#         "joystick_index": 0,\
#         "deadzone": 0.15,\
#         "polling_fps": 50,\
#         "max_pan_speed_dps": 60.0,\
#         "max_tilt_speed_dps": 45.0,\
#         "pan_axis": 3,\
#         "tilt_axis": 4,\
#         "invert_pan": false,\
#         "invert_tilt": false,\
#         "pan_range": [-90.0, 90.0],\
#         "tilt_range": [-30.0, 60.0]\
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

from functools import cached_property
from typing import Any, Dict

from ..teleoperator import Teleoperator
from ..bi_so101_leader.bi_so101_leader import BiSO101Leader
from ..lekiwi_base_gamepad.teleop_lekiwi_base_gamepad import LeKiwiBaseTeleop
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
        super().__init__(config)
        self.config = config
        self.arm_teleop = BiSO101Leader(config.arms_config)
        self.base_teleop = self._build_base_teleop()
        self.mount_teleop = XLeRobotMountGamepadTeleop(config.mount_config)

    def _build_base_teleop(self) -> Teleoperator:
        base_type = getattr(self.config, "base_type", XLeRobotLeaderGamepadConfig.BASE_TYPE_LEKIWI)
        if base_type == XLeRobotLeaderGamepadConfig.BASE_TYPE_LEKIWI:
            return LeKiwiBaseTeleop(self.config.base_config)
        if base_type == XLeRobotLeaderGamepadConfig.BASE_TYPE_BIWHEEL:
            # TODO: Instantiate biwheel teleoperator once available.
            raise NotImplementedError("TODO: add biwheel base teleoperator support.")
        raise ValueError(f"Unsupported base teleoperator type: {base_type}")

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        features.update(self.arm_teleop.action_features)
        features.update(self.base_teleop.action_features)
        features.update(self.mount_teleop.action_features)
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return self.arm_teleop.feedback_features

    @property
    def is_connected(self) -> bool:
        return (
            self.arm_teleop.is_connected
            and self.base_teleop.is_connected
            and self.mount_teleop.is_connected
        )

    def connect(self, calibrate: bool = True) -> None:
        self.arm_teleop.connect(calibrate=calibrate)
        self.base_teleop.connect(calibrate=calibrate)
        self.mount_teleop.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        self.arm_teleop.disconnect()
        self.base_teleop.disconnect()
        self.mount_teleop.disconnect()

    def calibrate(self) -> None:
        self.arm_teleop.calibrate()
        self.mount_teleop.calibrate()

    def configure(self) -> None:
        self.arm_teleop.configure()
        self.base_teleop.configure()
        self.mount_teleop.configure()

    def on_observation(self, robot_obs: dict[str, Any]) -> None:
        if hasattr(self.mount_teleop, "on_observation"):
            try:
                self.mount_teleop.on_observation(robot_obs)
            except Exception:
                pass

    def get_action(self) -> Dict[str, float]:
        action = dict(self.arm_teleop.get_action())
        action.update(self.base_teleop.get_action())
        action.update(self.mount_teleop.get_action())
        return action

    def send_feedback(self, feedback: Dict[str, float]) -> None:
        self.arm_teleop.send_feedback(feedback)

    @property
    def is_calibrated(self) -> bool:
        return True
