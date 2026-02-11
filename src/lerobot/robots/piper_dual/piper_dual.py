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

import logging
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors.piper.piper_slave import PiperMotorsBus, PiperMotorsBusConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_piper_dual import PIPERDualConfig

logger = logging.getLogger(__name__)


def get_motor_names(arm: dict[str, Any]) -> list[str]:
    return [motor for arm_key, bus in arm.items() for motor in bus.motors]


class PIPERDual(Robot):
    config_class = PIPERDualConfig
    name = "piper_dual"

    def __init__(self, config: PIPERDualConfig):
        super().__init__(config)
        self.config = config

        self.left_bus = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name=config.left_port,
                motors={
                    "joint_1": (1, "agilex_piper"),
                    "joint_2": (2, "agilex_piper"),
                    "joint_3": (3, "agilex_piper"),
                    "joint_4": (4, "agilex_piper"),
                    "joint_5": (5, "agilex_piper"),
                    "joint_6": (6, "agilex_piper"),
                    "gripper": (7, "agilex_piper"),
                },
            )
        )

        self.right_bus = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name=config.right_port,
                motors={
                    "joint_1": (1, "agilex_piper"),
                    "joint_2": (2, "agilex_piper"),
                    "joint_3": (3, "agilex_piper"),
                    "joint_4": (4, "agilex_piper"),
                    "joint_5": (5, "agilex_piper"),
                    "joint_6": (6, "agilex_piper"),
                    "gripper": (7, "agilex_piper"),
                },
            )
        )

        self.logs = {}
        self._is_connected = False
        self._is_calibrated = False
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        # Left Arm
        left_arm_dict = {"follower": self.left_bus}
        left_action_names = [f"left_{name}" for name in get_motor_names(left_arm_dict)]
        left_state_names = [f"left_{name}" for name in get_motor_names(left_arm_dict)]

        # Right Arm
        right_arm_dict = {"follower": self.right_bus}
        right_action_names = [f"right_{name}" for name in get_motor_names(right_arm_dict)]
        right_state_names = [f"right_{name}" for name in get_motor_names(right_arm_dict)]

        action_names = left_action_names + right_action_names
        state_names = left_state_names + right_state_names

        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        """用于 record/replay 的电机动作描述"""
        left_arm_dict = {"follower": self.left_bus}
        left_motor_names = get_motor_names(left_arm_dict)

        right_arm_dict = {"follower": self.right_bus}
        right_motor_names = get_motor_names(right_arm_dict)

        features = {}
        for name in left_motor_names:
            features[f"left_{name}.pos"] = float
        for name in right_motor_names:
            features[f"right_{name}.pos"] = float

        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """用于 record/replay 的相机图像描述"""
        return {cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def configure(self, **kwargs):
        pass

    @property
    def is_connected(self) -> bool:
        """机器人和所有相机是否都已连接"""
        return (
            self.left_bus.is_connected
            and self.right_bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        """机器人是否已完成标定"""
        return self._is_calibrated

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self) -> None:
        """Connect piper and cameras"""
        if self._is_connected:
            raise DeviceAlreadyConnectedError("Piper is already connected. Do not run robot.connect() twice.")

        print("Connecting left arm...")
        if not self.config.read_only:
            self.left_bus.connect(enable=True)
            print("Left arm connected (ACTIVE).")
        else:
            print("Left arm connected (PASSIVE).")

        print("Connecting right arm...")
        if not self.config.read_only:
            self.right_bus.connect(enable=True)
            print("Right arm connected (ACTIVE).")
        else:
            print("Right arm connected (PASSIVE).")

        print(f"piper follower dual connected (read_only={self.config.read_only})")

        # connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            print(f"camera {name} connected")

        print("All connected")
        self._is_connected = True

        if not self.config.read_only:
            self.calibrate()

    def disconnect(self) -> None:
        """move to home position, disenable piper and cameras"""

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self._is_connected = False

    def calibrate(self):
        """move piper to the home position"""
        if not self._is_connected:
            raise ConnectionError()

        self.left_bus.apply_calibration()
        self.right_bus.apply_calibration()
        self._is_calibrated = True

    def get_observation(self) -> dict:
        """Capture current joint positions and camera images"""
        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected. Run `robot.connect()` first.")

        # Read left arm
        left_state = self.left_bus.read()
        obs_dict = {f"left_{joint}.pos": float(val) for joint, val in left_state.items()}

        # Read right arm
        right_state = self.right_bus.read()
        obs_dict.update({f"right_{joint}.pos": float(val) for joint, val in right_state.items()})

        # Read cameras
        for name, cam in self.cameras.items():
            obs_dict[name] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Receive action dict from record() and send to motor"""

        if not self._is_connected:
            raise DeviceNotConnectedError("Piper is not connected.")

        motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]

        # Check for legacy dataset format (single "action" key with array)
        if "action" in action and "left_joint_1.pos" not in action:
            # Assume format is [left_7_joints, right_7_joints]
            raw_action = action["action"]
            if len(raw_action) == 14:
                left_target_joints = raw_action[:7]
                right_target_joints = raw_action[7:]
            else:
                print(f"WARNING: Unexpected action array length {len(raw_action)}, expected 14 for dual arm.")
                return action
        else:
            # Standard named format
            left_target_joints = [action[f"left_{motor}.pos"] for motor in motor_order]
            right_target_joints = [action[f"right_{motor}.pos"] for motor in motor_order]

        if not self.config.read_only:
            self.left_bus.write(left_target_joints)
            self.right_bus.write(right_target_joints)

        return action
