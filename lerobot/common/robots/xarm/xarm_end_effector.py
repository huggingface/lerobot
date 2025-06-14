# !/usr/bin/env python

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
import time
import os
from typing import Any

import numpy as np
from xarm.wrapper import XArmAPI
from teleop.utils.jacobi_robot import JacobiRobot

from lerobot.common.cameras import make_cameras_from_configs
from lerobot.common.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from ..robot import Robot

from .config_xarm import XarmEndEffector

logger = logging.getLogger(__name__)


# pip install xarm-python-sdk
class XarmEndEffector(Robot):
    config_class = XarmEndEffector
    name = "xarm_end_effector"

    def __init__(self, config: XarmEndEffector):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        this_dir = os.path.dirname(os.path.abspath(__file__))

        self.config = config
        self.is_connected = False
        self.arm = None
        self.jacobi = JacobiRobot(this_dir, "lite6.urdf", ee_link="link6")

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # TODO: Connect to the robot and set it to a servo mode
        self.arm = XArmAPI("192.168.1.184")

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        return {
            "dtype": "float32",
            "shape": (7),
            "names": {
                "joint1": 0,
                "joint2": 1,
                "joint3": 2,
                "joint4": 3,
                "joint5": 4,
                "joint6": 5,
                "gripper": 6,
            },
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Transform action from end-effector space to joint space and send to motors.

        Args:
            action: Dictionary with keys 'pose', 'gripper', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'.

        Returns:
            The joint-space action that was sent to the motors
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        gripper = action["gripper"]

        full_action = action.copy
        full_action["gripper.pos"] = gripper

        if "pose" in action:
            # Convert pose to joint positions
            # Fill full_action with joint positions (jointX.pos)
            # TODO: Use jacobi.servo_to_pose() and jacobi.get_joint_position(joint_name)
            pass

        joint_action = {"gripper.pos": gripper}
        # TODO: Add joint positions for other motors

        return super().send_action(joint_action)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        # TODO: Read joint positions from xarm
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def reset(self):
        pass
