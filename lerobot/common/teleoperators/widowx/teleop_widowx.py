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

import json
import logging
import time

import numpy as np

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import TorqueMode
from lerobot.common.motors.dynamixel import (
    DynamixelMotorsBus,
    run_arm_calibration,
)

from ..teleoperator import Teleoperator
from .configuration_widowx import WidowXTeleopConfig


class WidowXTeleop(Teleoperator):
    """
    [WidowX](https://www.trossenrobotics.com/widowx-250) developed by Trossen Robotics
    """

    config_class = WidowXTeleopConfig
    name = "widowx"

    def __init__(self, config: WidowXTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.arm = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "waist": config.waist,
                "shoulder": config.shoulder,
                "shoulder_shadow": config.shoulder_shadow,
                "elbow": config.elbow,
                "elbow_shadow": config.elbow_shadow,
                "forearm_roll": config.forearm_roll,
                "wrist_angle": config.wrist_angle,
                "wrist_rotate": config.wrist_rotate,
                "gripper": config.gripper,
            },
        )

        self.is_connected = False
        self.logs = {}

    @property
    def action_feature(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_feature(self) -> dict:
        return {}

    def _set_shadow_motors(self):
        """
        Set secondary/shadow ID for shoulder and elbow. These joints have two motors.
        As a result, if only one of them is required to move to a certain position,
        the other will follow. This is to avoid breaking the motors.
        """
        shoulder_idx = self.config.shoulder[0]
        self.arm.write("Secondary_ID", shoulder_idx, "shoulder_shadow")

        elbow_idx = self.config.elbow[0]
        self.arm.write("Secondary_ID", elbow_idx, "elbow_shadow")

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        logging.info("Connecting arm.")
        self.arm.connect()

        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration.
        self.arm.write("Torque_Enable", TorqueMode.DISABLED.value)
        self.calibrate()

        self._set_shadow_motors()

        logging.info("Activating torque.")
        self.arm.write("Torque_Enable", TorqueMode.ENABLED.value)

        # Check arm can be read
        self.arm.read("Present_Position")

        # Connect the cameras
        for cam in self.cameras.values():
            cam.connect()

        self.is_connected = True

    def calibrate(self) -> None:
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """
        if self.calibration_fpath.exists():
            with open(self.calibration_fpath) as f:
                calibration = json.load(f)
        else:
            # TODO(rcadene): display a warning in __init__ if calibration file not available
            logging.info(f"Missing calibration file '{self.calibration_fpath}'")
            calibration = run_arm_calibration(self.arm, self.robot_type, self.name, "leader")

            logging.info(f"Calibration is done! Saving calibration file '{self.calibration_fpath}'")
            self.calibration_fpath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.calibration_fpath, "w") as f:
                json.dump(calibration, f)

        self.arm.set_calibration(calibration)

    def get_action(self) -> np.ndarray:
        """The returned action does not have a batch dimension."""
        # Read arm position
        before_read_t = time.perf_counter()
        action = self.arm.read("Present_Position")
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return action

    def send_feedback(self, feedback: np.ndarray) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        self.arm.disconnect()
        self.is_connected = False
