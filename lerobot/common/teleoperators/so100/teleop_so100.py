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

import logging
import time

import numpy as np

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import CalibrationMode, Motor
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    TorqueMode,
    apply_feetech_offsets_from_calibration,
)

from ..teleoperator import Teleoperator
from .configuration_so100 import SO100TeleopConfig


class SO100Teleop(Teleoperator):
    """
    [SO-100 Leader Arm](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = SO100TeleopConfig
    name = "so100"

    def __init__(self, config: SO100TeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.arm = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", CalibrationMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", CalibrationMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", CalibrationMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", CalibrationMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", CalibrationMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", CalibrationMode.RANGE_0_100),
            },
        )

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

    def configure(self) -> None:
        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration.
        for name in self.arm.names:
            self.arm.write("Torque_Enable", name, TorqueMode.DISABLED.value)

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        logging.info("Connecting arm.")
        self.arm.connect()
        self.configure()
        self.calibrate()

        # Check arm can be read
        self.arm.sync_read("Present_Position")

    def calibrate(self) -> None:
        raise NotImplementedError

    def set_calibration(self) -> None:
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """
        if not self.calibration_fpath.exists():
            logging.error("Calibration file not found. Please run calibration first")
            raise FileNotFoundError(self.calibration_fpath)

        self.arm.set_calibration(self.calibration_fpath)
        apply_feetech_offsets_from_calibration(self.arm, self.arm.calibration)

    def get_action(self) -> np.ndarray:
        """The returned action does not have a batch dimension."""
        # Read arm position
        before_read_t = time.perf_counter()
        action = self.arm.sync_read("Present_Position")
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
