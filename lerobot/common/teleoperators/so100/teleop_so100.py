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
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    TorqueMode,
    apply_feetech_offsets_from_calibration,
    run_full_arm_calibration,
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
        self.id = config.id

        self.arm = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": config.shoulder_pan,
                "shoulder_lift": config.shoulder_lift,
                "elbow_flex": config.elbow_flex,
                "wrist_flex": config.wrist_flex,
                "wrist_roll": config.wrist_roll,
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

    def connect(self) -> None:
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

        # Check arm can be read
        self.arm.read("Present_Position")

        self.is_connected = True

    def calibrate(self) -> None:
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """
        arm_calib_path = self.calibration_dir / f"{self.id}.json"

        if arm_calib_path.exists():
            with open(arm_calib_path) as f:
                calibration = json.load(f)
        else:
            # TODO(rcadene): display a warning in __init__ if calibration file not available
            logging.info(f"Missing calibration file '{arm_calib_path}'")
            calibration = run_full_arm_calibration(self.arm, self.robot_type, self.name, "leader")

            logging.info(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
            arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
            with open(arm_calib_path, "w") as f:
                json.dump(calibration, f)

        self.arm.set_calibration(calibration)
        apply_feetech_offsets_from_calibration(self.arm, calibration)

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
