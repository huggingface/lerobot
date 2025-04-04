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

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_widowx import WidowXConfig

logger = logging.getLogger(__name__)


class WidowX(Teleoperator):
    """
    [WidowX](https://www.trossenrobotics.com/widowx-250) developed by Trossen Robotics
    """

    config_class = WidowXConfig
    name = "widowx"

    def __init__(self, config: WidowXConfig):
        super().__init__(config)
        self.config = config
        self.arm = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "waist": Motor(1, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "shoulder": Motor(2, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "shoulder_shadow": Motor(3, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "elbow": Motor(4, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "elbow_shadow": Motor(5, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "forearm_roll": Motor(6, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "wrist_angle": Motor(7, "xm430-w350", MotorNormMode.RANGE_M100_100),
                "wrist_rotate": Motor(8, "xl430-w250", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(9, "xc430-w150", MotorNormMode.RANGE_0_100),
            },
        )

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

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.arm.connect()
        if not self.is_calibrated:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        raise NotImplementedError  # TODO(aliberts): adapt code below (copied from koch)
        logger.info(f"\nRunning calibration of {self}")
        self.arm.disable_torque()
        for name in self.arm.names:
            self.arm.write("Operating_Mode", name, OperatingMode.EXTENDED_POSITION.value)

        self.arm.write("Drive_Mode", "elbow_flex", DriveMode.INVERTED.value)
        drive_modes = {name: 1 if name == "elbow_flex" else 0 for name in self.arm.names}

        input("Move robot to the middle of its range of motion and press ENTER....")
        homing_offsets = self.arm.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        unknown_range_motors = [name for name in self.arm.names if name not in full_turn_motors]
        logger.info(
            f"Move all joints except {full_turn_motors} sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.arm.record_ranges_of_motion(unknown_range_motors)
        for name in full_turn_motors:
            range_mins[name] = 0
            range_maxes[name] = 4095

        self.calibration = {}
        for name, motor in self.arm.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=drive_modes[name],
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.arm.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.arm.disable_torque()
        self.arm.configure_motors()

        # Set secondary/shadow ID for shoulder and elbow. These joints have two motors.
        # As a result, if only one of them is required to move to a certain position,
        # the other will follow. This is to avoid breaking the motors.
        self.arm.write("Secondary_ID", "shoulder_shadow", 2)
        self.arm.write("Secondary_ID", "elbow_shadow", 4)

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        action = self.arm.read("Present_Position")
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.arm.disconnect()
        logger.info(f"{self} disconnected.")
