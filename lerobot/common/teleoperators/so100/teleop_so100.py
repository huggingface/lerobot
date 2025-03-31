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
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
    TorqueMode,
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
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
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
        if not self.is_calibrated:
            self.calibrate()

        self.configure()

    def configure(self) -> None:
        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration.
        for name in self.arm.names:
            self.arm.write("Torque_Enable", name, TorqueMode.DISABLED.value)

    @property
    def is_calibrated(self) -> bool:
        motors_calibration = self.arm.get_calibration()
        return motors_calibration == self.calibration

    def calibrate(self) -> None:
        print(f"\nRunning calibration of {self.id} SO-100 teleop")
        for name in self.arm.names:
            self.arm.write("Torque_Enable", name, TorqueMode.DISABLED.value)
            self.arm.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move robot to the middle of its range of motion and press ENTER....")
        homing_offsets = self.arm.set_half_turn_homings()

        print(
            "Move all joints except 'wrist_roll' (id=5) sequentially through their entire ranges of motion."
        )
        print("Recording positions. Press ENTER to stop...")
        auto_range_motors = [name for name in self.arm.names if name != "wrist_roll"]
        ranges = self.arm.register_ranges_of_motion(auto_range_motors)
        ranges["wrist_roll"] = {"min": 0, "max": 4095}

        self.calibration = {}
        for name, motor in self.arm.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=ranges[name]["min"],
                range_max=ranges[name]["max"],
            )

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def get_action(self) -> dict[str, float]:
        """The returned action does not have a batch dimension."""
        # Read arm position
        before_read_t = time.perf_counter()
        action = self.arm.sync_read("Present_Position")
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        self.arm.disconnect()
