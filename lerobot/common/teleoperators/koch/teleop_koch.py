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
    TorqueMode,
)

from ..teleoperator import Teleoperator
from .configuration_koch import KochTeleopConfig


class KochTeleop(Teleoperator):
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
        expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    """

    config_class = KochTeleopConfig
    name = "koch"

    def __init__(self, config: KochTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.arm = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "xl330-m077", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "xl330-m077", MotorNormMode.RANGE_0_100),
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
        for name in self.arm.names:
            # Torque must be deactivated to change values in the motors' EEPROM area
            # We assume that at configuration time, arm is in a rest position,
            # and torque can be safely disabled to run calibration.
            self.arm.write("Torque_Enable", name, TorqueMode.DISABLED.value)
            if name != "gripper":
                # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
                # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while
                # assembling the arm, you could end up with a servo with a position 0 or 4095 at a crucial
                # point
                self.arm.write("Operating_Mode", name, OperatingMode.EXTENDED_POSITION.value)

        # Use 'position control current based' for gripper to be limited by the limit of the current.
        # For the follower gripper, it means it can grasp an object without forcing too much even tho,
        # its goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
        # to make it move, and it will move back to its original target position when we release the force.
        self.arm.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
        # Set gripper's goal pos in current position mode so that we can use it as a trigger.
        self.arm.write("Torque_Enable", "gripper", TorqueMode.ENABLED.value)
        self.arm.write("Goal_Position", "gripper", self.config.gripper_open_pos)

    @property
    def is_calibrated(self) -> bool:
        motors_calibration = self.arm.get_calibration()
        return motors_calibration == self.calibration

    def calibrate(self) -> None:
        print(f"\nRunning calibration of {self.id} Koch teleop")
        for name in self.arm.names:
            self.arm.write("Torque_Enable", name, TorqueMode.DISABLED.value)
            self.arm.write("Operating_Mode", name, OperatingMode.EXTENDED_POSITION.value)

        self.arm.write("Drive_Mode", "elbow_flex", DriveMode.INVERTED.value)
        drive_modes = {name: 1 if name == "elbow_flex" else 0 for name in self.arm.names}

        input("Move robot to the middle of its range of motion and press ENTER....")
        homing_offsets = self.arm.set_half_turn_homings()

        fixed_range = ["shoulder_pan", "wrist_roll"]
        auto_range_motors = [name for name in self.arm.names if name not in fixed_range]
        print(
            "Move all joints except 'wrist_roll' (id=5) sequentially through their entire ranges of motion."
        )
        print("Recording positions. Press ENTER to stop...")
        ranges = self.arm.register_ranges_of_motion(auto_range_motors)
        for name in fixed_range:
            ranges[name] = {"min": 0, "max": 4095}

        self.calibration = {}
        for name, motor in self.arm.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=drive_modes[name],
                homing_offset=homing_offsets[name],
                range_min=ranges[name]["min"],
                range_max=ranges[name]["max"],
            )

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def get_action(self) -> dict[str, float]:
        """The returned action does not have a batch dimension."""
        # Read arm position
        start_time = time.perf_counter()
        action = self.arm.sync_read("Present_Position")
        self.logs["read_pos_dt_s"] = time.perf_counter() - start_time

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        self.arm.disconnect()
