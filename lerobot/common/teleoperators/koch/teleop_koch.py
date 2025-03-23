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
from lerobot.common.motors import CalibrationMode, Motor, TorqueMode
from lerobot.common.motors.dynamixel import (
    DynamixelMotorsBus,
    run_arm_calibration,
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
                "shoulder_pan": Motor(1, "xl330-m077", CalibrationMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "xl330-m077", CalibrationMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "xl330-m077", CalibrationMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "xl330-m077", CalibrationMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "xl330-m077", CalibrationMode.RANGE_M100_100),
                "gripper": Motor(6, "xl330-m077", CalibrationMode.RANGE_0_100),
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

        # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
        # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
        # you could end up with a servo with a position 0 or 4095 at a crucial point See [
        # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
        all_motors_except_gripper = [name for name in self.arm.motor_names if name != "gripper"]
        if len(all_motors_except_gripper) > 0:
            # 4 corresponds to Extended Position on Koch motors
            self.arm.write("Operating_Mode", 4, all_motors_except_gripper)

        # Use 'position control current based' for gripper to be limited by the limit of the current.
        # For the follower gripper, it means it can grasp an object without forcing too much even tho,
        # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
        # to make it move, and it will move back to its original target position when we release the force.
        # 5 corresponds to Current Controlled Position on Koch gripper motors "xl330-m077, xl330-m288"
        self.arm.write("Operating_Mode", 5, "gripper")

        # Enable torque on the gripper and move it to 45 degrees so that we can use it as a trigger.
        logging.info("Activating torque.")
        self.arm.write("Torque_Enable", TorqueMode.ENABLED.value, "gripper")
        self.arm.write("Goal_Position", self.config.gripper_open_degree, "gripper")

        # Check arm can be read
        self.arm.read("Present_Position")

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
