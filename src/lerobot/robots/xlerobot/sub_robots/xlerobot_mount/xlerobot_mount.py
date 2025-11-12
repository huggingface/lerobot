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

"""XLeRobot Mount - Two-motor pan/tilt neck control for camera positioning."""

import logging
from functools import cached_property
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ....robot import Robot
from .config import XLeRobotMountConfig

logger = logging.getLogger(__name__)


class XLeRobotMount(Robot):
    config_class = XLeRobotMountConfig
    name = "xlerobot_mount"

    def __init__(self, config: XLeRobotMountConfig):
        super().__init__(config)
        self.config = config

        # Initialize motor bus with pan and tilt servos
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "mount_pan": Motor(
                    id=self.config.pan_motor_id,
                    model=self.config.motor_model,
                    norm_mode=MotorNormMode.DEGREES,
                ),
                "mount_tilt": Motor(
                    id=self.config.tilt_motor_id,
                    model=self.config.motor_model,
                    norm_mode=MotorNormMode.DEGREES,
                ),
            },
            calibration=self.calibration,
        )

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return {
            self.config.pan_key: float,
            self.config.tilt_key: float,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            self.config.pan_key: float,
            self.config.tilt_key: float,
        }

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True, handshake: bool | None = None) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()

        if calibrate and not self.is_calibrated:
            self.calibrate()

        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info("Calibrating %s", self)

        if self.calibration:
            user_input = input(
                f"Press ENTER to use existing calibration (ID: {self.id}), "
                f"or type 'c' and press ENTER to run new calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info("Using existing calibration file")
                self.bus.write_calibration(self.calibration)
                return

        # Run new calibration
        logger.info("Setting up new calibration for mount motors")
        self.bus.disable_torque()

        # Set motors to position mode for calibration
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move the mount to center position (pan=0°, tilt=0°) and press ENTER...")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move both pan and tilt joints through their full range of motion.\n"
            "Recording positions. Press ENTER when done..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        # Create calibration for each motor
        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info("Calibration saved to %s", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                # Set position control mode
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Adjust PID coefficients for smooth motion (lower P to reduce shakiness)
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")

        self.bus.disable_torque()
        self.bus.disconnect()
        logger.info("%s disconnected.", self)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")

        positions = self.bus.sync_read("Present_Position", ["mount_pan", "mount_tilt"])
        return {
            self.config.pan_key: positions["mount_pan"],
            self.config.tilt_key: positions["mount_tilt"],
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")

        # Enforce safety limits
        pan_cmd = max(self.config.pan_range[0], min(self.config.pan_range[1], action[self.config.pan_key]))
        tilt_cmd = max(
            self.config.tilt_range[0], min(self.config.tilt_range[1], action[self.config.tilt_key])
        )

        # Send commands to motors
        goal_positions = {
            "mount_pan": pan_cmd,
            "mount_tilt": tilt_cmd,
        }
        self.bus.sync_write("Goal_Position", goal_positions)

        return {
            self.config.pan_key: pan_cmd,
            self.config.tilt_key: tilt_cmd,
        }
