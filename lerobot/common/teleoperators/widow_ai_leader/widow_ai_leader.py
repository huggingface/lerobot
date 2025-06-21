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
from lerobot.common.motors.trossen import TrossenArmDriver

from ..teleoperator import Teleoperator
from .config_widow_ai_leader import WidowAILeaderConfig

logger = logging.getLogger(__name__)


class WidowAILeader(Teleoperator):
    """
    Trossen Widow AI Leader Arm for teleoperation.
    
    This teleoperator reads joint positions from a Trossen Widow AI arm
    and can optionally receive force feedback from the follower arm.
    """

    config_class = WidowAILeaderConfig
    name = "widow_ai_leader"

    def __init__(self, config: WidowAILeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = TrossenArmDriver(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "4340", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "4340", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "4340", MotorNormMode.DEGREES),
                "wrist_1": Motor(4, "4310", MotorNormMode.DEGREES),
                "wrist_2": Motor(5, "4310", MotorNormMode.DEGREES),
                "wrist_3": Motor(6, "4310", MotorNormMode.DEGREES),
                "gripper": Motor(7, "4310", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
            model=self.config.model,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        # Can receive force feedback from the follower arm
        return {f"{motor}.force": float for motor in self.bus.motors}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        # For Trossen arms, calibration is typically pre-configured
        # but we can still set up homing offsets if needed
        self.bus.disable_torque()
        
        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

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
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        # For Trossen arms, torque is enabled by default in position mode
        # No need to set specific PID values as they're pre-configured
        self.bus.configure_motors()

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        """Get the current joint positions from the leader arm."""
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Send force feedback to the leader arm.
        
        This can be used to provide haptic feedback from the follower arm's
        interaction with the environment.
        """
        # Extract force feedback for each motor
        force_feedback = {}
        for motor in self.bus.motors:
            force_key = f"{motor}.force"
            if force_key in feedback:
                force_feedback[motor] = feedback[force_key]
        
        if force_feedback:
            # Send external efforts to the Trossen arm for force feedback
            self.bus.sync_write("External_Efforts", force_feedback, normalize=False)
            logger.debug(f"{self} sent force feedback: {force_feedback}")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
