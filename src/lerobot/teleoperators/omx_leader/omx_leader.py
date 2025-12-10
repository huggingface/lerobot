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

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_omx_leader import OmxLeaderConfig

logger = logging.getLogger(__name__)


class OmxLeader(Teleoperator):
    """
    - [OMX](https://github.com/ROBOTIS-GIT/open_manipulator),
        expansion, developed by Woojin Wie and Junha Cha from [ROBOTIS](https://ai.robotis.com/)
    """

    config_class = OmxLeaderConfig
    name = "omx_leader"

    def __init__(self, config: OmxLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "xl330-m288", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "xl330-m077", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        
        # OMX robots don't require calibration - use calibration file values directly
        # Don't write to motors, just use the calibration file's range_min/max values
        if self.calibration:
            self.bus.calibration = self.calibration

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # OMX robots don't require calibration - always return True
        return True

    def calibrate(self) -> None:
        if self.calibration:
            # Use calibration file values (factory values) - don't write to motors
            logger.info(f"OMX robot {self.id} uses pre-calibrated values from calibration file. No calibration needed.")
            # Ensure bus calibration is set to match the file
            self.bus.calibration = self.calibration
        else:
            logger.warning(f"No calibration file found for {self.id}. Calibration file is required for OMX robots.")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            if motor != "gripper":
                # Use 'extended position mode' for all motors except gripper, because in joint mode the servos
                # can't rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while
                # assembling the arm, you could end up with a servo with a position 0 or 4095 at a crucial
                # point
                self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        # Use 'position control current based' for gripper to be limited by the limit of the current.
        # For the follower gripper, it means it can grasp an object without forcing too much even tho,
        # its goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
        # to make it move, and it will move back to its original target position when we release the force.
        self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)
        # Set gripper's goal pos in current position mode so that we can use it as a trigger.
        self.bus.enable_torque("gripper")
        if self.is_calibrated:
            self.bus.write("Goal_Position", "gripper", self.config.gripper_open_pos)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        # Read motors that need special handling (shoulder_pan, shoulder_lift, elbow_flex)
        special_motors = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
        other_motors = [m for m in self.bus.motors.keys() if m not in special_motors]
        action = self.bus.sync_read("Present_Position", other_motors) if other_motors else {}
        
        # Read special motors without normalization to use calibration range
        for motor_name in special_motors:
            if motor_name not in self.bus.motors:
                continue
            raw_pos = self.bus.read("Present_Position", motor_name, normalize=False)
            drive_mode = self.bus.apply_drive_mode and self.bus.calibration[motor_name].drive_mode
            norm_mode = self.bus.motors[motor_name].norm_mode
            
            if motor_name == "shoulder_pan":
                # Use extended range for EXTENDED_POSITION mode
                extended_range = 2097152  # 512 turns * 4096 steps per turn
                min_ = -extended_range // 2
                max_ = extended_range // 2 - 1
            else:
                # For shoulder_lift and elbow_flex, use calibration range_min/max
                # This ensures leader and follower use the same normalization range
                min_ = self.bus.calibration[motor_name].range_min
                max_ = self.bus.calibration[motor_name].range_max
            
            if norm_mode == MotorNormMode.RANGE_M100_100:
                norm = (((raw_pos - min_) / (max_ - min_)) * 200) - 100 if max_ != min_ else 0
                action[motor_name] = -norm if drive_mode else norm
            elif norm_mode == MotorNormMode.RANGE_0_100:
                norm = ((raw_pos - min_) / (max_ - min_)) * 100 if max_ != min_ else 0
                action[motor_name] = 100 - norm if drive_mode else norm
            elif norm_mode == MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.bus.model_resolution_table[self.bus.motors[motor_name].model] - 1
                action[motor_name] = (raw_pos - mid) * 360 / max_res
            else:
                action[motor_name] = raw_pos
        
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
