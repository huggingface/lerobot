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
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_koch_leader import KochLeaderConfig

logger = logging.getLogger(__name__)


class KochLeader(Teleoperator):
    """
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow
        expansion, developed by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    """

    config_class = KochLeaderConfig
    name = "koch_leader"

    def __init__(self, config: KochLeaderConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl330-m077", norm_mode_body),
                "shoulder_lift": Motor(2, "xl330-m077", norm_mode_body),
                "elbow_flex": Motor(3, "xl330-m077", norm_mode_body),
                "wrist_flex": Motor(4, "xl330-m077", norm_mode_body),
                "wrist_roll": Motor(5, "xl330-m077", norm_mode_body),
                "gripper": Motor(6, "xl330-m077", MotorNormMode.RANGE_0_100), # Always use the percentage for the grippers
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

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self._wrap_full_turn_offsets_once()
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        self.bus.disable_torque()
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            logger.info(f"Calibration exists, writing calibration file associated with the id {self.id} to the motors")
            self.bus.write_calibration(self.calibration)
            return
        logger.info(f"\nRunning calibration of {self}")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        self.bus.write("Drive_Mode", "elbow_flex", DriveMode.INVERTED.value)
        drive_modes = {motor: 1 if motor == "elbow_flex" else 0 for motor in self.bus.motors}

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"Move all joints except {full_turn_motors} sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=drive_modes[motor],
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

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

    def _wrap_full_turn_offsets_once(self) -> None:
        """
        Adjust Homing_Offset by integer multiples of resolution so Present_Position is wrapped into [0, res-1]
        for full-turn joints. This is a one-time alignment at startup; we do not modify readings afterwards.
        """
        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        # Ensure EEPROM writes are allowed
        with self.bus.torque_disabled(full_turn_motors):
            raw_by_name = self.bus.sync_read("Present_Position", normalize=False)
            current_offsets = self.bus.sync_read("Homing_Offset", normalize=False)
            for motor in full_turn_motors:
                if motor not in raw_by_name:
                    continue
                model = self.bus.motors[motor].model
                res = self.bus.model_resolution_table[model]
                present = raw_by_name[motor]
                k = present // res
                if k != 0:
                    new_offset = current_offsets[motor] - (k * res)
                    if new_offset != current_offsets[motor]:
                        self.bus.write("Homing_Offset", motor, new_offset, normalize=False)
                        logger.info(f"({self.config.id}): Wrapped offset for motor '{motor}' from {current_offsets[motor]} to {new_offset} to keep it in [0, {res-1}]")
        # Refresh in-memory calibration to reflect device state
        new_cal = self.bus.read_calibration()
        self.calibration = new_cal
        self.bus.calibration = new_cal

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def get_action_with_raw(self) -> tuple[dict[str, float], dict[str, int]]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        # Single device read in raw units, decode sign but do not normalize
        raw_by_name = self.bus.sync_read("Present_Position", normalize=False)

        # Convert raw-by-name → ids → normalize using calibration
        ids_values = {self.bus.motors[motor].id: val for motor, val in raw_by_name.items()}
        norm_by_id = self.bus._normalize(ids_values)
        norm_by_name = {self.bus._id_to_name(id_): val for id_, val in norm_by_id.items()}

        # Suffix keys with .pos for action dicts
        action_norm = {f"{motor}.pos": val for motor, val in norm_by_name.items()}
        action_raw = {f"{motor}.pos": val for motor, val in raw_by_name.items()}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action (norm+raw): {dt_ms:.1f}ms")
        return action_norm, action_raw

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
