#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.so101_7dof import (
    ACTION_KEYS,
    JOINT_NAMES,
    action_to_native_position,
    native_to_action_position,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_so101_7dof_follower import SO1017DoFFollowerConfig

logger = logging.getLogger(__name__)


class SO1017DoFFollower(Robot):
    """Seven-actuator Feetech follower accepting the reBot-compatible action contract."""

    config_class = SO1017DoFFollowerConfig
    name = "so101_7dof_follower"

    def __init__(self, config: SO1017DoFFollowerConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={
                motor: Motor(
                    config.motor_ids[motor],
                    "sts3215",
                    MotorNormMode.RANGE_0_100 if motor == "gripper" else MotorNormMode.DEGREES,
                )
                for motor in JOINT_NAMES
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_KEYS, float)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for camera_name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                features[camera_name] = (camera.height, camera.width, 3)
            if getattr(camera, "use_depth", False):
                features[f"{camera_name}_depth"] = (camera.height, camera.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(camera.is_connected for camera in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no "
                "calibration file found"
            )
            self.calibrate()

        for camera in self.cameras.values():
            camera.connect()
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their entire ranges of "
            "motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {
            motor: MotorCalibration(
                id=definition.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
            for motor, definition in self.bus.motors.items()
        }
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)
                    self.bus.write("Protection_Current", motor, 250)
                    self.bus.write("Overload_Torque", motor, 25)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def _native_positions_to_action(self, native_positions: dict[str, float]) -> dict[str, float]:
        return {
            motor: native_to_action_position(motor, float(native_positions[motor]))
            for motor in JOINT_NAMES
        }

    def _read_action_positions(self) -> dict[str, float]:
        native_positions = self.bus.sync_read("Present_Position")
        return self._native_positions_to_action(native_positions)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        positions = self._read_action_positions()
        observation: RobotObservation = {f"{motor}.pos": positions[motor] for motor in JOINT_NAMES}
        logger.debug(f"{self} read state: {(time.perf_counter() - start) * 1e3:.1f}ms")

        for camera_name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                observation[camera_name] = camera.read_latest()
            if getattr(camera, "use_depth", False):
                observation[f"{camera_name}_depth"] = camera.read_latest_depth()
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_positions = {
            key.removesuffix(".pos"): float(value)
            for key, value in action.items()
            if key.endswith(".pos")
        }

        native_goals = {
            motor: action_to_native_position(motor, goal_positions[motor])
            for motor in goal_positions
        }
        self.bus.sync_write("Goal_Position", native_goals)
        return {f"{motor}.pos": value for motor, value in goal_positions.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for camera in self.cameras.values():
            camera.disconnect()
        logger.info(f"{self} disconnected.")
