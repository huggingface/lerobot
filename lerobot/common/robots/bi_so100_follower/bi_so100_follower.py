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
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_bi_so100_follower import BiSO100FollowerConfig

logger = logging.getLogger(__name__)


class BiSO100Follower(Robot):
    """
    Bimanual SO-100 Follower Arms - manages two SO-100 arms for bimanual tasks
    """

    config_class = BiSO100FollowerConfig
    name = "bi_so100_follower"

    def __init__(self, config: BiSO100FollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # Create left arm bus
        self.left_bus = FeetechMotorsBus(
            port=self.config.left_port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self._get_left_calibration(),
        )

        # Create right arm bus
        self.right_bus = FeetechMotorsBus(
            port=self.config.right_port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self._get_right_calibration(),
        )

        self.cameras = make_cameras_from_configs(config.cameras)

    def _get_left_calibration(self):
        """Load calibration for left arm from existing so100_follower calibration"""
        from pathlib import Path

        import draccus

        left_calibration_path = (
            Path("/home/*/.cache/huggingface/lerobot/calibration/robots/so100_follower")
            / f"{self.config.left_id}.json"
        )
        if left_calibration_path.exists():
            try:
                with open(left_calibration_path) as f, draccus.config_type("json"):
                    return draccus.load(dict[str, MotorCalibration], f)
            except Exception as e:
                logger.warning(f"Failed to load left arm calibration: {e}")
        return None

    def _get_right_calibration(self):
        """Load calibration for right arm from existing so100_follower calibration"""
        from pathlib import Path

        import draccus

        right_calibration_path = (
            Path("/home/*/.cache/huggingface/lerobot/calibration/robots/so100_follower")
            / f"{self.config.right_id}.json"
        )
        if right_calibration_path.exists():
            try:
                with open(right_calibration_path) as f, draccus.config_type("json"):
                    return draccus.load(dict[str, MotorCalibration], f)
            except Exception as e:
                logger.warning(f"Failed to load right arm calibration: {e}")
        return None

    @property
    def _motors_ft(self) -> dict[str, type]:
        motors_ft = {}
        for motor in self.left_bus.motors:
            motors_ft[f"left_{motor}.pos"] = float
        for motor in self.right_bus.motors:
            motors_ft[f"right_{motor}.pos"] = float
        return motors_ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_bus.is_connected
            and self.right_bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, both arms are in rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect both buses
        self.left_bus.connect()
        self.right_bus.connect()

        if not self.is_calibrated and calibrate:
            self.calibrate()

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        left_calibrated = self.left_bus.is_calibrated and self._get_left_calibration() is not None
        right_calibrated = self.right_bus.is_calibrated and self._get_right_calibration() is not None
        return left_calibrated and right_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nLoading existing calibrations for {self}")

        # Load calibrations from existing files
        left_calibration = self._get_left_calibration()
        right_calibration = self._get_right_calibration()

        if left_calibration is None:
            raise ValueError(
                f"No calibration found for left arm (ID: {self.config.left_id}). "
                "Please ensure calibration exists at /home/*/.cache/huggingface/lerobot/calibration/robots/so100_follower/"
            )

        if right_calibration is None:
            raise ValueError(
                f"No calibration found for right arm (ID: {self.config.right_id}). "
                "Please ensure calibration exists at /home/*/.cache/huggingface/lerobot/calibration/robots/so100_follower/"
            )

        # Write calibrations to both buses
        self.left_bus.write_calibration(left_calibration)
        self.right_bus.write_calibration(right_calibration)

        logger.info("Successfully loaded calibrations for both arms")

    def configure(self) -> None:
        # Configure left arm
        with self.left_bus.torque_disabled():
            self.left_bus.configure_motors()
            for motor in self.left_bus.motors:
                self.left_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.left_bus.write("P_Coefficient", motor, 8)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.left_bus.write("I_Coefficient", motor, 0)
                self.left_bus.write("D_Coefficient", motor, 32)

        # Configure right arm
        with self.right_bus.torque_disabled():
            self.right_bus.configure_motors()
            for motor in self.right_bus.motors:
                self.right_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.right_bus.write("P_Coefficient", motor, 8)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.right_bus.write("I_Coefficient", motor, 0)
                self.right_bus.write("D_Coefficient", motor, 32)

    def setup_motors(self) -> None:
        print("Setting up left arm motors:")
        for motor in reversed(self.left_bus.motors):
            input(f"Connect the controller board to the left '{motor}' motor only and press enter.")
            self.left_bus.setup_motor(motor)
            print(f"Left '{motor}' motor id set to {self.left_bus.motors[motor].id}")

        print("Setting up right arm motors:")
        for motor in reversed(self.right_bus.motors):
            input(f"Connect the controller board to the right '{motor}' motor only and press enter.")
            self.right_bus.setup_motor(motor)
            print(f"Right '{motor}' motor id set to {self.right_bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm positions
        start = time.perf_counter()
        left_obs_dict = self.left_bus.sync_read("Present_Position")
        right_obs_dict = self.right_bus.sync_read("Present_Position")

        # Combine observations with prefixes
        obs_dict = {}
        for motor, val in left_obs_dict.items():
            obs_dict[f"left_{motor}.pos"] = val
        for motor, val in right_obs_dict.items():
            obs_dict[f"right_{motor}.pos"] = val

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command both arms to move to target joint configurations.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Split actions by arm
        left_goal_pos = {}
        right_goal_pos = {}

        for key, val in action.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                if motor_name.startswith("left_"):
                    left_goal_pos[motor_name.replace("left_", "")] = val
                elif motor_name.startswith("right_"):
                    right_goal_pos[motor_name.replace("right_", "")] = val

        # Cap goal position when too far away from present position for both arms
        if self.config.max_relative_target is not None:
            left_present_pos = self.left_bus.sync_read("Present_Position")
            right_present_pos = self.right_bus.sync_read("Present_Position")

            left_goal_present_pos = {
                key: (g_pos, left_present_pos[key]) for key, g_pos in left_goal_pos.items()
            }
            right_goal_present_pos = {
                key: (g_pos, right_present_pos[key]) for key, g_pos in right_goal_pos.items()
            }

            left_goal_pos = ensure_safe_goal_position(left_goal_present_pos, self.config.max_relative_target)
            right_goal_pos = ensure_safe_goal_position(
                right_goal_present_pos, self.config.max_relative_target
            )

        # Send goal positions to both arms
        if left_goal_pos:
            self.left_bus.sync_write("Goal_Position", left_goal_pos)
        if right_goal_pos:
            self.right_bus.sync_write("Goal_Position", right_goal_pos)

        # Return the action that was actually sent
        sent_action = {}
        for motor, val in left_goal_pos.items():
            sent_action[f"left_{motor}.pos"] = val
        for motor, val in right_goal_pos.items():
            sent_action[f"right_{motor}.pos"] = val

        return sent_action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.left_bus.disconnect(self.config.disable_torque_on_disconnect)
        self.right_bus.disconnect(self.config.disable_torque_on_disconnect)

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
