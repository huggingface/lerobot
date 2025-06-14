#!/usr/bin/env python

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

import json
import logging
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_bimanual_parcelot import BimanualParcelotConfig

logger = logging.getLogger(__name__)


class BimanualParcelot(Robot):
    """
    Bimanual Parcelot Robot consisting of two SO-101 Follower Arms and three cameras.
    Designed for dual-arm manipulation tasks.
    """

    config_class = BimanualParcelotConfig
    name = "bimanual_parcelot"

    def __init__(self, config: BimanualParcelotConfig):
        super().__init__(config)
        self.config = config
        
        # If individual arm IDs are provided, load their calibrations and merge them
        if config.left_arm_id and config.right_arm_id:
            self._load_and_merge_arm_calibrations()

        # Motor normalization mode
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        
        # Initialize left arm bus
        self.left_bus = FeetechMotorsBus(
            port=self.config.left_arm_port,
            motors={
                "left_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        
        # Initialize right arm bus
        self.right_bus = FeetechMotorsBus(
            port=self.config.right_arm_port,
            motors={
                "right_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "right_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "right_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "right_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "right_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "right_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        
        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    def _load_and_merge_arm_calibrations(self):
        """Load individual arm calibrations and merge them with appropriate key prefixes."""
        self.calibration = {}
        single_arm_calibration_dir = HF_LEROBOT_CALIBRATION / ROBOTS / "so101_follower"

        # Load and map left arm calibration
        left_calib_path = single_arm_calibration_dir / f"{self.config.left_arm_id}.json"
        if left_calib_path.is_file():
            with open(left_calib_path) as f:
                left_calib_data = json.load(f)
            for key, value in left_calib_data.items():
                self.calibration[f"left_{key}"] = MotorCalibration(**value)
        else:
            logger.warning(f"Calibration file not found for left arm: {left_calib_path}")

        # Load and map right arm calibration
        right_calib_path = single_arm_calibration_dir / f"{self.config.right_arm_id}.json"
        if right_calib_path.is_file():
            with open(right_calib_path) as f:
                right_calib_data = json.load(f)
            for key, value in right_calib_data.items():
                self.calibration[f"right_{key}"] = MotorCalibration(**value)
        else:
            logger.warning(f"Calibration file not found for right arm: {right_calib_path}")

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor feature types for both arms"""
        left_motors = {f"{motor}.pos": float for motor in self.left_bus.motors}
        right_motors = {f"{motor}.pos": float for motor in self.right_bus.motors}
        return {**left_motors, **right_motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera feature types"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from both arms and all cameras"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features for both arms"""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if both arms and all cameras are connected"""
        return (
            self.left_bus.is_connected 
            and self.right_bus.is_connected 
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect both arms and all cameras.
        Both arms are assumed to be pre-calibrated, so no calibration is performed.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect both buses
        self.left_bus.connect()
        self.right_bus.connect()
        
        # Note: Calibration is skipped - arms are assumed to be pre-calibrated

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        # Configure both arms
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if both arms are calibrated"""
        return self.left_bus.is_calibrated and self.right_bus.is_calibrated

    def calibrate(self) -> None:
        """Calibrate both arms sequentially"""
        logger.info(f"\nRunning calibration of {self}")
        self.calibration = {}

        # Calibrate left arm
        logger.info("Calibrating LEFT arm...")
        self._calibrate_arm(self.left_bus, "left")

        # Calibrate right arm
        logger.info("Calibrating RIGHT arm...")
        self._calibrate_arm(self.right_bus, "right")

        # Save separate calibration files for each arm
        self._save_individual_arm_calibrations()
        print("Individual arm calibration files saved.")

    def _calibrate_arm(self, bus: FeetechMotorsBus, arm_prefix: str) -> None:
        """Calibrate a single arm and store calibration data with a prefix."""
        bus.disable_torque()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {arm_prefix.upper()} arm to the middle of its range of motion and press ENTER....")
        homing_offsets = bus.set_half_turn_homings()

        print(
            f"Move all {arm_prefix.upper()} arm joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = bus.record_ranges_of_motion()

        # Update calibration for this arm, adding a prefix to the keys
        for motor, m in bus.motors.items():
            motor_name_without_prefix = motor.removeprefix(f"{arm_prefix}_")
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        bus.write_calibration(self.calibration)

    def _save_individual_arm_calibrations(self):
        """Save separate JSON calibration files for each arm."""
        single_arm_calibration_dir = HF_LEROBOT_CALIBRATION / ROBOTS / "so101_follower"
        single_arm_calibration_dir.mkdir(parents=True, exist_ok=True)

        left_calib_data = {}
        right_calib_data = {}

        for key, calib_obj in self.calibration.items():
            if key.startswith("left_"):
                motor_name = key.removeprefix("left_")
                left_calib_data[motor_name] = calib_obj.to_dict()
            elif key.startswith("right_"):
                motor_name = key.removeprefix("right_")
                right_calib_data[motor_name] = calib_obj.to_dict()

        if self.config.left_arm_id:
            left_path = single_arm_calibration_dir / f"{self.config.left_arm_id}.json"
            with open(left_path, "w") as f:
                json.dump(left_calib_data, f, indent=4)
            print(f"Saved left arm calibration to {left_path}")

        if self.config.right_arm_id:
            right_path = single_arm_calibration_dir / f"{self.config.right_arm_id}.json"
            with open(right_path, "w") as f:
                json.dump(right_calib_data, f, indent=4)
            print(f"Saved right arm calibration to {right_path}")

    def configure(self) -> None:
        """Configure both arms"""
        self._configure_arm(self.left_bus)
        self._configure_arm(self.right_bus)

    def _configure_arm(self, bus: FeetechMotorsBus) -> None:
        """Configure a single arm"""
        with bus.torque_disabled():
            bus.configure_motors()
            for motor in bus.motors:
                bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                bus.write("I_Coefficient", motor, 0)
                bus.write("D_Coefficient", motor, 32)

    def get_observation(self) -> dict[str, Any]:
        """Get observations from both arms and all cameras"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Read left arm position
        start = time.perf_counter()
        left_obs = self.left_bus.sync_read("Present_Position")
        left_obs = {f"{motor}.pos": val for motor, val in left_obs.items()}
        obs_dict.update(left_obs)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read left arm state: {dt_ms:.1f}ms")

        # Read right arm position
        start = time.perf_counter()
        right_obs = self.right_bus.sync_read("Present_Position")
        right_obs = {f"{motor}.pos": val for motor, val in right_obs.items()}
        obs_dict.update(right_obs)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read right arm state: {dt_ms:.1f}ms")

        # Capture images from all cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Command both arms to move to target joint configurations.
        
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

        # Separate actions for left and right arms
        left_action = {
            key.removesuffix(".pos"): val 
            for key, val in action.items() 
            if key.startswith("left_") and key.endswith(".pos")
        }
        right_action = {
            key.removesuffix(".pos"): val 
            for key, val in action.items() 
            if key.startswith("right_") and key.endswith(".pos")
        }

        # Send actions to both arms
        left_sent = self._send_arm_action(
            self.left_bus, left_action, self.config.left_arm_max_relative_target
        )
        right_sent = self._send_arm_action(
            self.right_bus, right_action, self.config.right_arm_max_relative_target
        )

        # Combine results
        return {**left_sent, **right_sent}

    def _send_arm_action(
        self, bus: FeetechMotorsBus, goal_pos: dict[str, float], max_relative_target: int | None
    ) -> dict[str, float]:
        """Send action to a single arm"""
        # Cap goal position when too far away from present position.
        if max_relative_target is not None:
            present_pos = bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, max_relative_target)

        # Send goal position to the arm
        bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self) -> None:
        """Disconnect both arms and all cameras"""
        if self.config.disable_torque_on_disconnect:
            if self.left_bus.is_connected:
                self.left_bus.disable_torque()
            if self.right_bus.is_connected:
                self.right_bus.disable_torque()

        # Disconnect buses
        if self.left_bus.is_connected:
            self.left_bus.disconnect()
        if self.right_bus.is_connected:
            self.right_bus.disconnect()

        # Disconnect cameras
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

        logger.info(f"{self} disconnected.")

    def _save_calibration(self) -> None:
        """Save calibration to a json file."""
        calibration_dict = {key: val.to_dict() for key, val in self.calibration.items()}
        with open(self.calibration_fpath, "w") as f:
            json.dump(calibration_dict, f, indent=4) 