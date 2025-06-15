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
from ..so101_follower.config_so101_follower import SO101FollowerConfig
from ..so101_follower.so101_follower import SO101Follower

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

        left_config = SO101FollowerConfig(
            port=self.config.left_arm_port,
            id=self.config.left_arm_id,
            use_degrees=config.use_degrees,
            max_relative_target=config.left_arm_max_relative_target,
            disable_torque_on_disconnect=config.disable_torque_on_disconnect,
        )
        self.left_arm = SO101Follower(left_config)

        right_config = SO101FollowerConfig(
            port=self.config.right_arm_port,
            id=self.config.right_arm_id,
            use_degrees=config.use_degrees,
            max_relative_target=config.right_arm_max_relative_target,
            disable_torque_on_disconnect=config.disable_torque_on_disconnect,
        )
        self.right_arm = SO101Follower(right_config)

        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)

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
        left_obs = {f"left_{key}": value for key, value in self.left_arm.observation_features.items()}
        right_obs = {f"right_{key}": value for key, value in self.right_arm.observation_features.items()}
        return {**left_obs, **right_obs, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features for both arms"""
        left_act = {f"left_{key}": value for key, value in self.left_arm.action_features.items()}
        right_act = {f"right_{key}": value for key, value in self.right_arm.action_features.items()}
        return {**left_act, **right_act}

    @property
    def is_connected(self) -> bool:
        """Check if both arms and all cameras are connected"""
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect both arms and all cameras.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if both arms are calibrated"""
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        """Calibrate both arms sequentially"""
        logger.info(f"\nRunning calibration of {self}")

        # Calibrate left arm
        logger.info("Calibrating LEFT arm...")
        self.left_arm.calibrate()

        # Calibrate right arm
        logger.info("Calibrating RIGHT arm...")
        self.right_arm.calibrate()

        print("Bimanual calibration complete.")

    def configure(self) -> None:
        """Configure both arms"""
        self.left_arm.configure()
        self.right_arm.configure()

    def get_observation(self) -> dict[str, Any]:
        """Get observations from both arms and all cameras"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # Capture images from all cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Separate actions for left and right arms
        left_action = {
            key.removeprefix("left_"): val for key, val in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): val for key, val in action.items() if key.startswith("right_")
        }

        # Send actions to both arms
        left_sent = self.left_arm.send_action(left_action)
        right_sent = self.right_arm.send_action(right_action)

        # Combine and re-prefix results
        sent_action = {f"left_{key}": value for key, value in left_sent.items()}
        sent_action.update({f"right_{key}": value for key, value in right_sent.items()})
        return sent_action

    def disconnect(self) -> None:
        """Disconnect both arms and all cameras"""
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        # Disconnect cameras
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()

        logger.info(f"{self} disconnected.")