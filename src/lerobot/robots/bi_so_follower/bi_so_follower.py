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

import logging
from functools import cached_property

from lerobot.cameras.opencv import OpenCVCamera  # or IntelRealSenseCamera etc.
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..so_follower import SOFollower, SOFollowerRobotConfig
from .config_bi_so_follower import BiSOFollowerConfig

logger = logging.getLogger(__name__)


class BiSOFollower(Robot):
    """
    [Bimanual SO Follower Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio

    Cameras can be attached in three ways:
      - per-arm:  left_arm_config.cameras / right_arm_config.cameras
      - global:   BiSOFollowerConfig.top_cameras   ← for overhead/external cameras
                  that don't belong to either arm
    """

    config_class = BiSOFollowerConfig
    name = "bi_so_follower"

    def __init__(self, config: BiSOFollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SOFollowerRobotConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            disable_torque_on_disconnect=config.left_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.left_arm_config.max_relative_target,
            use_degrees=config.left_arm_config.use_degrees,
            cameras=config.left_arm_config.cameras,
        )

        right_arm_config = SOFollowerRobotConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            disable_torque_on_disconnect=config.right_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.right_arm_config.max_relative_target,
            use_degrees=config.right_arm_config.use_degrees,
            cameras=config.right_arm_config.cameras,
        )

        self.left_arm = SOFollower(left_arm_config)
        self.right_arm = SOFollower(right_arm_config)

        # ── Top / global cameras ──────────────────────────────────────────────
        # Instantiate each camera from the configs provided in BiSOFollowerConfig.
        # These are NOT prefixed with left_/right_ — their keys are used as-is.
        self.top_cameras: dict = {
            name: OpenCVCamera(cam_cfg)
            for name, cam_cfg in (config.top_cameras or {}).items()
        }

        # Expose a unified cameras dict for compatibility with the rest of the codebase.
        # Order: left-arm cameras, right-arm cameras, then global top cameras.
        self.cameras = {
            **self.left_arm.cameras,
            **self.right_arm.cameras,
            **self.top_cameras,
        }

    # ── Feature descriptors ───────────────────────────────────────────────────

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v  for k, v in self.left_arm._motors_ft.items()},
            **{f"right_{k}": v for k, v in self.right_arm._motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # Per-arm camera features keep their left_/right_ prefixes (set by SOFollower).
        arm_camera_ft = {
            **{f"left_{k}": v  for k, v in self.left_arm._cameras_ft.items()},
            **{f"right_{k}": v for k, v in self.right_arm._cameras_ft.items()},
        }

        # ── Top camera features ───────────────────────────────────────────────
        # Shape is read directly from the camera config so it matches
        # the tensors returned by async_read() / read().
        top_camera_ft = {}
        for name, cam in self.top_cameras.items():
            top_camera_ft[f"observation.images.{name}"] = (
                cam.config.height,
                cam.config.width,
                cam.config.channels if hasattr(cam.config, "channels") else 3,
            )

        return {**arm_camera_ft, **top_camera_ft}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ── Connection lifecycle ──────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        top_connected = all(cam.is_connected for cam in self.top_cameras.values())
        return self.left_arm.is_connected and self.right_arm.is_connected and top_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        # ── Connect top cameras ───────────────────────────────────────────────
        for name, cam in self.top_cameras.items():
            cam.connect()
            logger.info(f"Top camera '{name}' connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    # ── Observation / action ──────────────────────────────────────────────────

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict = {}

        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # ── Read top cameras ──────────────────────────────────────────────────
        # Keys match those registered in _cameras_ft so the dataset writer
        # picks them up automatically.
        for name, cam in self.top_cameras.items():
            obs_dict[f"observation.images.{name}"] = cam.async_read(timeout_ms=200)

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action = {
            key.removeprefix("left_"): value
            for key, value in action.items()
            if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value
            for key, value in action.items()
            if key.startswith("right_")
        }

        sent_left  = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)

        return {
            **{f"left_{k}":  v for k, v in sent_left.items()},
            **{f"right_{k}": v for k, v in sent_right.items()},
        }

    @check_if_not_connected
    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        # ── Disconnect top cameras ────────────────────────────────────────────
        for name, cam in self.top_cameras.items():
            cam.disconnect()
            logger.info(f"Top camera '{name}' disconnected.")