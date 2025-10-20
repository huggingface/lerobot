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
import time
import numpy as np
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

from ..robot import Robot
from .config_bi_so101_follower import BiSO101FollowerConfig

logger = logging.getLogger(__name__)


class BiSO101Follower(Robot):
    """
    Implementation based on src/lerobot/robots/bi_so100_follower/bi_so100_follower.py
    [Bimanual SO-101 Follower Arms (in the same repository as so 100)](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = BiSO101FollowerConfig
    name = "bi_so101_follower"

    def __init__(self, config: BiSO101FollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SO101FollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )

        right_arm_config = SO101FollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )

        self.left_arm = SO101Follower(left_arm_config)
        self.right_arm = SO101Follower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._last_camera_obs: dict[str, Any] = {
            cam_key: self._make_blank_camera_obs(cam_key) for cam_key in self.cameras
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    def _make_blank_camera_obs(self, cam_key: str) -> np.ndarray:
        cam_config = self.config.cameras.get(cam_key)
        height = getattr(cam_config, "height", None) or 1
        width = getattr(cam_config, "width", None) or 1
        return np.zeros((height, width, 3), dtype=np.uint8)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

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

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Add "left_" prefix
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # Add "right_" prefix
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            try:
                obs = cam.async_read(timeout_ms=1000)
            except Exception as exc:  # noqa: BLE001 keep full traceback for debugging
                dt_ms = (time.perf_counter() - start) * 1e3
                obs = self._last_camera_obs.get(cam_key)
                if obs is None:
                    obs = self._make_blank_camera_obs(cam_key)
                    self._last_camera_obs[cam_key] = obs
                    logger.warning(
                        "Failed to read camera %s (%s); using blank observation. Took %.1fms",
                        cam_key,
                        exc,
                        dt_ms,
                    )
                else:
                    logger.warning(
                        "Failed to read camera %s (%s); reusing cached observation. Took %.1fms",
                        cam_key,
                        exc,
                        dt_ms,
                    )
            else:
                dt_ms = (time.perf_counter() - start) * 1e3
                self._last_camera_obs[cam_key] = obs
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            obs_dict[cam_key] = obs

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()
