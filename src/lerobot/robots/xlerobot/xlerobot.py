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
from dataclasses import replace
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from ..bi_so101_follower.bi_so101_follower import BiSO101Follower
from ..lekiwi_base.lekiwi_base import LeKiwiBase
from ..robot import Robot
from ..xlerobot_mount.xlerobot_mount import XLeRobotMount
from .config_xlerobot import XLeRobotConfig
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


class XLeRobot(Robot):
    """Combined platform: bimanual SO-101 follower arms mounted on a LeKiwi base."""

    config_class = XLeRobotConfig
    name = "xlerobot"

    def __init__(self, config: XLeRobotConfig):
        super().__init__(config)
        self.config = config

        self.arms = BiSO101Follower(replace(config.arms_config))
        self.base = self._build_base_robot()
        self.mount = XLeRobotMount(replace(config.mount_config))
        self.camera_configs = dict(config.cameras)
        self.cameras = make_cameras_from_configs(self.camera_configs)
        self._last_camera_obs: dict[str, np.ndarray] = {
            name: self._make_blank_camera_obs(name) for name in self.cameras
        }

    def _build_base_robot(self) -> Robot:
        base_type = getattr(self.config, "base_type", XLeRobotConfig.BASE_TYPE_LEKIWI)
        if base_type == XLeRobotConfig.BASE_TYPE_LEKIWI:
            return LeKiwiBase(replace(self.config.base_config))
        if base_type == XLeRobotConfig.BASE_TYPE_BIWHEEL:
            # TODO: instantiate biwheel base robot once available.
            raise NotImplementedError("TODO: add biwheel base robot support.")
        raise ValueError(f"Unsupported base robot type: {base_type}")

    def _make_blank_camera_obs(self, cam_key: str) -> np.ndarray:
        cam_config = self.camera_configs.get(cam_key)
        height = getattr(cam_config, "height", None) or 1
        width = getattr(cam_config, "width", None) or 1
        return np.zeros((height, width, 3), dtype=np.uint8)

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        camera_features: dict[str, tuple[int, int, int]] = {}
        for cam_key, cam_config in self.camera_configs.items():
            height = getattr(cam_config, "height", None) or 1
            width = getattr(cam_config, "width", None) or 1
            camera_features[cam_key] = (height, width, 3)
        return camera_features

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        features.update(self.arms.observation_features)
        features.update(self.base.observation_features)
        features.update(self.mount.observation_features)
        features.update(self._cameras_ft)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        features.update(self.arms.action_features)
        features.update(self.base.action_features)
        features.update(self.mount.action_features)
        return features

    @property
    def is_connected(self) -> bool:
        return (
            self.arms.is_connected
            and self.base.is_connected
            and self.mount.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.mount.connect(calibrate=calibrate)
        handshake = getattr(self.base.config, "handshake_on_connect", True)
        self.base.connect(calibrate=calibrate, handshake=handshake)
        self.arms.connect(calibrate=calibrate)
        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.arms.is_calibrated and self.base.is_calibrated and self.mount.is_calibrated

    def calibrate(self) -> None:
        logger.info("Calibrating XLeRobot components")
        self.base.calibrate()
        self.arms.calibrate()
        self.mount.calibrate()

    def configure(self) -> None:
        self.base.configure()
        self.arms.configure()
        self.mount.configure()

    def setup_motors(self) -> None:
        if hasattr(self.arms, "setup_motors"):
            self.arms.setup_motors()
        if hasattr(self.base, "setup_motors"):
            self.base.setup_motors()
        if hasattr(self.mount, "setup_motors"):
            self.mount.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("XLeRobot is not connected.")

        obs = {}
        obs.update(self.arms.get_observation())
        obs.update(self.base.get_observation())
        obs.update(self.mount.get_observation())
        for name, cam in self.cameras.items():
            try:
                frame = cam.async_read()
            except Exception as exc:
                logger.warning("Failed to read camera %s (%s); using cached frame", name, exc)
                frame = self._last_camera_obs.get(name)
                if frame is None:
                    frame = self._make_blank_camera_obs(name)
                    self._last_camera_obs[name] = frame
            else:
                self._last_camera_obs[name] = frame
            obs[name] = frame
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("XLeRobot is not connected.")

        arm_prefixes = ("left_", "right_")
        arm_action = {k: v for k, v in action.items() if k.startswith(arm_prefixes)}

        base_keys = ("x.vel", "y.vel", "theta.vel")
        base_action = {}
        for key in base_keys:
            if key in action:
                base_action[key] = action[key]
            else:
                prefixed = f"base.{key}"
                if prefixed in action:
                    base_action[key] = action[prefixed]

        mount_action: dict[str, float] = {}
        mount_keys = set(self.mount.action_features.keys())
        for key in mount_keys:
            if key in action:
                mount_action[key] = action[key]

        sent_arm = self.arms.send_action(arm_action)
        sent_base = self.base.send_action(base_action)
        sent_mount = self.mount.send_action(mount_action) if mount_action else {}
        return {**sent_arm, **sent_base, **sent_mount}

    def disconnect(self) -> None:
        if self.base.is_connected:
            try:
                if hasattr(self.base, "stop_base"):
                    self.base.stop_base()
                self.base.disconnect()
            except DeviceNotConnectedError:
                logger.debug("Base already disconnected", exc_info=False)
            except Exception:
                logger.warning("Failed to disconnect base", exc_info=True)
        if self.mount.is_connected:
            try:
                self.mount.disconnect()
            except DeviceNotConnectedError:
                logger.debug("Mount already disconnected", exc_info=False)
        if self.arms.is_connected:
            try:
                self.arms.disconnect()
            except DeviceNotConnectedError:
                logger.debug("Arms already disconnected", exc_info=False)
            except Exception:
                logger.warning("Failed to disconnect arms", exc_info=True)
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except DeviceNotConnectedError:
                logger.debug("Camera %s not connected during disconnect", cam, exc_info=False)
            except Exception:
                logger.warning("Failed to disconnect camera", exc_info=True)
