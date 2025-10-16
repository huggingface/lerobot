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

from lerobot.cameras.utils import make_cameras_from_configs
from ..bi_so101_follower.bi_so101_follower import BiSO101Follower
from ..lekiwi_base.lekiwi_base import LeKiwiBase
from ..robot import Robot
from ..xlerobot_mount.xlerobot_mount import XLeRobotMount
from .config_xlerobot import XLerobotConfig
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


class XLerobot(Robot):
    """Combined platform: bimanual SO-101 follower arms mounted on a LeKiwi base."""

    config_class = XLerobotConfig
    name = "xlerobot"

    def __init__(self, config: XLerobotConfig):
        super().__init__(config)
        self.config = config

        self.arms = BiSO101Follower(replace(config.arms_config))
        self.base = LeKiwiBase(replace(config.base_config))
        self.mount = XLeRobotMount(replace(config.mount_config))
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        features.update(self.arms.observation_features)
        features.update(self.base.observation_features)
        features.update(self.mount.observation_features)
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
        return self.arms.is_connected and self.base.is_connected and self.mount.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.mount.connect(calibrate=calibrate)
        handshake = getattr(self.base.config, "handshake_on_connect", True)
        self.base.connect(calibrate=calibrate, handshake=handshake)
        self.arms.connect(calibrate=calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.arms.is_calibrated and self.base.is_calibrated and self.mount.is_calibrated

    def calibrate(self) -> None:
        logger.info("Calibrating XLerobot components")
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
            raise DeviceNotConnectedError("XLerobot is not connected.")

        obs = {}
        obs.update(self.arms.get_observation())
        obs.update(self.base.get_observation())
        obs.update(self.mount.get_observation())
        for name, cam in self.cameras.items():
            try:
                obs[name] = cam.async_read()
            except Exception:
                logger.warning("Failed to read camera %s", name, exc_info=True)
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("XLerobot is not connected.")

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
            if hasattr(self.base, "stop_base"):
                self.base.stop_base()
            self.base.disconnect()
        if self.mount.is_connected:
            self.mount.disconnect()
        if self.arms.is_connected:
            self.arms.disconnect()
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                logger.warning("Failed to disconnect camera", exc_info=True)
