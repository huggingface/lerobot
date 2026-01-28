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

# XLeRobot integration based on
#
#   https://github.com/Astera-org/brainbot
#   https://github.com/Vector-Wangel/XLeRobot
#   https://github.com/bingogome/lerobot

from __future__ import annotations

from contextlib import suppress
from functools import cached_property
from typing import Any

from ...bi_so101_leader.bi_so101_leader import BiSO101Leader
from ...teleoperator import Teleoperator
from ..sub_teleoperators.biwheel_gamepad.teleop_biwheel_gamepad import BiwheelGamepadTeleop
from ..sub_teleoperators.lekiwi_base_gamepad.teleop_lekiwi_base_gamepad import LeKiwiBaseTeleop
from ..sub_teleoperators.xlerobot_mount_gamepad.teleop import XLeRobotMountGamepadTeleop
from .config import XLeRobotDefaultCompositeConfig


class XLeRobotDefaultComposite(Teleoperator):
    """Composite teleoperator for XLeRobot combining leader arms with gamepad inputs.

    This teleoperator combines three input methods:
    - BiSO101Leader: Leader arms for controlling follower arms
    - LeKiwiBaseTeleop: Xbox gamepad left stick for base movement
    - XLeRobotMountGamepadTeleop: Xbox gamepad right stick for mount pan/tilt

    All three inputs are merged into a single action dictionary that controls
    the complete XLeRobot system.
    """

    config_class = XLeRobotDefaultCompositeConfig
    name = "xlerobot_default_composite"

    def __init__(self, config: XLeRobotDefaultCompositeConfig):
        self.config = config
        super().__init__(config)
        self.arm_teleop = BiSO101Leader(config.arms_config) if config.arms_config else None
        self.base_teleop = self._build_base_teleop()
        self.mount_teleop = self._build_mount_teleop()

    def _build_base_teleop(self) -> Teleoperator | None:
        base_config = getattr(self.config, "base_config", None)
        if base_config is None:
            return None
        base_type = getattr(self.config, "base_type", None) or XLeRobotDefaultCompositeConfig.BASE_TYPE_LEKIWI
        if base_type == XLeRobotDefaultCompositeConfig.BASE_TYPE_LEKIWI:
            return LeKiwiBaseTeleop(base_config)
        if base_type == XLeRobotDefaultCompositeConfig.BASE_TYPE_BIWHEEL:
            return BiwheelGamepadTeleop(base_config)
        raise ValueError(f"Unsupported base teleoperator type: {base_type}")

    def _build_mount_teleop(self) -> Teleoperator | None:
        mount_config = getattr(self.config, "mount_config", None)
        if mount_config is None:
            return None
        return XLeRobotMountGamepadTeleop(mount_config)

    def _iter_active_teleops(self) -> tuple[Teleoperator, ...]:
        return tuple(tp for tp in (self.arm_teleop, self.base_teleop, self.mount_teleop) if tp is not None)

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for teleop in self._iter_active_teleops():
            features.update(teleop.action_features)
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for teleop in self._iter_active_teleops():
            features.update(teleop.feedback_features)
        return features

    @property
    def is_connected(self) -> bool:
        return all(teleop.is_connected for teleop in self._iter_active_teleops())

    def connect(self, calibrate: bool = True) -> None:
        for teleop in self._iter_active_teleops():
            teleop.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        for teleop in self._iter_active_teleops():
            teleop.disconnect()

    def calibrate(self) -> None:
        for teleop in self._iter_active_teleops():
            teleop.calibrate()

    def configure(self) -> None:
        for teleop in self._iter_active_teleops():
            teleop.configure()

    def on_observation(self, robot_obs: dict[str, Any]) -> None:
        if self.mount_teleop and hasattr(self.mount_teleop, "on_observation"):
            with suppress(Exception):
                self.mount_teleop.on_observation(robot_obs)

    def get_action(self) -> dict[str, float]:
        action: dict[str, float] = {}
        for teleop in self._iter_active_teleops():
            action.update(teleop.get_action())
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        for teleop in self._iter_active_teleops():
            teleop.send_feedback(feedback)

    @property
    def is_calibrated(self) -> bool:
        return all(getattr(teleop, "is_calibrated", True) for teleop in self._iter_active_teleops())
