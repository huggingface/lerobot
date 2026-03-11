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
from inspect import signature
from typing import Any

from ...keyboard import KeyboardRoverTeleop
from ...teleoperator import Teleoperator
from ..sub_teleoperators.panthera_keyboard_ee import PantheraKeyboardEETeleop, PantheraKeyboardEETeleopConfig
from ..sub_teleoperators.xlerobot_mount_gamepad.teleop import XLeRobotMountGamepadTeleop
from .config import PantheraArmKeyboardTeleopConfig, XLeRobotKeyboardCompositeConfig


class XLeRobotKeyboardComposite(Teleoperator):
    """Composite teleoperator for XLeRobot with keyboard base control and optional mount gamepad."""

    config_class = XLeRobotKeyboardCompositeConfig
    name = "xlerobot_keyboard_composite"

    def __init__(self, config: XLeRobotKeyboardCompositeConfig):
        self.config = config
        super().__init__(config)
        self.arm_teleop = self._make_arm_teleop(config.arm_config)
        self.base_teleop = KeyboardRoverTeleop(config.base_config) if config.base_config else None
        self.mount_teleop = XLeRobotMountGamepadTeleop(config.mount_config) if config.mount_config else None

    @staticmethod
    def _make_arm_teleop(config: PantheraArmKeyboardTeleopConfig | None) -> Teleoperator | None:
        if config is None:
            return None
        if isinstance(config, PantheraKeyboardEETeleopConfig):
            return PantheraKeyboardEETeleop(config)
        raise TypeError(f"Unsupported Panthera arm teleop config type: {type(config).__name__}")

    def _iter_active_teleops(self) -> tuple[Teleoperator, ...]:
        return tuple(tp for tp in (self.arm_teleop, self.base_teleop, self.mount_teleop) if tp is not None)

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for teleop in self._iter_active_teleops():
            if teleop is self.arm_teleop:
                prefix = self._arm_prefix()
                features.update({f"{prefix}{key}": value for key, value in teleop.action_features.items()})
            elif teleop is self.base_teleop:
                features.update({"x.vel": float, "theta.vel": float})
            else:
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
            self._connect_teleop(teleop, calibrate=calibrate)
        for teleop in self._iter_active_teleops():
            if not teleop.is_connected:
                raise RuntimeError(
                    f"{type(teleop).__name__} is unavailable: no active input listener. "
                    "On Linux, this usually means `pynput` could not access a DISPLAY. "
                    "When using SSH, run with X11 forwarding (e.g. `ssh -Y`) or use a non-keyboard teleop."
                )

    @staticmethod
    def _connect_teleop(teleop: Teleoperator, calibrate: bool) -> None:
        try:
            if "calibrate" in signature(teleop.connect).parameters:
                teleop.connect(calibrate=calibrate)
                return
        except (TypeError, ValueError):
            pass
        teleop.connect()

    def disconnect(self) -> None:
        for teleop in self._iter_active_teleops():
            if teleop.is_connected:
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
            if not teleop.is_connected:
                raise RuntimeError(
                    f"{type(teleop).__name__} is not connected. "
                    "Ensure a DISPLAY is available for pynput (local desktop or SSH X11 forwarding)."
                )
            teleop_action = teleop.get_action()
            if teleop is self.arm_teleop:
                action.update(self._map_arm_action(teleop_action))
            elif teleop is self.base_teleop:
                action.update(self._map_base_action(teleop_action))
            else:
                action.update(teleop_action)
        return action

    def _map_arm_action(self, action: dict[str, Any]) -> dict[str, float]:
        prefix = self._arm_prefix()
        return {f"{prefix}{key}": float(value) for key, value in action.items()}

    def _arm_prefix(self) -> str:
        if self.config.arm_side not in ("left", "right"):
            raise ValueError(f"Unsupported arm_side '{self.config.arm_side}'. Use 'left' or 'right'.")
        return f"{self.config.arm_side}_"

    @staticmethod
    def _map_base_action(action: dict[str, Any]) -> dict[str, float]:
        linear = action.get("x.vel", action.get("linear.vel", 0.0))
        angular = action.get("theta.vel", action.get("angular.vel", 0.0))
        return {"x.vel": float(linear), "theta.vel": float(angular)}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        for teleop in self._iter_active_teleops():
            teleop.send_feedback(feedback)

    @property
    def is_calibrated(self) -> bool:
        return all(getattr(teleop, "is_calibrated", True) for teleop in self._iter_active_teleops())
