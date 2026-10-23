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

"""Prototype device composition, independent of robots and action processors."""

import logging
from collections.abc import Mapping
from typing import Any

from lerobot.lerobot_types import RobotAction, RobotObservation

from .config import TeleoperatorConfig
from .teleoperator import Teleoperator

logger = logging.getLogger(__name__)


class CompositeTeleoperator(Teleoperator):
    """Combine any number of devices using explicit prefixes for their action names.

    For example, ``{"": keyboard, "arm_": leader}`` combines raw keyboard input and
    prefixed leader positions. Mapping and joint holding belong to the caller's pipeline.
    Observations are forwarded as read-only robot state in the robot's original key space.
    Feedback is routed using each child's advertised feedback features, with prefixes removed.
    """

    config_class = TeleoperatorConfig
    name = "composite"

    def __init__(self, devices: Mapping[str, Teleoperator], config: TeleoperatorConfig | None = None) -> None:
        self._owns_connections = False
        if not devices or len({id(device) for device in devices.values()}) != len(devices):
            raise ValueError("A composite requires distinct devices and at least one device.")
        self.devices = dict(devices)
        super().__init__(config or TeleoperatorConfig())
        # Validate static collisions before connecting any hardware. Dynamic ones are checked per tick.
        _ = self.action_features, self.feedback_features

    @staticmethod
    def _merge(target: dict, values: dict, prefix: str) -> None:
        renamed = {f"{prefix}{key}": value for key, value in values.items()}
        overlap = target.keys() & renamed.keys()
        if overlap:
            raise ValueError(f"Composite devices have overlapping keys: {sorted(overlap)}")
        target.update(renamed)

    @property
    def action_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        for prefix, device in self.devices.items():
            self._merge(features, device.action_features, prefix)
        return features

    @property
    def feedback_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        for prefix, device in self.devices.items():
            self._merge(features, device.feedback_features, prefix)
        return features

    @property
    def is_connected(self) -> bool:
        # A partially disconnected composite still has resources requiring cleanup.
        return any(device.is_connected for device in self.devices.values())

    @property
    def is_calibrated(self) -> bool:
        return all(device.is_calibrated for device in self.devices.values())

    def calibrate(self) -> None:
        for device in self.devices.values():
            device.calibrate()

    def configure(self) -> None:
        for device in self.devices.values():
            device.configure()

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(
                "Connect composite devices together, or use already-connected devices directly."
            )
        self._owns_connections = True
        try:
            for device in self.devices.values():
                device.connect(calibrate=calibrate)
        except BaseException:
            try:
                self.disconnect()
            except Exception:
                logger.exception("Could not fully clean up after composite connection failure")
            raise

    def disconnect(self) -> None:
        errors: list[Exception] = []
        for device in reversed(list(self.devices.values())):
            try:
                if device.is_connected:
                    device.disconnect()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Could not disconnect every composite device", errors)

    def __del__(self) -> None:
        if getattr(self, "_owns_connections", False):
            super().__del__()

    def get_action(self, observation: RobotObservation | None = None) -> RobotAction:
        action: RobotAction = {}
        for prefix, device in self.devices.items():
            self._merge(action, device.get_action(observation=observation), prefix)
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        unknown = feedback.keys() - self.feedback_features.keys()
        if unknown:
            raise ValueError(f"No composite device accepts feedback keys: {sorted(unknown)}")
        for prefix, device in self.devices.items():
            selected = {
                key: feedback[f"{prefix}{key}"]
                for key in device.feedback_features
                if f"{prefix}{key}" in feedback
            }
            if selected:
                device.send_feedback(selected)

    @property
    def supports_position_feedback(self) -> bool:
        devices = [device for device in self.devices.values() if device.feedback_features]
        return bool(devices) and all(
            getattr(device, "supports_position_feedback", True)
            and callable(getattr(device, "enable_torque", None))
            and callable(getattr(device, "disable_torque", None))
            for device in devices
        )

    def _set_torque(self, method: str) -> None:
        if not self.supports_position_feedback:
            raise NotImplementedError("Not every feedback recipient supports position feedback.")
        for device in self.devices.values():
            if device.feedback_features:
                getattr(device, method)()

    def enable_torque(self) -> None:
        self._set_torque("enable_torque")

    def disable_torque(self) -> None:
        self._set_torque("disable_torque")
