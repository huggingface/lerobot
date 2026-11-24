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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import draccus

from lerobot.lerobot_types import RobotAction
from lerobot.processor.base_action_processor import BaseActionProcessorStep

from .config import TeleoperatorConfig
from .teleoperator import Teleoperator


class DeviceConfig(TeleoperatorConfig, draccus.ChoiceRegistry):
    """Exclude composites from child choices to keep Draccus CLI expansion finite."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        # Draccus uses isinstance to encode optional child configs with their type tags.
        if cls is DeviceConfig:
            return TeleoperatorConfig in subclass.__mro__ and subclass is not CompositeTeleoperatorConfig
        return NotImplemented

    @classmethod
    def get_known_choices(cls) -> dict[str, type[TeleoperatorConfig]]:
        return {
            name: config
            for name, config in TeleoperatorConfig.get_known_choices().items()
            if name != "composite"
        }

    @classmethod
    def get_choice_class(cls, name: str) -> type[TeleoperatorConfig]:
        return cls.get_known_choices()[name]


@TeleoperatorConfig.register_subclass("composite")
@dataclass(kw_only=True)
class CompositeTeleoperatorConfig(TeleoperatorConfig):
    base: DeviceConfig
    arm: DeviceConfig | None = None
    # Defaults to the mapping registered for the base input.
    mapping: BaseActionProcessorStep | None = None


class CompositeTeleoperator(Teleoperator):
    """Combine devices with action prefixes, e.g. ``[(keyboard, ""), (leader, "arm_")]``.

    Feedback is routed to each device with its prefix removed.
    """

    config_class = CompositeTeleoperatorConfig
    name = "composite"

    def __init__(
        self,
        devices: Sequence[tuple[Teleoperator, str]],
        config: CompositeTeleoperatorConfig | None = None,
    ) -> None:
        self.devices = list(devices)
        if not 1 <= len(self.devices) <= 2:
            raise ValueError("A composite requires a base and optionally an arm.")
        super().__init__(config or TeleoperatorConfig())
        self.mapping = config.mapping if config is not None else None

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
        for device, prefix in self.devices:
            self._merge(features, device.action_features, prefix)
        return features

    @property
    def feedback_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        for device, prefix in self.devices:
            self._merge(features, device.feedback_features, prefix)
        return features

    @property
    def is_connected(self) -> bool:
        # Stay connected while any device still needs cleanup, e.g. after Esc disconnects the keyboard.
        return any(device.is_connected for device, _ in self.devices)

    @property
    def is_calibrated(self) -> bool:
        return all(device.is_calibrated for device, _ in self.devices)

    def calibrate(self) -> None:
        for device, _ in self.devices:
            device.calibrate()

    def configure(self) -> None:
        for device, _ in self.devices:
            device.configure()

    def connect(self, calibrate: bool = True) -> None:
        for device, _ in self.devices:
            device.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        for device, _ in reversed(self.devices):
            if device.is_connected:
                device.disconnect()

    def get_action(self) -> RobotAction:
        action: RobotAction = {}
        for device, prefix in self.devices:
            self._merge(action, device.get_action(), prefix)
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        for device, prefix in self.devices:
            selected = {
                key: feedback[f"{prefix}{key}"]
                for key in device.feedback_features
                if f"{prefix}{key}" in feedback
            }
            if selected:
                device.send_feedback(selected)
