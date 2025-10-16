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

from __future__ import annotations

from functools import cached_property
from typing import Any, Dict

from ..robot import Robot
from .config import XLeRobotMountConfig


class XLeRobotMount(Robot):
    """Placeholder robot representing the XLeRobot top camera mount joints."""

    config_class = XLeRobotMountConfig
    name = "xlerobot_mount"

    def __init__(self, config: XLeRobotMountConfig):
        super().__init__(config)
        self.config = config
        self._state: Dict[str, float] = {
            self.config.pan_key: 0.0,
            self.config.tilt_key: 0.0,
        }

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return {key: float for key in self._state.keys()}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in self._state.keys()}

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        return None

    def disconnect(self) -> None:
        return None

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any]:
        return dict(self._state)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        for key in self._state.keys():
            if key in action:
                self._state[key] = float(action[key])
        return {key: self._state[key] for key in self._state.keys()}
