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

from ..teleoperator import Teleoperator
from .config import XLeRobotMountIdleConfig


class XLeRobotMountIdle(Teleoperator):
    """Teleoperator that keeps the XLeRobot camera mount static."""

    config_class = XLeRobotMountIdleConfig
    name = "xlerobot_mount_idle"

    def __init__(self, config: XLeRobotMountIdleConfig):
        super().__init__(config)
        self.config = config
        self._action = {
            self.config.pan_key: 0.0,
            self.config.tilt_key: 0.0,
        }
        self._connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in self._action.keys()}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_action(self) -> Dict[str, float]:
        return dict(self._action)

    def send_feedback(self, feedback: Dict[str, Any]) -> None:  # noqa: ARG002
        return None

    @property
    def is_calibrated(self) -> bool:
        return True
