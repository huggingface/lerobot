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

"""An observation-driven teleoperator for the gamepad paired with the Duck."""

import math
import time
from typing import Any

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_microduck_gamepad import MicroDuckGamepadConfig


class MicroDuckGamepad(Teleoperator):
    config_class = MicroDuckGamepadConfig
    name = "microduck_gamepad"

    def __init__(self, config: MicroDuckGamepadConfig):
        super().__init__(config)
        self._connected = False
        self._action: RobotAction | None = None
        self._received_at = 0.0

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "x.vel",
                "y.vel",
                "theta.vel",
                "neck_pitch.pos",
                "head_pitch.pos",
                "head_yaw.pos",
                "head_roll.pos",
            ),
            float,
        )

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._action = None
        self._connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def on_observation(self, observation: RobotObservation) -> None:
        self._action = None
        sample = observation.get("microduck.gamepad")
        if not isinstance(sample, dict) or not isinstance(sample.get("action"), dict):
            raise ValueError("microduck_gamepad requires --robot.control_mode=gamepad")
        action = sample["action"]
        if action.keys() != self.action_features.keys():
            raise ValueError("The Microduck gamepad action has missing or unexpected fields")
        values = {key: float(value) for key, value in action.items()}
        received_at = float(sample["received_at"])
        if not all(math.isfinite(value) for value in [*values.values(), received_at]):
            raise ValueError("The Microduck gamepad sample contains non-finite values")
        self._action = values
        self._received_at = received_at

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action, self._action = self._action, None
        if action is None or not 0 <= time.monotonic() - self._received_at <= 1.0:
            raise RuntimeError("Read a fresh Microduck observation before each gamepad action")
        return dict(action)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        self._action = None
        self._connected = False
