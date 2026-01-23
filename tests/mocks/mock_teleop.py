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

import random
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected


@TeleoperatorConfig.register_subclass("mock_teleop")
@dataclass
class MockTeleopConfig(TeleoperatorConfig):
    n_motors: int = 3
    random_values: bool = True
    static_values: list[float] | None = None
    calibrated: bool = True

    def __post_init__(self):
        if self.n_motors < 1:
            raise ValueError(self.n_motors)

        if self.random_values and self.static_values is not None:
            raise ValueError("Choose either random values or static values")

        if self.static_values is not None and len(self.static_values) != self.n_motors:
            raise ValueError("Specify the same number of static values as motors")


class MockTeleop(Teleoperator):
    """Mock Teleoperator to be used for testing."""

    config_class = MockTeleopConfig
    name = "mock_teleop"

    def __init__(self, config: MockTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = config.calibrated
        self.motors = [f"motor_{i + 1}" for i in range(config.n_motors)]

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._is_connected = True
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @check_if_not_connected
    def calibrate(self) -> None:
        self._is_calibrated = True

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        if self.config.random_values:
            return {f"{motor}.pos": random.uniform(-100, 100) for motor in self.motors}
        else:
            return {
                f"{motor}.pos": val for motor, val in zip(self.motors, self.config.static_values, strict=True)
            }

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None: ...

    @check_if_not_connected
    def disconnect(self) -> None:
        self._is_connected = False
