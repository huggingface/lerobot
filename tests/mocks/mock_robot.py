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
from dataclasses import dataclass, field
from functools import cached_property

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.motors.motors_bus import Motor, MotorNormMode
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from tests.mocks.mock_motors_bus import MockMotorsBus


@RobotConfig.register_subclass("mock_robot")
@dataclass
class MockRobotConfig(RobotConfig):
    n_motors: int = 3
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
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

        if len(self.cameras) > 0:
            raise NotImplementedError  # TODO with the cameras refactor


class MockRobot(Robot):
    """Mock Robot to be used for testing."""

    config_class = MockRobotConfig
    name = "mock_robot"

    def __init__(self, config: MockRobotConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = config.calibrated
        self.cameras = make_cameras_from_configs(config.cameras)

        mock_motors = {}
        for i in range(config.n_motors):
            motor_name = f"motor_{i + 1}"
            mock_motors[motor_name] = Motor(
                id=i + 1,
                model="model_1",  # Use model_1 which exists in MockMotorsBus tables
                norm_mode=MotorNormMode.RANGE_M100_100,
            )

        self.bus = MockMotorsBus("/dev/dummy-port", mock_motors)

        # NOTE(fracapuano): The .motors attribute was used from the previous interface
        self.motors = [f"motor_{i + 1}" for i in range(config.n_motors)]

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

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
    def get_observation(self) -> RobotObservation:
        if self.config.random_values:
            return {f"{motor}.pos": random.uniform(-100, 100) for motor in self.motors}
        else:
            return {
                f"{motor}.pos": val for motor, val in zip(self.motors, self.config.static_values, strict=True)
            }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        return action

    @check_if_not_connected
    def disconnect(self) -> None:
        self._is_connected = False
