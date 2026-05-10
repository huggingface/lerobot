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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


def _validate_position_p_coefficient(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise ValueError(f"position_p_coefficient must be an integer in [0, 255], got {value!r}")


@dataclass
class SOFollowerConfig:
    """Base configuration class for SO Follower robots."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True

    # Position-mode proportional gain written to Feetech STS3215 motors at connect time.
    position_p_coefficient: int = 16

    def __post_init__(self) -> None:
        _validate_position_p_coefficient(self.position_p_coefficient)


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    def __post_init__(self) -> None:
        RobotConfig.__post_init__(self)
        _validate_position_p_coefficient(self.position_p_coefficient)


SO100FollowerConfig = SOFollowerRobotConfig
SO101FollowerConfig = SOFollowerRobotConfig
