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

# Upper bounds of the gripper protection registers, from the Feetech STS3215 memory table.
_GRIPPER_PROTECTION_MAX_VALUES = {
    "gripper_max_torque_limit": 1000,
    "gripper_protection_current": 511,
    "gripper_overload_torque": 100,
}


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

    # Position-mode PID gains written to Feetech STS3215 motors at connect time.
    position_p_coefficient: int = 16
    position_i_coefficient: int = 0
    position_d_coefficient: int = 32

    # Protection registers written to the gripper motor at connect time to avoid burning out the servo (#1809).
    # The defaults were chosen for the stock SO-100/SO-101 gripper; a gripper that needs more force to close can
    # raise them. Values are raw Feetech STS3215 register values:
    # - `gripper_max_torque_limit` -> `Max_Torque_Limit`: 0-1000, in 0.1% of stall torque. The memory table says
    #   the servo copies it into `Torque_Limit` at power-up, so a changed value may need a power cycle to apply.
    # - `gripper_protection_current` -> `Protection_Current`: 0-511, in 6.5 mA steps.
    # - `gripper_overload_torque` -> `Overload_Torque`: 0-100, in % of max torque. A load above this threshold
    #   for `Protection_Time` makes the servo drop its output to `Protective_Torque`.
    gripper_max_torque_limit: int = 500
    gripper_protection_current: int = 250
    gripper_overload_torque: int = 25

    # Number of extra attempts when a `sync_read` of the motors fails. Feetech buses can occasionally
    # return a corrupted status packet ("Incorrect status packet!"), especially when several joints move
    # at once, which otherwise aborts the control loop. Retries are immediate (no sleep) and only happen on
    # failure, so the steady-state read cost is unchanged.
    num_read_retries: int = 2

    def __post_init__(self) -> None:
        """Reject gripper protection values outside their STS3215 register ranges."""
        for name, max_value in _GRIPPER_PROTECTION_MAX_VALUES.items():
            value = getattr(self, name)
            if not 0 <= value <= max_value:
                raise ValueError(f"`{name}` must be between 0 and {max_value}, got {value}.")


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    def __post_init__(self) -> None:
        """Run the checks of both parent configs; `RobotConfig` alone would shadow `SOFollowerConfig`."""
        RobotConfig.__post_init__(self)
        SOFollowerConfig.__post_init__(self)


SO100FollowerConfig = SOFollowerRobotConfig
SO101FollowerConfig = SOFollowerRobotConfig
