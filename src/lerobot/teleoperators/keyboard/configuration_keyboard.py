#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Configuration for keyboard teleoperators."""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    """Configuration for the plain keyboard teleoperator.

    Args:
        id (`str`, *optional*):
            Identifier for this particular unit, used to tell apart several teleoperators of the same
            type. It also names the calibration file, so keep it stable for a given piece of hardware.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to a per-teleoperator directory under
            the LeRobot calibration home.
    """

    # TODO(Steven): Consider setting in here the keys that we want to capture/listen


@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    """Configuration for controlling a robot end-effector with keyboard inputs.

    Args:
        use_gripper (`bool`, *optional*, defaults to `True`):
            Whether to include a `gripper` entry in the produced actions.
        id (`str`, *optional*):
            Identifier for this particular unit, used to tell apart several teleoperators of the same
            type. It also names the calibration file, so keep it stable for a given piece of hardware.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to a per-teleoperator directory under
            the LeRobot calibration home.
    """

    use_gripper: bool = True


@TeleoperatorConfig.register_subclass("keyboard_rover")
@dataclass
class KeyboardRoverTeleopConfig(TeleoperatorConfig):
    """Configuration for the WASD-style keyboard teleoperator for mobile robots like EarthRover Mini Plus.

    Args:
        linear_speed (`float`, *optional*, defaults to 1.0):
            Initial linear velocity magnitude (-1 to 1 range for SDK robots).
        angular_speed (`float`, *optional*, defaults to 1.0):
            Initial angular velocity magnitude (-1 to 1 range for SDK robots).
        speed_increment (`float`, *optional*, defaults to 0.1):
            Amount `current_linear_speed` changes by on each `+`/`-` key press.
        turn_assist_ratio (`float`, *optional*, defaults to 0.3):
            Forward-motion multiplier applied when turning with `a`/`d` while otherwise stationary.
        angular_speed_ratio (`float`, *optional*, defaults to 0.6):
            Ratio of angular to linear speed increment, so both scale together on `+`/`-`.
        min_linear_speed (`float`, *optional*, defaults to 0.1):
            Floor for `current_linear_speed` when decreasing it.
        min_angular_speed (`float`, *optional*, defaults to 0.05):
            Floor for `current_angular_speed` when decreasing it.
        id (`str`, *optional*):
            Identifier for this particular unit, used to tell apart several teleoperators of the same
            type. It also names the calibration file, so keep it stable for a given piece of hardware.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to a per-teleoperator directory under
            the LeRobot calibration home.
    """

    linear_speed: float = 1.0
    angular_speed: float = 1.0
    speed_increment: float = 0.1
    turn_assist_ratio: float = 0.3
    angular_speed_ratio: float = 0.6
    min_linear_speed: float = 0.1
    min_angular_speed: float = 0.05
