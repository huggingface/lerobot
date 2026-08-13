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


@RobotConfig.register_subclass("hope_jr_hand")
@dataclass
class HopeJrHandConfig(RobotConfig):
    """Configuration for one Hope Jr hand.

    Each hand is a separate robot, so a two-handed setup uses two of these with different `side` and
    `port` values.

    Args:
        port (`str`):
            Serial port the hand is connected to. Run `lerobot-find-port` to identify it.
        side (`str`):
            Which hand this is, `"left"` or `"right"`. Determines the motor layout, so it must match the
            hardware.
        disable_torque_on_disconnect (`bool`, *optional*, defaults to `True`):
            Whether to release the motors on disconnect.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras to read alongside the hand's joint positions.
        id (`str`, *optional*):
            Identifier for this particular hand; also names its calibration file.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to the LeRobot calibration home.
    """

    port: str  # Port to connect to the hand
    side: str  # "left" / "right"

    disable_torque_on_disconnect: bool = True

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the camera settings and the hand side.

        Raises:
            ValueError: If `side` is not `"left"` or `"right"`, or if a camera omits `width`, `height` or
                `fps`.
        """
        super().__post_init__()
        if self.side not in ["right", "left"]:
            raise ValueError(self.side)


@RobotConfig.register_subclass("hope_jr_arm")
@dataclass
class HopeJrArmConfig(RobotConfig):
    """Configuration for one Hope Jr arm.

    Args:
        port (`str`):
            Serial port the arm is connected to. Run `lerobot-find-port` to identify it.
        disable_torque_on_disconnect (`bool`, *optional*, defaults to `True`):
            Whether to release the motors on disconnect. Leave `True` unless the arm is holding a load it
            must not drop.
        max_relative_target (`float | dict[str, float]`, *optional*):
            Caps how far a single action may move the arm from its present position, as a safety limit. A
            scalar applies to every motor; a dict maps motor name to a per-motor cap. `None` disables
            clipping.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras to read alongside the arm's joint positions.
        id (`str`, *optional*):
            Identifier for this particular arm; also names its calibration file.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to the LeRobot calibration home.
    """

    port: str  # Port to connect to the hand
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
