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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("homunculus_glove")
@dataclass
class HomunculusGloveConfig(TeleoperatorConfig):
    """Configuration for the Homunculus Glove teleoperator.

    Args:
        port (`str`):
            Serial port the glove is connected to, e.g. `/dev/ttyACM0`.
        side (`str`):
            Which hand the glove is worn on, `"left"` or `"right"`. Selects which joints get their drive
            mode inverted so the produced action matches the HopeJR hand convention.
        baud_rate (`int`, *optional*, defaults to 115200):
            Serial communication speed in bauds.
        id (`str`, *optional*):
            Identifier for this particular unit, used to tell apart several teleoperators of the same
            type. It also names the calibration file, so keep it stable for a given piece of hardware.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to a per-teleoperator directory under
            the LeRobot calibration home.
    """

    port: str  # Port to connect to the glove
    side: str  # "left" / "right"
    baud_rate: int = 115_200

    def __post_init__(self):
        """Validate that `side` is one of `"left"` or `"right"`.

        Raises:
            ValueError: If `side` is neither `"left"` nor `"right"`.
        """
        if self.side not in ["right", "left"]:
            raise ValueError(self.side)


@TeleoperatorConfig.register_subclass("homunculus_arm")
@dataclass
class HomunculusArmConfig(TeleoperatorConfig):
    """Configuration for the Homunculus Arm teleoperator.

    Args:
        port (`str`):
            Serial port the arm is connected to, e.g. `/dev/ttyACM0`.
        baud_rate (`int`, *optional*, defaults to 115200):
            Serial communication speed in bauds.
        id (`str`, *optional*):
            Identifier for this particular unit, used to tell apart several teleoperators of the same
            type. It also names the calibration file, so keep it stable for a given piece of hardware.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to a per-teleoperator directory under
            the LeRobot calibration home.
    """

    port: str  # Port to connect to the arm
    baud_rate: int = 115_200
