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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("omx_leader")
@dataclass
class OmxLeaderConfig(TeleoperatorConfig):
    """Configuration for the OMX leader arm.

    Args:
        port (`str`):
            Serial port the arm is connected to, e.g. `/dev/ttyACM0` on Linux or `COM3` on Windows. Run
            `lerobot-find-port` to identify it.
        gripper_open_pos (`float`, *optional*, defaults to 60.0):
            Goal position written to the gripper motor, held under current-based position control so the
            gripper springs back to this position when released, letting it be used as a physical trigger.
        id (`str`, *optional*):
            Identifier for this particular arm; also names its calibration file.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to the LeRobot calibration home.

    Example:
        ```python
        >>> from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig
        >>> config = OmxLeaderConfig(port="/dev/ttyACM0")  # doctest: +SKIP
        >>> teleop = OmxLeader(config)  # doctest: +SKIP
        ```
    """

    # Port to connect to the arm
    port: str

    # Sets the arm in torque mode with the gripper motor set to this value. This makes it possible to squeeze
    # the gripper and have it spring back to an open position on its own.
    gripper_open_pos: float = 60.0
