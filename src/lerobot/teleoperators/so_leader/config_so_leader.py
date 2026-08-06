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


@dataclass
class SOLeaderConfig:
    """Field definitions shared by the SO-family leader arms.

    This class only carries the fields. The registered configuration users instantiate is
    [`SOLeaderTeleopConfig`], which combines these with [`~teleoperators.TeleoperatorConfig`] and documents
    them all in one place — doc-builder renders only a class's own docstring, never its bases'.
    """

    # Port to connect to the arm
    port: str

    # Whether to use degrees for angles
    use_degrees: bool = True

    # Number of extra attempts when a `sync_read` of the motors fails. Feetech buses can occasionally
    # return a corrupted status packet ("Incorrect status packet!"), especially when several joints move
    # at once, which otherwise aborts the teleoperation loop. Retries are immediate (no sleep) and only
    # happen on failure, so the steady-state read cost is unchanged.
    num_read_retries: int = 2


@TeleoperatorConfig.register_subclass("so101_leader")
@TeleoperatorConfig.register_subclass("so100_leader")
@dataclass
class SOLeaderTeleopConfig(TeleoperatorConfig, SOLeaderConfig):
    """Configuration for the SO-100 and SO-101 leader arms.

    Both arms share this class; `SO100LeaderConfig` and `SO101LeaderConfig` are aliases for it. They differ
    in their calibration and gearing, not in their control code.

    Args:
        port (`str`):
            Serial port the arm is connected to, e.g. `/dev/ttyACM0` on Linux or `COM3` on Windows. Run
            `lerobot-find-port` to identify it.
        use_degrees (`bool`, *optional*, defaults to `True`):
            Whether to report joint positions in degrees. Keep `True` for compatibility with existing
            policies and datasets.
        num_read_retries (`int`, *optional*, defaults to 2):
            Extra attempts when a `sync_read` fails. Feetech buses occasionally return a corrupted status
            packet, especially when several joints move at once, which would otherwise abort the
            teleoperation loop. Retries are immediate and only happen on failure, so steady-state read cost
            is unchanged.
        id (`str`, *optional*):
            Identifier for this particular arm; also names its calibration file.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to the LeRobot calibration home.

    Example:
        ```python
        >>> from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        >>> config = SO101LeaderConfig(port="/dev/ttyACM0")  # doctest: +SKIP
        >>> teleop = SO101Leader(config)  # doctest: +SKIP
        ```
    """

    pass


SO100LeaderConfig = SOLeaderTeleopConfig
SO101LeaderConfig = SOLeaderTeleopConfig
