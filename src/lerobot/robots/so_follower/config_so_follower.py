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


@dataclass
class SOFollowerConfig:
    """Field definitions shared by the SO-family follower arms.

    This class only carries the fields. The registered configuration users instantiate is
    [`SOFollowerRobotConfig`], which combines these with [`~robots.RobotConfig`] and documents them all in
    one place — doc-builder renders only a class's own docstring, never its bases'.
    """

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

    # Number of extra attempts when a `sync_read` of the motors fails. Feetech buses can occasionally
    # return a corrupted status packet ("Incorrect status packet!"), especially when several joints move
    # at once, which otherwise aborts the control loop. Retries are immediate (no sleep) and only happen on
    # failure, so the steady-state read cost is unchanged.
    num_read_retries: int = 2


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    """Configuration for the SO-100 and SO-101 follower arms.

    Both arms share this class; `SO100FollowerConfig` and `SO101FollowerConfig` are aliases for it. They
    differ in their calibration and gearing, not in their control code.

    Args:
        port (`str`):
            Serial port the arm is connected to, e.g. `/dev/ttyACM0` on Linux or `COM3` on Windows. Run
            `lerobot-find-port` to identify it.
        disable_torque_on_disconnect (`bool`, *optional*, defaults to `True`):
            Whether to release the motors on disconnect. Leave `True` unless the arm is holding a load it
            must not drop.
        max_relative_target (`float | dict[str, float]`, *optional*):
            Caps how far a single action may move the arm from its present position, as a safety limit. A
            scalar applies to every motor; a dict maps motor name to a per-motor cap. `None` disables
            clipping. Enabling this costs an extra read of the present position on every step.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras to read alongside the arm's joint positions, keyed by the name they appear under in
            observations. Each must specify `width`, `height` and `fps`.
        use_degrees (`bool`, *optional*, defaults to `True`):
            Whether to report and accept joint positions in degrees. Keep `True` for compatibility with
            existing policies and datasets.
        position_p_coefficient (`int`, *optional*, defaults to 16):
            Proportional gain written to the Feetech STS3215 motors at connect time.
        position_i_coefficient (`int`, *optional*, defaults to 0):
            Integral gain written to the motors at connect time.
        position_d_coefficient (`int`, *optional*, defaults to 32):
            Derivative gain written to the motors at connect time.
        num_read_retries (`int`, *optional*, defaults to 2):
            Extra attempts when a `sync_read` fails. Feetech buses occasionally return a corrupted status
            packet, especially when several joints move at once, which would otherwise abort the control
            loop. Retries are immediate and only happen on failure, so steady-state read cost is unchanged.
        id (`str`, *optional*):
            Identifier for this particular arm; also names its calibration file.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to the LeRobot calibration home.

    Example:
        ```python
        >>> from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        >>> config = SO101FollowerConfig(port="/dev/ttyACM0", max_relative_target=5.0)  # doctest: +SKIP
        >>> robot = SO101Follower(config)  # doctest: +SKIP
        ```
    """

    pass


SO100FollowerConfig = SOFollowerRobotConfig
SO101FollowerConfig = SOFollowerRobotConfig
