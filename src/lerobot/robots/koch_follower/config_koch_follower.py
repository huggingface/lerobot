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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("koch_follower")
@dataclass
class KochFollowerConfig(RobotConfig):
    """Configuration for the Koch v1.1 follower arm.

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
            clipping.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras to read alongside the arm's joint positions, keyed by the name they appear under in
            observations. Each must specify `width`, `height` and `fps`.
        use_degrees (`bool`, *optional*, defaults to `False`):
            Whether to report and accept joint positions in degrees rather than as a normalised range.
        id (`str`, *optional*):
            Identifier for this particular arm; also names its calibration file.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to the LeRobot calibration home.
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
    use_degrees: bool = False
