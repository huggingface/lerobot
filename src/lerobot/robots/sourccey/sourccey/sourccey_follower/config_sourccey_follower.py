# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 Vulcan Robotics, Inc. All rights reserved.
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

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.motors.motors_bus import Motor
from lerobot.robots.config import RobotConfig


def sourccey_motor_models() -> dict[str, str]:
    return {
        "shoulder_pan": "sts3215",
        "shoulder_lift": "sts3250",
        "elbow_flex": "sts3250",
        "wrist_flex": "sts3215",
        "wrist_roll": "sts3215",
        "gripper": "sts3215",
    }

def sourccey_cameras_config() -> dict[str, CameraConfig]:
    return {
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=320, height=240
        ),
    }

@RobotConfig.register_subclass("sourccey_follower")
@dataclass
class SourcceyFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str
    orientation: str = "left"

    # The models of the motors to use for the follower arms.
    motor_models: dict[str, str] = field(default_factory=sourccey_motor_models)

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # `max_current_safety_threshold` is the maximum current threshold for safety purposes.
    max_current_safety_threshold: int = 2500

    # `max_current_calibration_threshold` is the maximum current threshold for calibration purposes.
    max_current_calibration_threshold: int = 75

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=sourccey_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
