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

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def lekiwi_cameras_config() -> dict[str, CameraConfig]:
    return {
        "head_top": OpenCVCameraConfig(
            index_or_path="/dev/am_camera_head_top",
            fps=30,
            width=640,
            height=480,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
        #     "head_back": OpenCVCameraConfig(
        #         index_or_path="/dev/am_camera_head_back", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        #     ),
        #     "head_front": OpenCVCameraConfig(
        #         index_or_path="/dev/am_camera_head_front", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        #     ),
        #     "wrist_left": OpenCVCameraConfig(
        #         index_or_path="/dev/am_camera_wrist_left", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        #     ),
        #     "wrist_right": OpenCVCameraConfig(
        #         index_or_path="/dev/am_camera_wrist_right", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        #     ),
    }


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiConfig(RobotConfig):
    left_port: str = "/dev/am_arm_follower_left"  # port to connect to the bus
    right_port: str = "/dev/am_arm_follower_right"  # port to connect to the bus
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False


@dataclass
class LeKiwiHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 6000

    # Watchdog: stop the robot if no command is received for over 1.5 seconds.
    watchdog_timeout_ms: int = 1500

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("lekiwi_client")
@dataclass
class LeKiwiClientConfig(RobotConfig):
    # Network Configuration
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "z",
            "right": "x",
            "rotate_left": "a",
            "rotate_right": "d",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # Z axis
            "lift_up": "u",
            "lift_down": "j",
            # quit teleop
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
