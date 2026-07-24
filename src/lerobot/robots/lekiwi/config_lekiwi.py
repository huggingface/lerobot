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

from lerobot.cameras import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig

from ..config import RobotConfig


def lekiwi_cameras_config() -> dict[str, CameraConfig]:
    # NOTE: width/height must match the resolution the cameras actually deliver on the host.
    # These USB cameras return their native resolution regardless of the requested size, so we
    # declare the native shapes here. The client declares its expected frame shapes from this same
    # config, so host capture and client/dataset shapes stay in sync. NO_ROTATION keeps the declared
    # (height, width) equal to the received frame shape (the client does not rotate received frames).
    return {
        "front": OpenCVCameraConfig(
            index_or_path="/dev/video0", fps=30, width=1280, height=720, rotation=Cv2Rotation.NO_ROTATION
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
        # USB "Web Camera", native 640x480, mounted rotated -> portrait (640, 480, 3) after ROTATE_90.
        # Use the stable by-id path: the raw /dev/videoN number shuffles on replug.
        "up": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-id/usb-Web_Camera_Web_Camera_202512181-video-index0",
            fps=30,
            width=480,
            height=640,
            rotation=Cv2Rotation.ROTATE_90,
        ),
    }


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiConfig(RobotConfig):
    port: str = "/dev/ttyACM0"  # port to connect to the bus

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True


@dataclass
class LeKiwiHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 30

    # Watchdog: stop the robot if no command is received for over 0.5 seconds.
    watchdog_timeout_ms: int = 500

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30

    # Optional: stream live observations/actions from the host to Foxglove.
    # When enabled, the host starts a Foxglove WebSocket server; connect the Foxglove
    # app to ws://<host-ip>:<foxglove_port>. Requires the `viz` extra (foxglove-sdk).
    enable_foxglove: bool = False
    foxglove_host: str = "0.0.0.0"  # bind interface (0.0.0.0 = reachable from other machines)
    foxglove_port: int = 8765


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
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
