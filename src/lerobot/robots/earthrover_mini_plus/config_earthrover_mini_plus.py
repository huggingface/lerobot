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
from lerobot.cameras.earthrover_mini_camera.configuration_earthrover_mini import EarthRoverMiniCameraConfig, ColorMode
from lerobot.cameras.earthrover_mini_camera import EarthRoverMiniCamera
from ..config import RobotConfig

def earthrover_mini_plus_cameras_config() -> dict[str, CameraConfig]:
    # to edit based on earth rover's cameras
    return {
        "front main": EarthRoverMiniCameraConfig(
            index_or_path= EarthRoverMiniCameraConfig.FRONT_CAM_MAIN, color_mode=ColorMode.RGB
        ),
        "rear main": EarthRoverMiniCameraConfig(
            index_or_path=EarthRoverMiniCameraConfig.REAR_CAM_MAIN, color_mode=ColorMode.RGB
        )
    #     "front sub": EarthRoverMiniCameraConfig(
    #         index_or_path= EarthRoverMiniCameraConfig.FRONT_CAM_SUB, color_mode=ColorMode.RGB
    #     ),
    #     "rear sub": EarthRoverMiniCameraConfig(
    #         index_or_path=EarthRoverMiniCameraConfig.REAR_CAM_SUB, color_mode=ColorMode.RGB
    #     )
    }


@RobotConfig.register_subclass("earthrover_mini_plus")
@dataclass
class EarthRoverMiniPlusConfig(RobotConfig):

    port: str = "8888"  #"/dev/ttyACM0" 
    remote_ip: str = "192.168.11.1"  # port to be changed

    cameras: dict[str, CameraConfig] = field(default_factory=earthrover_mini_plus_cameras_config)

    # any other configs we want




"""
# todo: maybe have client and host configs
@dataclass
class EarthRoverMiniPlusHostConfig:
    # Network Configuration
    port_cmd: int = 5555
    port_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 30

    # Watchdog: stop the robot if no command is received for over 0.5 seconds.
    watchdog_timeout_ms: int = 500

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30

@RobotConfig.register_subclass("earthrover_mini_plus_client")  
@dataclass
class EarthRoverMiniPlusClientConfig:
    # Network Configuration
    remote_ip: str
    port_cmd: int = 5555
    port_observations: int = 5556

    # todo: update this based on earthrover miniplus teleoperator
    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "up",
            "backward": "down",
            "increase_time": "+",
            "decrease_time": "-",
            "rotate_left": "left",
            "rotate_right": "right",
            # Speed control
            # "speed_up": "r",
            # "speed_down": "f",
            # quit teleop
            # "quit": "q",
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=earthrover_mini_plus_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
    """