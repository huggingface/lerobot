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

from lerobot.cameras.configs import CameraConfig, Cv2Rotation, ColorMode
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

from ..config import RobotConfig


def xlerobot_cameras_config() -> dict[str, CameraConfig]:
    return {
        # Head camera MUST be opened FIRST to avoid resource conflicts
        # Opening it after wrist cameras causes it to fail
        # Using supported resolutions and FPS to enable all 3 cameras
        # Note: head/left_wrist cameras only support 30fps (they ignore lower FPS requests)
        #       right_wrist supports multiple FPS values, but we use 30fps for consistency
        # Note: Different cameras support different resolutions, so we use optimal ones for each
        # Full resolution kept for robot control and recording - Rerun visualization downsamples separately
        "head": OpenCVCameraConfig(
            index_or_path="/dev/video0", 
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
            rotation=Cv2Rotation.NO_ROTATION,
        ),
        
        "left_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2",
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
            rotation=Cv2Rotation.NO_ROTATION,
        ),     

        "right_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video4",
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
            rotation=Cv2Rotation.NO_ROTATION,
        ),        
        
        # "head": RealSenseCameraConfig(
        #     serial_number_or_name="125322060037",  # Replace with camera SN
        #     fps=30,
        #     width=1280,
        #     height=720,
        #     color_mode=ColorMode.BGR, # Request BGR output
        #     rotation=Cv2Rotation.NO_ROTATION,
        #     use_depth=True
        # ),
    }


@RobotConfig.register_subclass("xlerobot")
@dataclass
class XLerobotConfig(RobotConfig):
    
    port1: str = "/dev/ttyACM0"  # port to connect to the bus (left arm motors 1-6)
    port2: str = "/dev/ttyACM2"  # port to connect to the bus (right arm motors 1-6)
    port3: str = "/dev/ttyACM1"  # port to connect to the bus (base motors 7-9)
    camera_start_order: tuple[str, ...] | None = ("head", "left_wrist", "right_wrist")
    camera_start_delay_s: float = 0.5
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "i",
            "backward": "k",
            "left": "j",
            "right": "l",
            "rotate_left": "u",
            "rotate_right": "o",
            # Speed control
            "speed_up": "n",
            "speed_down": "m",
            # quit teleop
            "quit": "b",
        }
    )



# ============================================================================
# CLIENT/HOST CONFIGURATIONS DISABLED - DIRECT USB CONNECTION ONLY
# ============================================================================
# The following configurations are commented out to enforce direct USB connection
# to the operating PC. Uncomment these if you need remote operation via ZMQ.
# ============================================================================

# @dataclass
# class XLerobotHostConfig:
#     # Network Configuration
#     port_zmq_cmd: int = 5555
#     port_zmq_observations: int = 5556
#
#     # Duration of the application
#     connection_time_s: int = 3600
#
#     # Watchdog: stop the robot if no command is received for over 0.5 seconds.
#     watchdog_timeout_ms: int = 500
#
#     # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
#     max_loop_freq_hz: int = 30

# @RobotConfig.register_subclass("xlerobot_client")
# @dataclass
# class XLerobotClientConfig(RobotConfig):
#     # Network Configuration
#     remote_ip: str
#     port_zmq_cmd: int = 5555
#     port_zmq_observations: int = 5556
#
#     teleop_keys: dict[str, str] = field(
#         default_factory=lambda: {
#             # Movement
#             "forward": "i",
#             "backward": "k",
#             "left": "j",
#             "right": "l",
#             "rotate_left": "u",
#             "rotate_right": "o",
#             # Speed control
#             "speed_up": "n",
#             "speed_down": "m",
#             # quit teleop
#             "quit": "b",
#         }
#     )
#
#     cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)
#
#     polling_timeout_ms: int = 15
#     connect_timeout_s: int = 5
