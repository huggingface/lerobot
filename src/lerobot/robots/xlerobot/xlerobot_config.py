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
from .xlerobot_form_factor import ROBOT_PARTS, BUS_MAPPINGS


def xlerobot_cameras_config() -> dict[str, CameraConfig]:
    return {
        "left_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video6",
            fps=30,
            width=1024,
            height=768,
            rotation=Cv2Rotation.NO_ROTATION,
            fourcc="MJPG",
        ),

        "right_wrist": OpenCVCameraConfig(
            index_or_path="/dev/video8",
            fps=30,
            width=1024,
            height=768,
            rotation=Cv2Rotation.NO_ROTATION,
            fourcc="MJPG",
        ),
        # Pick either the OpenCV or RealSense camera
        # "head(RGDB)": OpenCVCameraConfig(
        #     index_or_path="/dev/video2", fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        # ),
        "head": RealSenseCameraConfig(
            serial_number_or_name="142422251177",  # Replace with camera SN
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.BGR,
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        ),
    }

teleop_keys_default = {
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

def default_motor_ids() -> dict[str, int | None]:
    """
    Provide a configurable mapping for servo IDs.

    Leaving a value as ``None`` tells the robot runtime to fall back to either
    the calibration file (if available) or the hardware defaults from
    :mod:`robot_hardware`. Override a subset of entries by specifying a value.
    """
    return {name: None for name in ROBOT_PARTS}


@RobotConfig.register_subclass("xlerobot")
@dataclass
class XLerobotConfig(RobotConfig):

    # Motor configuration layout: "4_bus" or "2_bus"
    motor_layout: str = "4_bus"

    # Map of logical bus roles to device ports
    # You can bind them to stable names like /dev/left-hand
    # 1. Get serian number udevadm info --attribute-walk --name=/dev/ttyACM1 | grep -E 'idVendor|idProduct|serial|DEVPATH' -n
    # 2. Create udev rule (example for neck):
    # sudo tee -a /etc/udev/rules.d/99-my-tty.rules >/dev/null <<'EOF'
    # ATTRS{serial}=="YOUR_SERIAL_HERE", SYMLINK+="neck"
    # EOF
    ports: dict[str, str | None] = field(
        default_factory=lambda: {
            "left_arm": "/dev/left-hand",
            "right_arm": "/dev/right-hand",
            "head": "/dev/neck",
            "base": "/dev/wheels",
        }
    )
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)
    motor_ids: dict[str, int | None] = field(default_factory=default_motor_ids)
    verify_motors_on_connect: bool = True

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: teleop_keys_default
    )


@dataclass
class XLerobotHostConfig:
    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 3600

    # Watchdog: stop the robot if no command is received for over 30 seconds.
    watchdog_timeout_ms: int = 30000

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30

@RobotConfig.register_subclass("xlerobot_client")
@dataclass
class XLerobotClientConfig(RobotConfig):
    # Network Configuration
    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: teleop_keys_default
    )

    cameras: dict[str, CameraConfig] = field(default_factory=xlerobot_cameras_config)

    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5
