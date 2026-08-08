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
    """Build the default camera set for a LeKiwi base.

    Returns:
        `dict[str, CameraConfig]`: The `front` and `wrist` OpenCV cameras at the device paths and
        rotations of a standard LeKiwi build. Override the `cameras` field if yours is wired differently.
    """
    return {
        "front": OpenCVCameraConfig(
            index_or_path="/dev/video0",
            fps=30,
            width=640,
            height=480,
            fourcc="MJPG",
            rotation=Cv2Rotation.ROTATE_180,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/video2",
            fps=30,
            width=480,
            height=640,
            fourcc="MJPG",
            rotation=Cv2Rotation.ROTATE_90,
        ),
    }


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiConfig(RobotConfig):
    """Configuration for LeKiwi, running on the robot itself.

    This is the config used by the process on the LeKiwi's own computer. To drive one from another machine,
    use [`LeKiwiClientConfig`] instead.

    Args:
        port (`str`, *optional*, defaults to `"/dev/ttyACM0"`):
            Serial port of the motor bus on the robot's computer.
        disable_torque_on_disconnect (`bool`, *optional*, defaults to `True`):
            Whether to release the motors on disconnect.
        max_relative_target (`float | dict[str, float]`, *optional*):
            Caps how far a single action may move the arm from its present position, as a safety limit. A
            scalar applies to every motor; a dict maps motor name to a per-motor cap. `None` disables
            clipping.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras to read alongside the joint positions. Defaults to the standard `front` and `wrist`
            build; see [`lekiwi_cameras_config`].
        use_degrees (`bool`, *optional*, defaults to `True`):
            Whether to report and accept arm joint positions in degrees.
        num_read_retries (`int`, *optional*, defaults to 2):
            Extra attempts when a `sync_read` fails. Feetech buses occasionally return a corrupted status
            packet, which would otherwise abort the control loop.
        id (`str`, *optional*):
            Identifier for this particular robot; also names its calibration file.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to the LeRobot calibration home.
    """

    port: str = "/dev/ttyACM0"  # port to connect to the bus

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_cameras_config)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True

    # Number of extra attempts when a `sync_read` of the motors fails. Feetech buses can occasionally
    # return a corrupted status packet ("Incorrect status packet!"), especially when several joints move
    # at once, which otherwise aborts the control loop. Retries are immediate (no sleep) and only happen on
    # failure, so the steady-state read cost is unchanged.
    num_read_retries: int = 2


@dataclass
class LeKiwiHostConfig:
    """Configuration for the host process that serves a LeKiwi over the network.

    Args:
        port_zmq_cmd (`int`, *optional*, defaults to 5555):
            ZMQ port the host listens on for actions.
        port_zmq_observations (`int`, *optional*, defaults to 5556):
            ZMQ port the host publishes observations on.
        connection_time_s (`int`, *optional*, defaults to 30):
            How long the host stays up before shutting down.
        watchdog_timeout_ms (`int`, *optional*, defaults to 500):
            Stop the robot if no command arrives within this window. Guards against a dropped client
            leaving the base driving.
        max_loop_freq_hz (`int`, *optional*, defaults to 30):
            Control loop frequency. Lower it if the robot jitters, and watch CPU load with `top`.
    """

    # Network Configuration
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # Duration of the application
    connection_time_s: int = 30

    # Watchdog: stop the robot if no command is received for over 0.5 seconds.
    watchdog_timeout_ms: int = 500

    # If robot jitters decrease the frequency and monitor cpu load with `top` in cmd
    max_loop_freq_hz: int = 30


@RobotConfig.register_subclass("lekiwi_client")
@dataclass
class LeKiwiClientConfig(RobotConfig):
    """Configuration for driving a LeKiwi from another machine.

    Presents the same [`~robots.Robot`] interface as the robot-side [`LeKiwiConfig`], but every call goes
    over ZMQ to the host process. Calibration lives on the robot, so nothing here configures it.

    Args:
        remote_ip (`str`):
            IP address of the LeKiwi's computer on the network.
        port_zmq_cmd (`int`, *optional*, defaults to 5555):
            ZMQ port to send actions to. Must match the host's `port_zmq_cmd`.
        port_zmq_observations (`int`, *optional*, defaults to 5556):
            ZMQ port to receive observations on. Must match the host's `port_zmq_observations`.
        teleop_keys (`dict[str, str]`, *optional*):
            Keyboard bindings for driving the base: movement, rotation, speed control and quit.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras expected in the observation stream. Defaults to the standard `front` and `wrist` build.
        polling_timeout_ms (`int`, *optional*, defaults to 15):
            How long to wait for an observation before giving up on that step.
        connect_timeout_s (`int`, *optional*, defaults to 5):
            How long to wait for the host to answer when connecting.
        id (`str`, *optional*):
            Identifier for this particular robot.
        calibration_dir (`Path`, *optional*):
            Unused by the client: calibration is held on the robot.
    """

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
