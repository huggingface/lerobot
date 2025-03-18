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

import abc
from dataclasses import dataclass, field
from typing import Sequence

import draccus

from lerobot.common.robot_devices.cameras.configs import (
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import (
    DynamixelMotorsBusConfig,
    FeetechMotorsBusConfig,
    MotorsBusConfig,
)


@dataclass
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


# TODO(rcadene, aliberts): remove ManipulatorRobotConfig abstraction
@dataclass
class ManipulatorRobotConfig(RobotConfig):
    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length
    # as the number of motors in your follower arms (assumes all follower arms have the same number of
    # motors).
    max_relative_target: list[float] | float | None = None

    # Optionally set the leader arm in torque mode with the gripper motor set to this angle. This makes it
    # possible to squeeze the gripper and have it spring back to an open position on its own. If None, the
    # gripper is not put in torque mode.
    gripper_open_degree: float | None = None

    mock: bool = False

    def __post_init__(self):
        if self.mock:
            for arm in self.leader_arms.values():
                if not arm.mock:
                    arm.mock = True
            for arm in self.follower_arms.values():
                if not arm.mock:
                    arm.mock = True
            for cam in self.cameras.values():
                if not cam.mock:
                    cam.mock = True

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )


@RobotConfig.register_subclass("aloha")
@dataclass
class AlohaRobotConfig(ManipulatorRobotConfig):
    # Specific to Aloha, LeRobot comes with default calibration files. Assuming the motors have been
    # properly assembled, no manual calibration step is expected. If you need to run manual calibration,
    # simply update this path to ".cache/calibration/aloha"
    calibration_dir: str = ".cache/calibration/aloha_default"

    # /!\ FOR SAFETY, READ THIS /!\
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    # For Aloha, for every goal position request, motor rotations are capped at 5 degrees by default.
    # When you feel more confident with teleoperation or running the policy, you can extend
    # this safety limit and even removing it by setting it to `null`.
    # Also, everything is expected to work safely out-of-the-box, but we highly advise to
    # first try to teleoperate the grippers only (by commenting out the rest of the motors in this yaml),
    # then to gradually add more motors (by uncommenting), until you can teleoperate both arms fully
    max_relative_target: int | None = 5

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                # window_x
                port="/dev/ttyDXL_leader_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm430-w350"],
                    "shoulder": [2, "xm430-w350"],
                    "shoulder_shadow": [3, "xm430-w350"],
                    "elbow": [4, "xm430-w350"],
                    "elbow_shadow": [5, "xm430-w350"],
                    "forearm_roll": [6, "xm430-w350"],
                    "wrist_angle": [7, "xm430-w350"],
                    "wrist_rotate": [8, "xl430-w250"],
                    "gripper": [9, "xc430-w150"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                # window_x
                port="/dev/ttyDXL_leader_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm430-w350"],
                    "shoulder": [2, "xm430-w350"],
                    "shoulder_shadow": [3, "xm430-w350"],
                    "elbow": [4, "xm430-w350"],
                    "elbow_shadow": [5, "xm430-w350"],
                    "forearm_roll": [6, "xm430-w350"],
                    "wrist_angle": [7, "xm430-w350"],
                    "wrist_rotate": [8, "xl430-w250"],
                    "gripper": [9, "xc430-w150"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_left",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/ttyDXL_follower_right",
                motors={
                    # name: (index, model)
                    "waist": [1, "xm540-w270"],
                    "shoulder": [2, "xm540-w270"],
                    "shoulder_shadow": [3, "xm540-w270"],
                    "elbow": [4, "xm540-w270"],
                    "elbow_shadow": [5, "xm540-w270"],
                    "forearm_roll": [6, "xm540-w270"],
                    "wrist_angle": [7, "xm540-w270"],
                    "wrist_rotate": [8, "xm430-w350"],
                    "gripper": [9, "xm430-w350"],
                },
            ),
        }
    )

    # Troubleshooting: If one of your IntelRealSense cameras freeze during
    # data recording due to bandwidth limit, you might need to plug the camera
    # on another USB hub or PCIe card.
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_high": IntelRealSenseCameraConfig(
                serial_number=128422271347,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_low": IntelRealSenseCameraConfig(
                serial_number=130322270656,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": IntelRealSenseCameraConfig(
                serial_number=218622272670,
                fps=30,
                width=640,
                height=480,
            ),
            "cam_right_wrist": IntelRealSenseCameraConfig(
                serial_number=130322272300,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("koch")
@dataclass
class KochRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0085511",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


@RobotConfig.register_subclass("koch_bimanual")
@dataclass
class KochBimanualRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch_bimanual"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0085511",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem575E0031751",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
            "right": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem575E0032081",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


@RobotConfig.register_subclass("moss")
@dataclass
class MossRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/moss"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("so100")
@dataclass
class So100RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/so100"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem58760431091",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=1,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("stretch")
@dataclass
class StretchRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "navigation": OpenCVCameraConfig(
                camera_index="/dev/hello-nav-head-camera",
                fps=10,
                width=1280,
                height=720,
                rotation=-90,
            ),
            "head": IntelRealSenseCameraConfig(
                name="Intel RealSense D435I",
                fps=30,
                width=640,
                height=480,
                rotation=90,
            ),
            "wrist": IntelRealSenseCameraConfig(
                name="Intel RealSense D405",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    mock: bool = False


@RobotConfig.register_subclass("lekiwi")
@dataclass
class LeKiwiRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Network Configuration
    ip: str = "192.168.0.193"
    port: int = 5555
    video_port: int = 5556

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(
                camera_index="/dev/video0", fps=30, width=640, height=480, rotation=90
            ),
            "wrist": OpenCVCameraConfig(
                camera_index="/dev/video2", fps=30, width=640, height=480, rotation=180
            ),
        }
    )

    calibration_dir: str = ".cache/calibration/lekiwi"

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/tty.usbmodem585A0077581",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                    "left_wheel": (7, "sts3215"),
                    "back_wheel": (8, "sts3215"),
                    "right_wheel": (9, "sts3215"),
                },
            ),
        }
    )

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

    mock: bool = False
