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
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("reachy2")
@dataclass
class Reachy2RobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors.
    max_relative_target: float | None = None

    # IP address of the Reachy 2 robot
    ip_address: str | None = "localhost"

    # If True, turn_off_smoothly() will be sent to the robot before disconnecting.
    disable_torque_on_disconnect: bool = False

    # Tag for external commands control
    # Set to True if you use an external commands system to control the robot,
    # such as the official teleoperation application: https://github.com/pollen-robotics/Reachy2Teleoperation
    # If True, robot.send_action() will not send commands to the robot.
    use_external_commands: bool = False

    # Robot parts
    # Set to False to not add the corresponding joints part to the robot list of joints.
    # By default, all parts are set to True.
    with_mobile_base: bool = True
    with_l_arm: bool = True
    with_r_arm: bool = True
    with_neck: bool = True
    with_antennas: bool = True

    # Robot cameras
    # Set to True if you want to use the corresponding cameras in the observations.
    # By default, only the teleop cameras are used.
    with_left_teleop_camera: bool = True
    with_right_teleop_camera: bool = True
    with_torso_camera: bool = False

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Add cameras with same ip_address as the robot
        if self.with_left_teleop_camera:
            self.cameras["teleop_left"] = Reachy2CameraConfig(
                name="teleop",
                image_type="left",
                ip_address=self.ip_address,
                fps=15,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
            )
        if self.with_right_teleop_camera:
            self.cameras["teleop_right"] = Reachy2CameraConfig(
                name="teleop",
                image_type="right",
                ip_address=self.ip_address,
                fps=15,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
            )
        if self.with_torso_camera:
            self.cameras["torso_rgb"] = Reachy2CameraConfig(
                name="depth",
                image_type="rgb",
                ip_address=self.ip_address,
                fps=15,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
            )

        super().__post_init__()

        if not (
            self.with_mobile_base
            or self.with_l_arm
            or self.with_r_arm
            or self.with_neck
            or self.with_antennas
        ):
            raise ValueError(
                "No Reachy2Robot part used.\n"
                "At least one part of the robot must be set to True "
                "(with_mobile_base, with_l_arm, with_r_arm, with_neck, with_antennas)"
            )
