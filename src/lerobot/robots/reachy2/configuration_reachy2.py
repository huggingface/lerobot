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

from lerobot.cameras import CameraConfig, ColorMode
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("reachy2")
@dataclass
class Reachy2RobotConfig(RobotConfig):
    """Configuration for the Reachy 2 humanoid.

    Reachy 2 is driven over the network rather than a serial bus, so `port` is a TCP port on the robot's
    gRPC service rather than a device path. Calibration is handled by the robot itself and there is no
    LeRobot calibration file.

    Which joints appear in observations and actions is selected by the `with_*` flags: turning a part off
    removes its joints entirely. At least one part must stay enabled.

    Args:
        max_relative_target (`float`, *optional*):
            Caps how far a single action may move a joint from its present position, as a safety limit.
            `None` disables clipping.
        ip_address (`str`, *optional*, defaults to `"localhost"`):
            Address of the Reachy 2 robot.
        port (`int`, *optional*, defaults to 50065):
            TCP port of the robot's service. Not a serial port.
        disable_torque_on_disconnect (`bool`, *optional*, defaults to `False`):
            Whether to call `turn_off_smoothly()` before disconnecting.
        use_external_commands (`bool`, *optional*, defaults to `False`):
            Set `True` when another system drives the robot, such as the official
            [teleoperation app](https://github.com/pollen-robotics/Reachy2Teleoperation). In that mode
            [`~robots.Robot.send_action`] does not send anything to the robot.
        with_mobile_base (`bool`, *optional*, defaults to `True`):
            Whether to include the mobile base's joints.
        with_l_arm (`bool`, *optional*, defaults to `True`):
            Whether to include the left arm's joints.
        with_r_arm (`bool`, *optional*, defaults to `True`):
            Whether to include the right arm's joints.
        with_neck (`bool`, *optional*, defaults to `True`):
            Whether to include the neck's joints.
        with_antennas (`bool`, *optional*, defaults to `True`):
            Whether to include the antennas' joints.
        with_left_teleop_camera (`bool`, *optional*, defaults to `False`):
            Whether to add the left teleoperation camera to observations.
        with_right_teleop_camera (`bool`, *optional*, defaults to `False`):
            Whether to add the right teleoperation camera to observations.
        with_torso_camera (`bool`, *optional*, defaults to `False`):
            Whether to add the torso RGB camera to observations.
        camera_width (`int`, *optional*, defaults to 640):
            Frame width for the built-in cameras. Their frame rate is fixed at 30 and is not configurable.
        camera_height (`int`, *optional*, defaults to 480):
            Frame height for the built-in cameras.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Additional cameras beyond the three built-in ones. The `with_*_camera` flags populate this
            field, so anything set here is merged with them.
        id (`str`, *optional*):
            Identifier for this particular robot.
        calibration_dir (`Path`, *optional*):
            Unused: Reachy 2 manages its own calibration.
    """

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors.
    max_relative_target: float | None = None

    # IP address of the Reachy 2 robot
    ip_address: str | None = "localhost"
    # Port of the Reachy 2 robot
    port: int = 50065

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
    # By default, no camera is used.
    with_left_teleop_camera: bool = False
    with_right_teleop_camera: bool = False
    with_torso_camera: bool = False

    # Camera parameters
    camera_width: int = 640
    camera_height: int = 480

    # For cameras other than the 3 default Reachy 2 cameras.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Add the built-in cameras selected by the `with_*_camera` flags and validate the part selection.

        Raises:
            ValueError: If every robot part is disabled, which would leave no joints to control.
        """
        # Add cameras with same ip_address as the robot
        if self.with_left_teleop_camera:
            self.cameras["teleop_left"] = Reachy2CameraConfig(
                name="teleop",
                image_type="left",
                ip_address=self.ip_address,
                port=self.port,
                width=self.camera_width,
                height=self.camera_height,
                fps=30,  # Not configurable for Reachy 2 cameras
                color_mode=ColorMode.RGB,
            )
        if self.with_right_teleop_camera:
            self.cameras["teleop_right"] = Reachy2CameraConfig(
                name="teleop",
                image_type="right",
                ip_address=self.ip_address,
                port=self.port,
                width=self.camera_width,
                height=self.camera_height,
                fps=30,  # Not configurable for Reachy 2 cameras
                color_mode=ColorMode.RGB,
            )
        if self.with_torso_camera:
            self.cameras["torso_rgb"] = Reachy2CameraConfig(
                name="depth",
                image_type="rgb",
                ip_address=self.ip_address,
                port=self.port,
                width=self.camera_width,
                height=self.camera_height,
                fps=30,  # Not configurable for Reachy 2 cameras
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
