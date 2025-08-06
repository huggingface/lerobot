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
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation

from ..config import RobotConfig


@RobotConfig.register_subclass("reachy2")
@dataclass
class Reachy2RobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None
    ip_address: str | None = "localhost"
    use_external_commands: bool = False
    with_mobile_base: bool = True

    mock: bool = False

    def __post_init__(self):
        # cameras
        self.cameras: dict[str, CameraConfig] = {
                "teleop_left": Reachy2CameraConfig(
                        name="teleop",
                        image_type="left",
                        ip_address=self.ip_address,
                        fps=30,
                        width=640,
                        height=480,
                        color_mode=ColorMode.RGB,
                        rotation=Cv2Rotation.NO_ROTATION
                ),
                "teleop_right": Reachy2CameraConfig(
                        name="teleop",
                        image_type="right",
                        ip_address=self.ip_address,
                        fps=30,
                        width=640,
                        height=480,
                        color_mode=ColorMode.RGB,
                        rotation=Cv2Rotation.NO_ROTATION
                ),
        }
        super().__post_init__()


# #     cameras
#     cameras: dict[str, CameraConfig] = field(
#         default_factory=lambda: {
#                 "teleop_left": Reachy2CameraConfig(
#                         name="teleop",
#                         image_type="left",
#                         fps=30,
#                         width=640,
#                         height=480,
#                         color_mode=ColorMode.RGB,
#                         rotation=Cv2Rotation.NO_ROTATION
#                 ),
#                 "teleop_right": Reachy2CameraConfig(
#                         name="teleop",
#                         image_type="right",
#                         fps=30,
#                         width=640,
#                         height=480,
#                         color_mode=ColorMode.RGB,
#                         rotation=Cv2Rotation.NO_ROTATION
#                 ),
#                 "torso_rgb": Reachy2CameraConfig(
#                         name="depth",
#                         image_type="rgb",
#                         fps=30,
#                         width=640,
#                         height=480,
#                         color_mode=ColorMode.RGB,
#                         rotation=Cv2Rotation.NO_ROTATION
#                 ),
#                 "torso_depth": Reachy2CameraConfig(
#                         name="depth",
#                         image_type="depth",
#                         fps=30,
#                         width=640,
#                         height=480,
#                         color_mode=ColorMode.RGB,
#                         rotation=Cv2Rotation.NO_ROTATION,
#                         use_depth=True
#                 )

#                 # REAL ROBOT
#                 "teleop_left": Reachy2CameraConfig(
#                         name="teleop",
#                         image_type="left",
#                         ip_address="192.168.0.199",
#                         # ip_address="172.18.131.66",
#                         fps=30,
#                         width=960,
#                         height=720,
#                         color_mode=ColorMode.RGB,
#                         rotation=Cv2Rotation.NO_ROTATION
#                 ),
#                 "teleop_right": Reachy2CameraConfig(
#                         name="teleop",
#                         image_type="right",
#                         ip_address="192.168.0.199",
#                         # ip_address="172.18.131.66",
#                         fps=30,
#                         width=960,
#                         height=720,
#                         color_mode=ColorMode.RGB,
#                         rotation=Cv2Rotation.NO_ROTATION
#                 ),
#                 "torso_rgb": Reachy2CameraConfig(
#                         name="depth",
#                         image_type="rgb",
#                         ip_address="172.18.131.66",
#                         fps=30,
#                         width=1280,
#                         height=720,
#                         color_mode=ColorMode.RGB,
#                         rotation=Cv2Rotation.NO_ROTATION
#                 ),

#                 # REAL ROBOT REDUCED IMAGE SIZE
#                 "teleop_left": Reachy2CameraConfig(
#                         name="teleop",
#                         image_type="left",
#                         ip_address="192.168.0.199",
#                         fps=30,
#                         width=640,
#                         height=480,
#                         color_mode=ColorMode.RGB,
#                         rotation=Cv2Rotation.NO_ROTATION
#                 ),
#                 "teleop_right": Reachy2CameraConfig(
#                         name="teleop",
#                         image_type="right",
#                         ip_address="192.168.0.199",
#                         fps=30,
#                         width=640,
#                         height=480,
#                         color_mode=ColorMode.RGB,
#                         rotation=Cv2Rotation.NO_ROTATION
#                 ),

#                 # Reduced size for testing
#                 "teleop_left": Reachy2CameraConfig(
#                         name="teleop",
#                         image_type="left",
#                         ip_address="172.18.131.66",
#                         fps=30,
#                         width=480,
#                         height=360,
#                         color_mode=ColorMode.RGB,
#                         rotation=Cv2Rotation.NO_ROTATION
#                 ),
#                 "teleop_right": Reachy2CameraConfig(
#                         name="teleop",
#                         image_type="right",
#                         ip_address="172.18.131.66",
#                         fps=30,
#                         width=480,
#                         height=360,
#                         color_mode=ColorMode.RGB,
#                         rotation=Cv2Rotation.NO_ROTATION
#                 ),
#         }
#     )
