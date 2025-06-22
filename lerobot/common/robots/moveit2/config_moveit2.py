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
from enum import Enum

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig


class ActionType(Enum):
    CARTESIAN_VELOCITY = "cartesian_velocity"
    JOINT_POSITION = "joint_position"
    # For future extension:
    JOINT_VELOCITY = "joint_velocity"


@dataclass
class MoveIt2InterfaceConfig:
    # Namespace used by MoveIt2 nodes
    namespace: str = ""

    # The MoveIt2 base link name.
    base_link: str = "base_link"

    # TODO: move to MoveIt2Config
    arm_joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
    )

    gripper_joint_name: str = "gripper_joint"

    # Only applicable if velocity control is used.
    max_linear_velocity: float = 0.05
    max_angular_velocity: float = 0.25  # rad/s

    # Only applicable if position control is used.
    min_joint_positions: list[float] | None = None
    max_joint_positions: list[float] | None = None

    gripper_open_position: float = 0.0
    gripper_close_position: float = 1.0


@dataclass
class MoveIt2Config(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Action type for controlling the robot. Can be 'cartesian_velocity' or 'joint_position'.
    action_type: ActionType = ActionType.JOINT_POSITION

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # MoveIt2 interface configuration
    moveit2_interface: MoveIt2InterfaceConfig = field(default_factory=MoveIt2InterfaceConfig)

    action_from_keyboard: bool = False


@RobotConfig.register_subclass("annin_ar4")
@dataclass
class AnninAR4Config(MoveIt2Config):
    """Annin Robotics AR4 robot configuration - extends MoveIt2Config with
    AR4-specific settings
    """

    moveit2_interface: MoveIt2InterfaceConfig = field(
        default_factory=lambda: MoveIt2InterfaceConfig(
            base_link="base_link",
            arm_joint_names=[
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ],
            gripper_joint_name="gripper_jaw1_joint",
            gripper_open_position=0.014,
            gripper_close_position=0.0,
        )
    )
