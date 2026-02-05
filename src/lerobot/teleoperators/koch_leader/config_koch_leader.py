#!/usr/bin/env python

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

from dataclasses import dataclass

from ..config import TeleoperatorConfig
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import ForwardKinematicsJointsToEE


@TeleoperatorConfig.register_subclass("koch_leader")
@dataclass
class KochLeaderConfig(TeleoperatorConfig):
    # Port to connect to the arm
    port: str

    # Sets the arm in torque mode with the gripper motor set to this value. This makes it possible to squeeze
    # the gripper and have it spring back to an open position on its own.
    gripper_open_pos: float = 50.0

    # Set to `True` for backward compatibility with previous policies/dataset. Use degrees for FK / IK.
    use_degrees: bool = True


def make_koch_teleop_processors(teleop, display_data: bool) -> RobotProcessorPipeline[RobotAction, RobotAction]:
    """Create processor pipeline for single-arm Koch teleoperator.

    Converts joint angles from teleoperator to end-effector pose using forward kinematics.

    Args:
        teleop: Koch leader teleoperator instance
        display_data: Whether to enable visualization in rerun

    Returns:
        Pipeline that converts teleop joint angles to EE pose
    """
    URDF_PATH = "/home/steven/research/lerobot/assets/koch_follower.urdf"
    teleop_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="ee_frame",
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
    )

    teleop_motor_names = list(teleop.bus.motors.keys())

    teleop_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=teleop_kinematics_solver,
                motor_names=teleop_motor_names,
                gripper_name="gripper",
                display_data=display_data,
                entity_path_prefix="leader",
                offset=0.2,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return teleop_to_ee
