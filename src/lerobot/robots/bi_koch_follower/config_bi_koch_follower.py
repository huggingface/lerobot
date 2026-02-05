#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from ..config import RobotConfig


MAX_RELATIVE_TARGET_DEFAULT = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 15.0,
    "elbow_flex": 15.0,
    "wrist_flex": 15.0,
    "wrist_roll": 20.0,
    "gripper": 30.0,
}


@RobotConfig.register_subclass("bi_koch_follower")
@dataclass
class BiKochFollowerConfig(RobotConfig):
    left_arm_port: str
    right_arm_port: str

    # Optional
    left_arm_disable_torque_on_disconnect: bool = True
    left_arm_max_relative_target: float | dict[str, float] | None = field(
        default_factory=lambda: MAX_RELATIVE_TARGET_DEFAULT.copy()
    )
    left_arm_use_degrees: bool = True
    right_arm_disable_torque_on_disconnect: bool = True
    right_arm_max_relative_target: float | dict[str, float] | None = field(
        default_factory=lambda: MAX_RELATIVE_TARGET_DEFAULT.copy()
    )
    right_arm_use_degrees: bool = True

    # cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


def make_bimanual_koch_robot_processors(robot, display_data: bool) -> RobotProcessorPipeline:
    # Build pipeline to convert teleop joints to EE action
    # PATH="assets/koch_follower.urdf"
    URDF_PATH = "/home/steven/research/lerobot/assets/koch_follower.urdf"
    left_robot_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="ee_frame",
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
    )
    right_robot_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="ee_frame",
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
    )

    robot_motor_names = list(robot.left_arm.bus.motors.keys())
    left_robot_motor_names = ["left_" + motor for motor in robot_motor_names]
    right_robot_motor_names = ["right_" + motor for motor in robot_motor_names]

    # build pipeline to convert EE action to robot joints
    ee_to_robot_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.25, -0.2, 0.0], "max": [0., 0.2, 0.4]},
                max_ee_step_m=0.15,
                # max_ee_twist_step_rad=0.50,
                prefix="left_",
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.25, -0.2, 0.0], "max": [0., 0.2, 0.4]},
                max_ee_step_m=0.15,
                # max_ee_twist_step_rad=0.50,
                prefix="right_",
            ),
            InverseKinematicsEEToJoints(
                kinematics=left_robot_kinematics_solver,
                motor_names=left_robot_motor_names,
                initial_guess_current_joints=False,
                prefix="left_",
                display_data=display_data,
                entity_path_prefix="follower_left",
                offset=0.0,
            ),
            InverseKinematicsEEToJoints(
                kinematics=right_robot_kinematics_solver,
                motor_names=right_robot_motor_names,
                initial_guess_current_joints=False,
                prefix="right_",
                display_data=display_data,
                entity_path_prefix="follower_right",
                offset=0.2,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return ee_to_robot_joints
