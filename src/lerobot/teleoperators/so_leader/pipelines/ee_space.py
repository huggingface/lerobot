#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Forward-kinematics pipeline for SO-100/101 leader (teleoperator) arm.

Converts raw leader joint positions into end-effector pose. Attach this to a leader
via ``set_output_pipeline`` so that ``get_action()`` returns EE coordinates instead of
raw joint angles.

Example::

    from lerobot.teleoperators.so_leader.pipelines import make_so10x_leader_fk_pipeline

    motor_names = list(leader.bus.motors.keys())
    leader.set_output_pipeline(make_so10x_leader_fk_pipeline(URDF_PATH, motor_names))
    action = leader.get_action()  # now contains ee.x, ee.y, ee.z, ...
"""

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import ForwardKinematicsJointsToEE

_DEFAULT_GRIPPER_FRAME = "gripper_frame_link"


def make_so10x_leader_fk_pipeline(
    urdf_path: str,
    motor_names: list[str],
    *,
    target_frame_name: str = _DEFAULT_GRIPPER_FRAME,
) -> RobotProcessorPipeline[RobotAction, RobotAction]:
    """
    Create a forward-kinematics action pipeline for SO-100/101 leader teleoperators.

    Converts raw leader joint positions (action) into end-effector pose (position +
    orientation + gripper). Attach this to a leader via ``set_output_pipeline`` so that
    ``get_action()`` returns EE coordinates instead of raw joint angles.

    Args:
        urdf_path: Path to the SO-100/101 URDF file used for kinematics.
        motor_names: Ordered list of motor names matching the URDF joint names.
        target_frame_name: Name of the end-effector frame in the URDF.

    Returns:
        A RobotProcessorPipeline that maps joint actions to EE actions.

    Example::

        motor_names = list(leader.bus.motors.keys())
        leader.set_output_pipeline(
            make_so10x_leader_fk_pipeline("./so101.urdf", motor_names)
        )
        action = leader.get_action()  # returns ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_vel
    """
    kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=target_frame_name,
        joint_names=motor_names,
    )
    return RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=motor_names)],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )
