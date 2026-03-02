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
End-effector space pipelines for SO-100/101 follower robots.

These factory functions return ready-to-use pipelines that convert between joint space
and Cartesian end-effector space. Attach them to a robot with ``set_output_pipeline`` /
``set_input_pipeline`` to enable EE-space recording and teleoperation.

Example::

    from lerobot.robots.so_follower.pipelines import (
        make_so10x_fk_observation_pipeline,
        make_so10x_ik_action_pipeline,
    )

    motor_names = list(follower.bus.motors.keys())
    follower.set_output_pipeline(make_so10x_fk_observation_pipeline(URDF_PATH, motor_names))
    follower.set_input_pipeline(make_so10x_ik_action_pipeline(URDF_PATH, motor_names))
"""

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)

_DEFAULT_EE_BOUNDS = {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
_DEFAULT_GRIPPER_FRAME = "gripper_frame_link"


def make_so10x_fk_observation_pipeline(
    urdf_path: str,
    motor_names: list[str],
    *,
    target_frame_name: str = _DEFAULT_GRIPPER_FRAME,
) -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    """
    Create a forward-kinematics observation pipeline for SO-100/101 follower robots.

    Converts raw joint positions (observation) into end-effector pose (position + orientation).
    Attach this to a follower robot via ``set_output_pipeline`` so that ``get_observation()``
    returns EE coordinates instead of raw joint angles.

    Args:
        urdf_path: Path to the SO-100/101 URDF file used for kinematics.
        motor_names: Ordered list of motor names matching the URDF joint names.
        target_frame_name: Name of the end-effector frame in the URDF.

    Returns:
        A RobotProcessorPipeline that maps joint observations to EE observations.

    Example::

        follower.set_output_pipeline(
            make_so10x_fk_observation_pipeline("./so101.urdf", motor_names)
        )
        obs = follower.get_observation()  # now contains ee.x, ee.y, ee.z, ...
    """
    kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=target_frame_name,
        joint_names=motor_names,
    )
    return RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=motor_names)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )


def make_so10x_ik_action_pipeline(
    urdf_path: str,
    motor_names: list[str],
    *,
    target_frame_name: str = _DEFAULT_GRIPPER_FRAME,
    end_effector_bounds: dict | None = None,
    max_ee_step_m: float = 0.10,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """
    Create an inverse-kinematics action pipeline for SO-100/101 follower robots.

    Converts incoming end-effector pose commands into joint positions, applying safety
    bounds and step-size limits before solving IK. The current joint positions are used
    as the IK initial guess (taken from the cached ``_last_raw_obs``).

    Attach this to a follower robot via ``set_input_pipeline`` so that ``send_action()``
    receives EE commands and translates them to motor positions before the hardware write.

    Args:
        urdf_path: Path to the SO-100/101 URDF file used for kinematics.
        motor_names: Ordered list of motor names matching the URDF joint names.
        target_frame_name: Name of the end-effector frame in the URDF.
        end_effector_bounds: Dict with ``"min"`` and ``"max"`` lists (3D position bounds in metres).
            Defaults to ``{"min": [-1, -1, -1], "max": [1, 1, 1]}``.
        max_ee_step_m: Maximum allowed EE position change per step in metres.

    Returns:
        A RobotProcessorPipeline that maps (EE action, raw obs) to joint action.

    Example::

        follower.set_input_pipeline(
            make_so10x_ik_action_pipeline("./so101.urdf", motor_names)
        )
        # send_action() now accepts ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_vel
    """
    kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=target_frame_name,
        joint_names=motor_names,
    )
    bounds = end_effector_bounds or _DEFAULT_EE_BOUNDS
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEBoundsAndSafety(end_effector_bounds=bounds, max_ee_step_m=max_ee_step_m),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
