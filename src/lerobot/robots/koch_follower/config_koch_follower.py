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
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)

from ..config import RobotConfig


@RobotConfig.register_subclass("koch_follower")
@dataclass
class KochFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset. Use degrees for the IK.
    use_degrees: bool = True


def make_koch_robot_processors(robot, display_data: bool) -> RobotProcessorPipeline:
    """Create processor pipeline for single-arm Koch robot.

    Converts end-effector pose to joint angles using inverse kinematics with safety bounds.

    Args:
        robot: Koch follower robot instance
        display_data: Whether to enable visualization in rerun

    Returns:
        Pipeline that converts EE pose to robot joint angles
    """
    URDF_PATH = "/home/steven/research/lerobot/assets/koch_follower.urdf"
    robot_kinematics_solver = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="ee_frame",
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
    )

    robot_motor_names = list(robot.bus.motors.keys())

    ee_to_robot_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.25, -0.2, 0.0], "max": [0., 0.2, 0.4]},
                max_ee_step_m=0.15,
                max_ee_twist_step_rad=0.50,
            ),
            InverseKinematicsEEToJoints(
                kinematics=robot_kinematics_solver,
                motor_names=robot_motor_names,
                initial_guess_current_joints=False,
                display_data=display_data,
                entity_path_prefix="follower",
                offset=0.0,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return ee_to_robot_joints
