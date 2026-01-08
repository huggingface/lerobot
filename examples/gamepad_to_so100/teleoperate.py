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

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig
from lerobot.teleoperators.gamepad.gamepad_processor import MapGamepadActionToRobotAction
from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30

# Initialize the robot and teleoperator
robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0", id="black", use_degrees=True
)
teleop_config = GamepadTeleopConfig(use_gripper=False)

# Initialize the robot and teleoperator
robot = SO101Follower(robot_config)
teleop_device = GamepadTeleop(teleop_config)

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="./SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Build pipeline to convert gamepad action to ee pose action to joint action
gamepad_to_robot_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    steps=[
        MapGamepadActionToRobotAction(),
        EEReferenceAndDelta(
            kinematics=kinematics_solver,
            end_effector_step_sizes={"x": 0.02, "y": 0.01, "z": 0.05},  # ← SAFE: 1cm/frame max speed
            motor_names=list(robot.bus.motors.keys()),
            use_latched_reference=False,  # ← FALSE = velocity control (continuous movement)
        ),
        EEBoundsAndSafety(
            end_effector_bounds={"min": [0.05, -0.25, -0.05], "max": [0.35, 0.25, 0.45]},  # ← More conservative safe bounds
            max_ee_step_m=0.02,  # ← CRITICAL: Reduced from 0.10 to 0.02 (2cm max step prevents violent shaking!)
        ),
        GripperVelocityToJoint(
            speed_factor=20.0,
        ),
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver,
            motor_names=list(robot.bus.motors.keys()),
            initial_guess_current_joints=True,
        ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Connect to the robot and teleoperator
# run torque disable first
robot.connect()
teleop_device.connect()

# Init rerun viewer
init_rerun(session_name="gamepad_so100_teleop")

if not robot.is_connected or not teleop_device.is_connected:
    raise ValueError("Robot or teleop is not connected!")

print("Starting teleop loop. Use your gamepad to teleoperate the robot...")
try:
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        robot_obs = robot.get_observation()

        # Get teleop action
        gamepad_obs = teleop_device.get_action()

        # Gamepad -> EE pose -> Joints transition
        joint_action = gamepad_to_robot_joints_processor((gamepad_obs, robot_obs))

        # Send action to robot
        _ = robot.send_action(joint_action)

        # Visualize
        log_rerun_data(observation=gamepad_obs, action=joint_action)

        busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
except KeyboardInterrupt:
    print("Teleop loop interrupted by user")
finally:
    robot.disconnect()
    teleop_device.disconnect()

