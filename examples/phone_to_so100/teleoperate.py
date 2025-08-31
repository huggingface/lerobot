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
# See the License for the specif

import time

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessor
from lerobot.processor.converters import to_output_robot_action, to_transition_teleop_action
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    AddRobotObservationAsComplimentaryData,
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone

# Initialize the robot and teleoperator
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58760434471", id="my_awesome_follower_arm", use_degrees=True
)
teleop_config = PhoneConfig(phone_os=PhoneOS.IOS)  # or PhoneOS.ANDROID

# Initialize the robot and teleoperator
robot = SO100Follower(robot_config)
teleop_device = Phone(teleop_config)

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="./src/lerobot/teleoperators/sim/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Build pipeline to convert phone action to ee pose action to joint action
phone_to_robot_joints = RobotProcessor(
    steps=[
        MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
        AddRobotObservationAsComplimentaryData(robot=robot),
        EEReferenceAndDelta(
            kinematics=kinematics_solver,
            end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
            motor_names=list(robot.bus.motors.keys()),
        ),
        EEBoundsAndSafety(
            end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            max_ee_step_m=0.10,
            max_ee_twist_step_rad=0.50,
        ),
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver,
            motor_names=list(robot.bus.motors.keys()),
        ),
        GripperVelocityToJoint(
            motor_names=list(robot.bus.motors.keys()),
            speed_factor=20.0,
        ),
    ],
    to_transition=to_transition_teleop_action,
    to_output=to_output_robot_action,
)

robot.connect()
teleop_device.connect()

print("Starting teleop loop. Move your phone to teleoperate the robot.")
while True:
    # Get teleop observation
    phone_obs = teleop_device.get_action()

    # Phone -> EE pose -> Joints transition
    joint_action = phone_to_robot_joints(phone_obs)

    if joint_action:
        robot.send_action(joint_action)

    time.sleep(0.01)
