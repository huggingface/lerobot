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
from lerobot.processor import RobotProcessor, TransitionKey
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    InverseKinematics,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone import Phone
from lerobot.teleoperators.phone.phone_processor import PhoneAxisRemapToAction

robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58760434471",
    id="so100_follower_end_effector",
)
robot = SO100Follower(robot_config)

# TODO(pepijn): Add LeKiwi example

teleop_config = PhoneConfig(phone_os=PhoneOS.IOS)
teleop_device = Phone(teleop_config)

# Path to URDF file for kinematics
# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
# https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="./src/lerobot/teleoperators/sim/so101_new_calib.urdf", target_frame_name="gripper_frame_link"
)

ee_ref = EEReferenceAndDelta(
    kinematics=kinematics_solver,
    end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
)
ee_safety = EEBoundsAndSafety(
    end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
    max_ee_step_m=0.1,
)
ik_step = InverseKinematics(
    kinematics=kinematics_solver,
    motor_names=list(robot.bus.motors.keys()),
    gripper_speed_factor=5.0,
)

# Create the robot pipeline
robot_pipeline = RobotProcessor(steps=[ee_ref, ee_safety, ik_step], name="so100_phone_teleop_pipeline")

phone_pipeline = RobotProcessor(
    steps=[PhoneAxisRemapToAction(platform=teleop_config.phone_os)],
    name="phone_mapping_pipeline",
)

robot.connect()
teleop_device.connect()

print("Starting teleop loop. Move your phone to teleoperate the robot.")

while True:
    phone_obs = teleop_device.get_action()
    if not phone_obs:
        time.sleep(0.01)
        continue

    # Map phone obs -> end-effector action
    tr_phone = {TransitionKey.OBSERVATION: phone_obs}
    tr_phone = phone_pipeline(tr_phone)
    ee_action = tr_phone.get(TransitionKey.ACTION, {})

    # Get robot obs and convert EE action -> joint action
    robot_obs = robot.get_observation()
    tr_robot = {
        TransitionKey.OBSERVATION: robot_obs,
        TransitionKey.ACTION: ee_action,
    }
    tr_robot = robot_pipeline(tr_robot)
    joint_action = tr_robot.get(TransitionKey.ACTION, {})

    if joint_action:
        robot.send_action(joint_action)

    time.sleep(0.01)
