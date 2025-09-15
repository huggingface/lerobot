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
from lerobot.processor import RobotAction, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    AddRobotObservationAsComplimentaryData,
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader

follower_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem5A460814411", id="my_awesome_follower_arm", use_degrees=True
)

leader_config = SO100LeaderConfig(port="/dev/tty.usbmodem5A460819811", id="my_awesome_leader_arm")

follower = SO100Follower(follower_config)
leader = SO100Leader(leader_config)

follower_kinematics_solver = RobotKinematics(
    urdf_path="./examples/phone_to_so100/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(follower.bus.motors.keys()),
)

leader_kinematics_solver = RobotKinematics(
    urdf_path="./examples/phone_to_so100/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(leader.bus.motors.keys()),
)


leader_to_ee = RobotProcessorPipeline[RobotAction, RobotAction](
    steps=[
        ForwardKinematicsJointsToEE(
            kinematics=leader_kinematics_solver, motor_names=list(leader.bus.motors.keys())
        ),
    ],
    to_transition=robot_action_to_transition,
    to_output=transition_to_robot_action,
)

ee_to_follower_joints = RobotProcessorPipeline[RobotAction, RobotAction](
    [
        AddRobotObservationAsComplimentaryData(robot=follower),
        EEBoundsAndSafety(
            end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            max_ee_step_m=0.10,
            max_ee_twist_step_rad=0.50,
        ),
        InverseKinematicsEEToJoints(
            kinematics=follower_kinematics_solver,
            motor_names=list(follower.bus.motors.keys()),
            initial_guess_current_joints=False,
        ),
    ],
    to_transition=robot_action_to_transition,
    to_output=transition_to_robot_action,
)

follower.connect()
leader.connect()

print("Starting teleop loop. Move your phone to teleoperate the robot.")
while True:
    # Get leader joints observations
    leader_joints_obs = leader.get_action()

    # Convert them to EE
    leader_ee_act = leader_to_ee(leader_joints_obs)

    # Convert EE to follower joints actions
    follower_joints_act = ee_to_follower_joints(leader_ee_act)

    if follower_joints_act:
        follower.send_action(follower_joints_act)

    time.sleep(0.01)
