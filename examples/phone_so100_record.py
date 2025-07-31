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

"""
Pipeline schema for phone teleop:

[TELEOP]
   │
   ▼
post_teleop_pipeline      # (deltas → safe EE)
   │
   └──► Dataset Action: {ee.{x,y,z,qx,qy,qz,qw}, gripper}
   │
   ▼
pre_robot_pipeline        # (IK → joints)
   │
   ▼
[ROBOT] send_action(joints)
   │
   ▼
post_robot_pipeline       # (FK → EE pose)
   │
   └──► Dataset Observation: {images, ee.{x,y,z,qx,qy,qz,qw}, gripper}
"""

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessor
from lerobot.record import record_loop
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEE,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone import Phone
from lerobot.teleoperators.phone.phone_processor import PhoneAxisRemapToAction
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "pepijn223/phone_so100_record_test0"

# Create the robot and teleoperator configurations
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58760434471",
    id="my_phone_teleop_follower_arm",
    cameras=camera_config,
)
teleop_config = PhoneConfig(phone_os=PhoneOS.IOS)  # or PhoneOS.ANDROID

# Initialize the robot and teleoperator
robot = SO100Follower(robot_config)
phone = Phone(teleop_config)

# Recommended URDF (from SO-ARM100 repo): https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="./src/lerobot/teleoperators/sim/so101_new_calib.urdf", target_frame_name="gripper_frame_link"
)


post_teleop = RobotProcessor(
    steps=[
        PhoneAxisRemapToAction(platform=teleop_config.phone_os),
        EEReferenceAndDelta(
            kinematics=kinematics_solver,
            end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
            motor_names=list(robot.bus.motors.keys()),
        ),
        EEBoundsAndSafety(
            end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            max_ee_step_m=0.10,
        ),
    ],
    name="post_teleop_pipeline",
)

pre_robot = RobotProcessor(
    steps=[
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver,
            motor_names=list(robot.bus.motors.keys()),
        ),
        GripperVelocityToJoint(
            motor_names=list(robot.bus.motors.keys()),
            speed_factor=5.0,
        ),
    ],
    name="pre_robot_pipeline",
)

post_robot = RobotProcessor(
    steps=[
        ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys()))
    ],
    name="post_robot_pipeline",
)

# Configure dataset features in EE space
ee_action_features = post_teleop.feature_contract(initial_features=phone.action_features)
ee_obs_features = post_robot.feature_contract(initial_features=robot.observation_features)

dataset_features = {ee_action_features, ee_obs_features}

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_REPO_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="recording_phone")

# Connect the robot and teleoperator
robot.connect()
phone.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(  # TODO(pepijn): Integrate post_teleop_processor, pre_robot_processor and post_robot_processor in record_loop
        robot=robot,
        events=events,
        fps=FPS,
        teleop=phone,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        post_teleop_processor=post_teleop,
        pre_robot_processor=pre_robot,
        post_robot_processor=post_robot,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=phone,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            post_teleop_processor=post_teleop,
            pre_robot_processor=pre_robot,
            post_robot_processor=post_robot,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
phone.disconnect()
dataset.push_to_hub()
