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
High-level dataflow
───────────────────

TODO(pepijn): add schema
"""

import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.pipeline import RobotProcessor
from lerobot.processor.utils import (
    DatasetFeatureType,
    merge_transitions,
    pipeline_to_robot_action,
    prepare_robot_observation_pipeline,
    prepare_teleop_action_pipeline,
    transition_to_dataset_batch,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    AddRobotObservationAsComplimentaryData,
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEE,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone import Phone
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "pepijn223/phone_teleop_pipeline_9"

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

phone_to_robot_ee_pose = RobotProcessor(
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
            max_ee_twist_step_rad=0.20,
        ),
    ],
    name="post_teleop_pipeline",
)

robot_ee_to_joints = RobotProcessor(
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

robot_joints_to_ee_pose = RobotProcessor(
    steps=[
        ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys()))
    ],
    name="post_robot_pipeline",
)

ee_action_features = phone_to_robot_ee_pose.to_dataset_features(
    initial_features=phone.action_features,
    use_videos=True,
    include=("action",),
    action_type=DatasetFeatureType.EE,
)

ee_observation_features = robot_joints_to_ee_pose.to_dataset_features(
    initial_features=robot.observation_features,
    use_videos=True,
    include=("observation",),
    action_type=DatasetFeatureType.EE,
)

print("All dataset features: ", {**ee_action_features, **ee_observation_features})

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_REPO_ID,
    fps=FPS,
    features={**ee_action_features, **ee_observation_features},
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
# TODO(pepijn): add back record loop and integrate functionality below in it, so record_loop works with pipeline and without it
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    start_t = time.perf_counter()
    while time.perf_counter() - start_t < EPISODE_TIME_SEC and not events["exit_early"]:
        loop_t = time.perf_counter()

        # Read
        teleop_action = phone.get_action()
        robot_observation = robot.get_observation()

        # Prepare for pipeline
        teleop_action_tr = prepare_teleop_action_pipeline(
            teleop_action
        )  # TODO(pepijn): replace with to transition of Pipeline
        robot_observation_tr = prepare_robot_observation_pipeline(
            robot_observation
        )  # TODO(pepijn): replace with to transition of Pipeline

        # Run pipelines to get ee pose and corresponding joints action
        ee_pose_action = phone_to_robot_ee_pose(teleop_action_tr)
        joints_action = robot_ee_to_joints(ee_pose_action)

        # Send to robot
        robot_action = pipeline_to_robot_action(
            joints_action
        )  # TODO(pepijn): replace with from transition of Pipeline
        robot.send_action(robot_action)

        # Run pipeline to get ee pose observation
        ee_pose_observation = robot_joints_to_ee_pose(robot_observation_tr)

        # Merge and write
        merged = merge_transitions(ee_pose_observation, ee_pose_action)
        frame = transition_to_dataset_batch(merged)
        dataset.add_frame(frame, task=TASK_DESCRIPTION)

        dt = time.perf_counter() - loop_t
        time.sleep(max(0.0, 1.0 / FPS - dt))

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
