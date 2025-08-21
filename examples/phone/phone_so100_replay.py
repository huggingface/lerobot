# !/usr/bin/env python

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

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.converters import to_output_robot_action, to_transition_teleop_action
from lerobot.processor.pipeline import RobotProcessor
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    AddRobotObservationAsComplimentaryData,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

EPISODE_IDX = 0
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"

robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58760434471", id="my_awesome_follower_arm", use_degrees=True
)
robot = SO100Follower(robot_config)
robot.connect()

dataset = LeRobotDataset(HF_REPO_ID, episodes=[EPISODE_IDX])
actions = dataset.hf_dataset.select_columns("action")

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="./src/lerobot/teleoperators/sim/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Build pipeline to convert ee pose action to joint action
robot_ee_to_joints = RobotProcessor(
    steps=[
        AddRobotObservationAsComplimentaryData(robot=robot),
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver,
            motor_names=list(robot.bus.motors.keys()),
            initial_guess_current_joints=False,  # Because replay is open loop
        ),
    ],
    to_transition=to_transition_teleop_action,
    to_output=to_output_robot_action,
)

robot_ee_to_joints.reset()

log_say(f"Replaying episode {EPISODE_IDX}")
for idx in range(dataset.num_frames):
    t0 = time.perf_counter()

    ee_action = {
        name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
    }

    joint_action = robot_ee_to_joints(ee_action)
    action_sent = robot.send_action(joint_action)

    busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

robot.disconnect()
