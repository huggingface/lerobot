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

"""
Phone-to-SO101 replay example.

Replays a recorded dataset on the SO-101 arm by converting EE actions
back to joint commands via IK.
"""

import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

# --- Configuration (update these for your setup) ---
EPISODE_IDX = 0
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"

ROBOT_PORT = "/dev/tty.usbmodem5AAF2879361"
ROBOT_ID = "follower"

URDF_PATH = str(
    Path(__file__).resolve().parents[2] / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
)

# Initialize the robot
robot_config = SO101FollowerConfig(
    port=ROBOT_PORT, id=ROBOT_ID, use_degrees=True
)
robot = SO101Follower(robot_config)

# Initialize kinematics solver
kinematics_solver = RobotKinematics(
    urdf_path=URDF_PATH,
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Build pipeline: EE action → joint commands
robot_ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    steps=[
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver,
            motor_names=list(robot.bus.motors.keys()),
            initial_guess_current_joints=False,  # Open-loop for replay
        ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Fetch the dataset to replay
dataset = LeRobotDataset(HF_REPO_ID, episodes=[EPISODE_IDX])
episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_IDX)
actions = episode_frames.select_columns(ACTION)

# Connect to the robot
robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

print("Starting replay loop...")
log_say(f"Replaying episode {EPISODE_IDX}")
try:
    for idx in range(len(episode_frames)):
        t0 = time.perf_counter()

        # Get recorded action from dataset
        ee_action = {
            name: float(actions[idx][ACTION][i])
            for i, name in enumerate(dataset.features[ACTION]["names"])
        }

        # Get robot observation
        robot_obs = robot.get_observation()

        # Dataset EE → robot joints
        joint_action = robot_ee_to_joints_processor((ee_action, robot_obs))

        # Send action to robot
        _ = robot.send_action(joint_action)

        busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))
except KeyboardInterrupt:
    print("\nReplay stopped.")
finally:
    robot.disconnect()
    print("Disconnected.")
