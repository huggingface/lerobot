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

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import DataProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    transition_to_robot_action,
)
from lerobot.record import record_loop
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    AddRobotObservationAsComplimentaryData,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "My task description"
HF_MODEL_ID = "<hf_username>/<model_repo_id>"
HF_DATASET_ID = "<hf_username>/<dataset_repo_id>"

# Initialize the robot with degrees
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58760434471",
    id="my_awesome_follower_arm",
    cameras=camera_config,
    use_degrees=True,
)

# Initialize the robot
robot = SO100Follower(robot_config)

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="./src/lerobot/teleoperators/sim/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Build pipeline to convert ee pose action to joint action
robot_ee_to_joints_processor = DataProcessorPipeline(
    steps=[
        AddRobotObservationAsComplimentaryData(robot=robot),
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver,
            motor_names=list(robot.bus.motors.keys()),
            initial_guess_current_joints=True,
        ),
    ],
    to_transition=lambda tr: tr,
    to_output=transition_to_robot_action,
)

# Build pipeline to convert joint observation to ee pose observation
robot_joints_to_ee_pose_processor = DataProcessorPipeline(
    steps=[
        ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys()))
    ],
    to_transition=observation_to_transition,
    to_output=lambda tr: tr,
)

# Build dataset action and gripper features
action_ee_and_gripper = aggregate_pipeline_dataset_features(
    pipeline=robot_ee_to_joints_processor,
    initial_features={},
    use_videos=True,
    patterns=["action.ee", "action.gripper.pos", "observation.state.gripper.pos"],
)  # Get all ee action features + gripper pos action features

# Build dataset observation features
obs_ee = aggregate_pipeline_dataset_features(
    pipeline=robot_joints_to_ee_pose_processor,
    initial_features=robot.observation_features,
    use_videos=True,
    patterns=["observation.state.ee"],
)  # Get all ee observation features

dataset_features = combine_feature_dicts(obs_ee, action_ee_and_gripper)

print("All dataset features: ", dataset_features)

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,
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

episode_idx = 0

policy = ACTPolicy.from_pretrained(HF_MODEL_ID)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=HF_MODEL_ID,
    dataset_stats=dataset.meta.stats,
)

for episode_idx in range(NUM_EPISODES):
    log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        robot_action_processor=robot_ee_to_joints_processor,
        robot_observation_processor=robot_joints_to_ee_pose_processor,
    )
    dataset.save_episode()

# Clean up
log_say("Stop recording")
robot.disconnect()
dataset.push_to_hub()
