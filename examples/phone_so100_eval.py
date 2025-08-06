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
from lerobot.configs.types import DatasetFeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import merge_grouped_features
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_processor
from lerobot.processor.pipeline import RobotProcessor
from lerobot.processor.utils import (
    to_dataset_frame,
    to_output_robot_action,
    to_transition_robot_observation,
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

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "Pickup the blue block"
HF_REPO_ID = "pepijn223/eval_phone_pipeline_pickup_block4"

# Initialize the robot and teleoperator
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58760434471",
    id="my_phone_teleop_follower_arm",
    cameras=camera_config,
    use_degrees=True,
)

# Initialize the robot and teleoperator
robot = SO100Follower(robot_config)

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
        ),
    ],
    to_transition=lambda tr: tr,
    to_output=to_output_robot_action,
)

# Build pipeline to convert joint observation to ee pose observation
robot_joints_to_ee_pose = RobotProcessor(
    steps=[
        ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys()))
    ],
    to_transition=to_transition_robot_observation,
    to_output=lambda tr: tr,
)

# Build dataset action features
action_ee = robot_ee_to_joints.aggregate_dataset_features(
    initial_features={},
    use_videos=True,
    include=("action",),
    action_type=[DatasetFeatureType.EE, DatasetFeatureType.JOINT],
)  # Get all ee action features + gripper pos action features

# Build dataset observation features
obs_ee = robot_joints_to_ee_pose.aggregate_dataset_features(
    initial_features=robot.observation_features,
    use_videos=True,
    include=("observation",),
    action_type=DatasetFeatureType.EE,
)  # Get all ee observation features
obs_joint = robot_ee_to_joints.aggregate_dataset_features(
    initial_features={},
    use_videos=True,
    include=("observation",),
    action_type=DatasetFeatureType.JOINT,
)  # Get gripper pos observation features
observation_features = merge_grouped_features(obs_ee, obs_joint)

print("All dataset features: ", {**action_ee, **observation_features})

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_REPO_ID,
    fps=FPS,
    features={**action_ee, **observation_features},
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Create a function to convert the pipelines output to the dataset format using the expected features
to_dataset_features = to_dataset_frame(dataset.features)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="recording_phone")

# Connect the robot and teleoperator
robot.connect()

episode_idx = 0

policy = ACTPolicy.from_pretrained("pepijn223/phone_pipeline_pickup1_migrated")
preprocessor, postprocessor = make_processor(
    policy_cfg=policy,
    pretrained_path="pepijn223/phone_pipeline_pickup1_migrated",
    dataset_stats=dataset.meta.stats,
    preprocessor_overrides={"device_processor": {"device": "mps"}},
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
        robot_action_processor=robot_ee_to_joints,
        robot_observation_processor=robot_joints_to_ee_pose,
        to_dataset_frame=to_dataset_features,
    )

    dataset.save_episode()

# Clean up
log_say("Stop recording")
robot.disconnect()
dataset.push_to_hub()
