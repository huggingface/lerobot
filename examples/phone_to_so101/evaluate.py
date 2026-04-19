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
Phone-to-SO101 evaluation example.

Runs a trained policy (e.g. ACT) on the SO-101 arm with EE-space
actions converted to joint commands via IK.
"""

from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_teleop_action_processor,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# --- Configuration (update these for your setup) ---
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "My task description"
HF_MODEL_ID = "<hf_username>/<model_repo_id>"
HF_DATASET_ID = "<hf_username>/<dataset_repo_id>"

ROBOT_PORT = "/dev/tty.usbmodem5AAF2879361"
ROBOT_ID = "follower"

URDF_PATH = str(
    Path(__file__).resolve().parents[2] / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
)

# Create the robot configuration & robot
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
robot_config = SO101FollowerConfig(
    port=ROBOT_PORT,
    id=ROBOT_ID,
    cameras=camera_config,
    use_degrees=True,
)

robot = SO101Follower(robot_config)

# Create policy
policy = ACTPolicy.from_pretrained(HF_MODEL_ID)

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
            initial_guess_current_joints=True,
        ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Build pipeline: joint observation → EE observation
robot_joints_to_ee_pose_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
    steps=[
        ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys()))
    ],
    to_transition=observation_to_transition,
    to_output=transition_to_observation,
)

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,
    fps=FPS,
    features=combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_joints_to_ee_pose_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=make_default_teleop_action_processor(),
            initial_features=create_initial_features(
                action={
                    f"ee.{k}": PolicyFeature(type=FeatureType.ACTION, shape=(1,))
                    for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]
                }
            ),
            use_videos=True,
        ),
    ),
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Build Policy Processors
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy,
    pretrained_path=HF_MODEL_ID,
    dataset_stats=dataset.meta.stats,
    preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
)

# Connect the robot
robot.connect()

# Initialize the keyboard listener and rerun visualization
listener, events = init_keyboard_listener()
init_rerun(session_name="phone_so101_evaluate")

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

print("Starting evaluate loop...")
episode_idx = 0
for episode_idx in range(NUM_EPISODES):
    log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

    # Main record loop
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
        teleop_action_processor=make_default_teleop_action_processor(),
        robot_action_processor=robot_ee_to_joints_processor,
        robot_observation_processor=robot_joints_to_ee_pose_processor,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and ((episode_idx < NUM_EPISODES - 1) or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=make_default_teleop_action_processor(),
            robot_action_processor=robot_ee_to_joints_processor,
            robot_observation_processor=robot_joints_to_ee_pose_processor,
        )

    if events["rerecord_episode"]:
        log_say("Re-record episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    # Save episode
    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
listener.stop()

dataset.finalize()
dataset.push_to_hub()
