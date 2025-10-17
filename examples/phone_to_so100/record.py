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
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEE,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"

# Create the robot and teleoperator configurations
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem5A460814411",
    id="my_awesome_follower_arm",
    cameras=camera_config,
    use_degrees=True,
)
teleop_config = PhoneConfig(phone_os=PhoneOS.IOS)  # or PhoneOS.ANDROID

# Initialize the robot and teleoperator
robot = SO100Follower(robot_config)
phone = Phone(teleop_config)

# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="./SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Build pipeline to convert phone action to EE action
phone_to_robot_ee_pose_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    steps=[
        MapPhoneActionToRobotAction(platform=teleop_config.phone_os),
        EEReferenceAndDelta(
            kinematics=kinematics_solver,
            end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
            motor_names=list(robot.bus.motors.keys()),
            use_latched_reference=True,
        ),
        EEBoundsAndSafety(
            end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            max_ee_step_m=0.20,
        ),
        GripperVelocityToJoint(speed_factor=20.0),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Build pipeline to convert EE action to joints action
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

# Build pipeline to convert joint observation to EE observation
robot_joints_to_ee_pose = RobotProcessorPipeline[RobotObservation, RobotObservation](
    steps=[
        ForwardKinematicsJointsToEE(kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys()))
    ],
    to_transition=observation_to_transition,
    to_output=transition_to_observation,
)

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=HF_REPO_ID,
    fps=FPS,
    features=combine_feature_dicts(
        # Run the feature contract of the pipelines
        # This tells you how the features would look like after the pipeline steps
        aggregate_pipeline_dataset_features(
            pipeline=phone_to_robot_ee_pose_processor,
            initial_features=create_initial_features(action=phone.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_joints_to_ee_pose,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    ),
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Connect the robot and teleoperator
robot.connect()
phone.connect()

# Initialize the keyboard listener and rerun visualization
listener, events = init_keyboard_listener()
init_rerun(session_name="phone_so100_record")

if not robot.is_connected or not phone.is_connected:
    raise ValueError("Robot or teleop is not connected!")


print("Starting record loop. Move your phone to teleoperate the robot...")
episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    # Main record loop
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=phone,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=phone_to_robot_ee_pose_processor,
        robot_action_processor=robot_ee_to_joints_processor,
        robot_observation_processor=robot_joints_to_ee_pose,
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
            teleop_action_processor=phone_to_robot_ee_pose_processor,
            robot_action_processor=robot_ee_to_joints_processor,
            robot_observation_processor=robot_joints_to_ee_pose,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
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
phone.disconnect()
listener.stop()

dataset.finalize()
dataset.push_to_hub()
