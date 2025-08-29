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
from lerobot.datasets.utils import merge_features
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.converters import (
    to_output_robot_action,
    to_transition_robot_observation,
    to_transition_teleop_action,
)
from lerobot.processor.pipeline import RobotProcessor
from lerobot.record import record_loop
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
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"

# Initialize the robot and teleoperator
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
robot_config = SO100FollowerConfig(
    port="/dev/tty.usbmodem58760434471",
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
    urdf_path="./src/lerobot/teleoperators/sim/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Build pipeline to convert phone action to ee pose action
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
            max_ee_step_m=0.20,
            max_ee_twist_step_rad=0.50,
        ),
    ],
    to_transition=to_transition_teleop_action,
    to_output=lambda tr: tr,
)

# Build pipeline to convert ee pose action to joint action
robot_ee_to_joints = RobotProcessor(
    steps=[
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver,
            motor_names=list(robot.bus.motors.keys()),
            initial_guess_current_joints=True,
        ),
        GripperVelocityToJoint(
            motor_names=list(robot.bus.motors.keys()),
            speed_factor=20.0,
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

# Build dataset ee action features
action_ee = aggregate_pipeline_dataset_features(
    pipeline=phone_to_robot_ee_pose,
    initial_features=phone.action_features,
    use_videos=True,
    patterns=["action.ee"],
)

# Get gripper pos action features
gripper = aggregate_pipeline_dataset_features(
    pipeline=robot_ee_to_joints,
    initial_features={},
    use_videos=True,
    patterns=["action.gripper.pos", "observation.state.gripper.pos"],
)

# Build dataset ee observation features
observation_ee = aggregate_pipeline_dataset_features(
    pipeline=robot_joints_to_ee_pose,
    initial_features=robot.observation_features,
    use_videos=True,
    patterns=["observation.state.ee"],
)

dataset_features = merge_features(action_ee, gripper, observation_ee)

print("All dataset features: ", dataset_features)

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

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=phone,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=phone_to_robot_ee_pose,
        robot_action_processor=robot_ee_to_joints,
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
            teleop_action_processor=phone_to_robot_ee_pose,
            robot_action_processor=robot_ee_to_joints,
            robot_observation_processor=robot_joints_to_ee_pose,
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
