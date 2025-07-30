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

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessor, TransitionKey
from lerobot.record import record_loop
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    InverseKinematics,
)
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone import Phone
from lerobot.teleoperators.phone.phone_processor import PhoneAxisRemapToAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "My task description"
HF_REPO_ID = "pepijn223/phone_so100_record_test0"


class PhoneToSO100Adapter(Teleoperator):
    """
    An adapter that makes the phone behave like a 'teleop' for record_loop.
    On each get_action():
      - read phone obs
      - map to EE action via phone pipeline
      - read robot obs
      - map EE action -> joint action via robot pipeline
      - return joint action (matches robot.action_features)
    """

    name = "phone_to_so100_adapter"
    config_class = PhoneConfig

    def __init__(
        self,
        phone: Phone,
        robot: SO100Follower,
        phone_postprocessor: RobotProcessor,
        robot_preprocessor: RobotProcessor,
    ):
        super().__init__(phone.config)
        self._phone = phone
        self._robot = robot
        self._phone_postprocessor = phone_postprocessor
        self._robot_preprocessor = robot_preprocessor

    def connect(self):
        self._phone.connect()

    def disconnect(self):
        self._phone.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._robot.is_connected

    def calibrate(self) -> None:
        self._phone.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return self._phone.is_calibrated

    @property
    def action_features(self) -> dict[str, type]:
        return self._phone.action_features

    @property
    def feedback_features(self) -> dict[str, type]:
        pass

    def configure(self) -> None:
        self._phone.configure()

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def get_action(self) -> dict:
        phone_obs = self._phone.get_action()
        if not phone_obs:
            return {}

        # Phone observation to EE action
        transition_phone = {TransitionKey.OBSERVATION: phone_obs}
        transition_phone = self._phone_postprocessor(transition_phone)
        ee_action = transition_phone.get(TransitionKey.ACTION, {})

        # EE action and robot obs to joint action
        robot_obs = self._robot.get_observation()
        transition_robot = {
            TransitionKey.OBSERVATION: robot_obs,
            TransitionKey.ACTION: ee_action,
        }
        transition_robot = self._robot_preprocessor(transition_robot)
        joint_action = transition_robot.get(TransitionKey.ACTION, {})

        return joint_action


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

# Create the pipelines
# Recommended URDF (from SO-ARM100 repo): https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="./src/lerobot/teleoperators/sim/so101_new_calib.urdf", target_frame_name="gripper_frame_link"
)

ee_ref = EEReferenceAndDelta(
    kinematics=kinematics_solver,
    end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},  # meters per unit input
    motor_names=list(robot.bus.motors.keys()),
)

ee_safety = EEBoundsAndSafety(
    end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
    max_ee_step_m=0.10,
)

ik_step = InverseKinematics(
    kinematics=kinematics_solver,
    motor_names=list(robot.bus.motors.keys()),
    gripper_speed_factor=5.0,
)

robot_preprocessor = RobotProcessor(
    steps=[ee_ref, ee_safety, ik_step],
    name="so100_phone_record_pipeline",
)

phone_postprocessor = RobotProcessor(
    steps=[PhoneAxisRemapToAction(platform=teleop_config.phone_os)],
    name="phone_mapping_pipeline",
)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

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

# Teleop adapter that record_loop will poll for actions
teleop_adapter = PhoneToSO100Adapter(
    phone=phone, robot=robot, phone_postprocessor=phone_postprocessor, robot_preprocessor=robot_preprocessor
)

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop_adapter,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop_adapter,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
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
