#!/usr/bin/env python

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
Inference script for a pi0 model trained with **relative EE actions**.

This uses the built-in ``DeriveStateFromActionStep`` (no-op at inference),
``RelativeActionsProcessorStep``, ``AbsoluteActionsProcessorStep``, and
``RelativeStateProcessorStep`` that are already wired into pi0's processor
pipeline.

The inference loop:
  1. Reads joint positions from the robot.
  2. Converts to EE pose via forward kinematics (FK).
     This produces ``observation.state`` with the current EE pose.
  3. The pi0 preprocessor:
     a) ``DeriveStateFromActionStep`` — no-op (state comes from robot).
     b) ``RelativeActionsProcessorStep`` caches the raw state.
     c) ``RelativeStateProcessorStep`` buffers prev state, stacks, subtracts.
     d) ``NormalizerProcessorStep`` normalizes state and actions.
  4. pi0 predicts relative action chunk.
  5. The pi0 postprocessor:
     a) ``UnnormalizerProcessorStep`` unnormalizes.
     b) ``AbsoluteActionsProcessorStep`` adds cached state → absolute EE.
  6. IK converts absolute EE → joint targets → robot.

Based on the so100_to_so100_EE/evaluate.py example.

Usage:
    python evaluate.py
"""

from __future__ import annotations

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.feature_utils import combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.processor import (
    RelativeStateProcessorStep,
    RobotProcessorPipeline,
    make_default_teleop_action_processor,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 5
FPS = 10
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "manipulation task"

HF_MODEL_ID = "<hf_username>/<model_repo_id>"
HF_DATASET_ID = "<hf_username>/<dataset_repo_id>"

# Latency compensation: skip this many steps from the start of each predicted
# action chunk. Formula: ceil(total_latency_ms / (1000 / FPS)).
# E.g. at 10Hz with ~200ms total system latency: ceil(200 / 100) = 2.
LATENCY_SKIP_STEPS = 0

# EE feature keys produced by ForwardKinematicsJointsToEE
EE_KEYS = ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]


def main():
    camera_config = {"wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5A460814411",
        id="my_awesome_follower_arm",
        cameras=camera_config,
        use_degrees=True,
    )
    robot = SO100Follower(robot_config)

    policy = PI0Policy.from_pretrained(HF_MODEL_ID)
    policy.config.latency_skip_steps = LATENCY_SKIP_STEPS

    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
    )

    # FK: joint observation → EE observation (produces observation.state)
    robot_joints_to_ee_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics_solver,
                motor_names=list(robot.bus.motors.keys()),
            )
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # IK: EE action → joint targets
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

    # Dataset handle for stats (used by preprocessor/postprocessor)
    dataset = LeRobotDataset.create(
        repo_id=HF_DATASET_ID,
        fps=FPS,
        features=combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=robot_joints_to_ee_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=make_default_teleop_action_processor(),
                initial_features=create_initial_features(
                    action={f"ee.{k}": PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for k in EE_KEYS}
                ),
                use_videos=True,
            ),
        ),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Build pre/post processors from the trained model.
    # The pi0 processor pipeline already includes:
    #   pre:  ... → RelativeStateProcessorStep → RelativeActionsProcessorStep → NormalizerProcessorStep
    #   post: UnnormalizerProcessorStep → AbsoluteActionsProcessorStep → ...
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=HF_MODEL_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # Find the relative state step (if present) so we can reset its buffer between episodes.
    _relative_state_steps = [s for s in preprocessor.steps if isinstance(s, RelativeStateProcessorStep)]

    robot.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="umi_pi0_relative_ee_evaluate")

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        print("Starting evaluate loop...")
        for episode_idx in range(NUM_EPISODES):
            log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

            # Reset relative state buffer so velocity is zero at episode start
            for step in _relative_state_steps:
                step.reset()

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
                robot_observation_processor=robot_joints_to_ee_processor,
            )

            if not events["stop_recording"] and (
                (episode_idx < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
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
                    robot_observation_processor=robot_joints_to_ee_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
    finally:
        log_say("Stop recording")
        robot.disconnect()
        listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
