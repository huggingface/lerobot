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

import logging
import time

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import init_keyboard_listener, predict_action
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.model.kinematics import RobotKinematics
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    RobotProcessorPipeline,
    make_default_teleop_action_processor,
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
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "My task description"
HF_MODEL_ID = "<hf_username>/<model_repo_id>"
HF_DATASET_ID = "<hf_username>/<dataset_repo_id>"


def main():
    # NOTE: For production policy deployment, use `lerobot-rollout` CLI instead.
    # This script provides a self-contained example for educational purposes.

    # Create the robot configuration & robot
    camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem5A460814411",
        id="my_awesome_follower_arm",
        cameras=camera_config,
        use_degrees=True,
    )

    robot = SO100Follower(robot_config)

    # Create policy
    policy = ACTPolicy.from_pretrained(HF_MODEL_ID)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    kinematics_solver = RobotKinematics(
        urdf_path="./SO101/so101_new_calib.urdf",
        target_frame_name="gripper_frame_link",
        joint_names=list(robot.bus.motors.keys()),
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

    # Build pipeline to convert joints observation to EE observation
    robot_joints_to_ee_pose_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics_solver, motor_names=list(robot.bus.motors.keys())
            )
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
            # User for now should be explicit on the feature keys that were used for record
            # Alternatively, the user can pass the processor step that has the right features
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
        # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # Connect the robot and teleoperator
    robot.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="so100_so100_evaluate")

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        print("Starting evaluate loop...")
        control_interval = 1 / FPS
        episode_idx = 0
        for episode_idx in range(NUM_EPISODES):
            log_say(f"Running inference, recording eval episode {episode_idx + 1} of {NUM_EPISODES}")

            # Inline evaluation loop: predict actions and send to robot
            timestamp = 0
            start_episode_t = time.perf_counter()
            while timestamp < EPISODE_TIME_SEC:
                start_loop_t = time.perf_counter()

                if events["exit_early"]:
                    events["exit_early"] = False
                    break

                # Get robot observation
                obs = robot.get_observation()
                obs_processed = robot_joints_to_ee_pose_processor(obs)
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

                # Predict action using the policy
                action_tensor = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=policy.config.device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.device.type == "cuda",
                    task=TASK_DESCRIPTION,
                    robot_type=robot.name,
                )

                # Convert policy output to robot action dict
                action_values = make_robot_action(action_tensor, dataset.features)

                # Process and send action to robot (EE -> joints via IK)
                robot_action_to_send = robot_ee_to_joints_processor((action_values, obs))
                robot.send_action(robot_action_to_send)

                # Write to dataset
                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                frame = {**observation_frame, **action_frame, "task": TASK_DESCRIPTION}
                dataset.add_frame(frame)

                log_rerun_data(observation=obs_processed, action=action_values)

                dt_s = time.perf_counter() - start_loop_t
                sleep_time_s = control_interval - dt_s
                if sleep_time_s < 0:
                    logging.warning(
                        f"Evaluate loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({FPS} Hz)."
                    )
                precise_sleep(max(sleep_time_s, 0.0))
                timestamp = time.perf_counter() - start_episode_t

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (
                (episode_idx < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                log_say("Waiting for environment reset, press right arrow key when ready...")

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
    finally:
        # Clean up
        log_say("Stop recording")
        robot.disconnect()
        listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
