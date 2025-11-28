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
# See the License for the specific language governing permissions and
# limitations under the License.

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.earthrover_mini_plus import EarthRoverMiniPlus, EarthRoverMiniPlusConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say

# Configuration
NUM_EPISODES = 2
FPS = 10  # Lower FPS for cloud-based robot
EPISODE_TIME_SEC = 30
RESET_TIME_SEC = 5
TASK_DESCRIPTION = "Navigate around obstacles"
HF_REPO_ID = "lilkm/earthrover-navigation"  # Update with your username


def main():
    # Create the robot and teleoperator configurations
    robot_config = EarthRoverMiniPlusConfig()
    teleop_config = KeyboardRoverTeleopConfig(
        linear_speed=50.0,
        angular_speed=30.0,
        speed_increment=10.0
    )

    # Initialize the robot and teleoperator
    robot = EarthRoverMiniPlus(robot_config)
    teleop = KeyboardRoverTeleop(teleop_config)

    # Create processors (use default for now)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # Create the dataset
    log_say(f"Creating dataset: {HF_REPO_ID}")
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=2,  # Lower threads for cloud robot
    )

    # Connect the robot and teleoperator
    log_say("Connecting to robot (Frodobots SDK at localhost:8000)...")
    robot.connect()
    log_say("Connecting keyboard teleoperator...")
    teleop.connect()

    # Initialize the keyboard listener
    listener, events = init_keyboard_listener()

    if not robot.is_connected or not teleop.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    log_say("Starting record loop...")
    log_say(f"Controls: WASD (movement), Q/E (rotation), Space (stop), ESC (exit)")
    
    recorded_episodes = 0
    while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {recorded_episodes + 1}/{NUM_EPISODES}")
        log_say(f"Task: {TASK_DESCRIPTION}")

        # Main record loop
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            dataset=dataset,
            teleop=teleop,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and (
            (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
        ):
            log_say("Move robot back to starting position")
            log_say(f"You have {RESET_TIME_SEC} seconds to reset")
            
            # Reconnect teleop if needed (ESC disconnects it)
            if not teleop.is_connected:
                teleop = KeyboardRoverTeleop(teleop_config)
                teleop.connect()
            
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # Save episode
        dataset.save_episode()
        recorded_episodes += 1
        log_say(f"Saved episode {recorded_episodes}/{NUM_EPISODES}")

    # Clean up
    log_say("Stopping recording")
    robot.disconnect()
    if teleop.is_connected:
        teleop.disconnect()
    listener.stop()

    # Finalize and upload
    log_say("Finalizing dataset...")
    dataset.finalize()
    
    log_say(f"Pushing dataset to HuggingFace Hub: {HF_REPO_ID}")
    dataset.push_to_hub()
    
    log_say("✓ Dataset recording complete!")
    log_say(f"View your dataset at: https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
