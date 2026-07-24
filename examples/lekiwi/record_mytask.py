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

"""Tailored LeKiwi data-collection entrypoint.

Prerequisite: the host must be running on the LeKiwi (matching `REMOTE_IP` below):

    python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi

Then run this script on your laptop (with the leader arm connected):

    uv run python examples/lekiwi/record_mytask.py

Controls while recording (keyboard focus on this terminal / rerun window):
  - Leader arm drives the follower arm.
  - w/a/s/d move the base, z/x rotate, r/f change speed.
  - Right arrow: end the current episode early.
  - Left arrow: re-record the current episode.
  - Esc: stop recording and upload.
"""

from lerobot.datasets import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# ─── Session settings — edit these ──────────────────────────────────────────
HF_REPO_ID = "alexzai/lekiwi_v1"
TASK_DESCRIPTION = "My task description"  # <- describe the task you are collecting

REMOTE_IP = "192.168.50.187"  # LeKiwi host IP
ROBOT_ID = "my_awesome_kiwi"  # must match the id used to calibrate / run the host
LEADER_PORT = "/dev/tty.usbmodem5B8E1169971"  # <- verify with `lerobot-find-port`
LEADER_ID = "my_leader_arm"  # must match the id used to calibrate the leader

NUM_EPISODES = 20
FPS = 30
EPISODE_TIME_SEC = 30  # length of each recorded episode
RESET_TIME_SEC = 10  # time between episodes to reposition the LeKiwi to a new angle
PUSH_TO_HUB = True  # set False to keep the dataset local only
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # Camera shapes come from the shared `lekiwi_cameras_config()` in config_lekiwi.py, so the host
    # capture and the client/dataset declarations stay in sync from a single source on this branch.
    # Create the robot and teleoperator configurations
    robot_config = LeKiwiClientConfig(remote_ip=REMOTE_IP, id=ROBOT_ID)
    leader_arm_config = SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID)
    keyboard_config = KeyboardTeleopConfig()

    # Initialize the robot and teleoperator
    robot = LeKiwiClient(robot_config)
    leader_arm = SO101Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
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

    # Connect the robot and teleoperator
    # The host must already be running on LeKiwi:
    #   python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_record")

    try:
        if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )

        print("Starting record loop...")
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {recorded_episodes}")

            # Main record loop
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                teleop=[leader_arm, keyboard],
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

            # Reset the environment if not stopping or re-recording.
            # Use this window to drive/reposition the LeKiwi to the next angle.
            if not events["stop_recording"] and (
                (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=[leader_arm, keyboard],
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            recorded_episodes += 1
    finally:
        # Clean up
        log_say("Stop recording")
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()
        listener.stop()

        dataset.finalize()
        if PUSH_TO_HUB:
            dataset.push_to_hub()


if __name__ == "__main__":
    main()
