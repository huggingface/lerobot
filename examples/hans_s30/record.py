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

"""Record a LeRobotDataset with the Hans Robot S30 via zero-force teaching.

The operator manually guides the arm while joint positions and camera frames
are recorded.  After ``NUM_EPISODES`` episodes the dataset is saved and
optionally uploaded to the Hugging Face Hub.

Edit the constants below to match your setup before running::

    python examples/hans_s30/record.py

Key settings to edit:
- ``ROBOT_IP``        – IPv4 address of the Hans controller.
- ``HF_REPO_ID``      – Hugging Face dataset repository (``<user>/<dataset>``).
- ``TASK_DESCRIPTION``– Natural-language description of the task.
- ``NUM_EPISODES``    – Number of demos to collect.
- ``EPISODE_TIME_SEC``– Duration of each demonstration in seconds.
- ``RESET_TIME_SEC``  – Duration for the manual environment reset between demos.
"""

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import init_keyboard_listener
from lerobot.datasets import LeRobotDataset, create_initial_features
from lerobot.robots.hans_s30 import HansS30, HansS30RobotConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# ── User settings ────────────────────────────────────────────────────────────
ROBOT_IP = "192.168.115.11"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"
TASK_DESCRIPTION = "Pick up the red block and place it in the bin"

NUM_EPISODES = 10
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 20
# ─────────────────────────────────────────────────────────────────────────────


def main():
    camera_config = {
        "wrist_cam": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
        "base_cam": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
    }

    robot_config = HansS30RobotConfig(
        ip=ROBOT_IP,
        port=10003,
        id="my_hans_s30",
        cameras=camera_config,
    )

    robot = HansS30(robot_config)

    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=create_initial_features(observation=robot.observation_features),
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    robot.connect()
    listener, events = init_keyboard_listener()
    init_rerun(session_name="hans_s30_record")

    try:
        if not robot.is_connected:
            raise RuntimeError("Robot is not connected!")

        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {recorded_episodes + 1} of {NUM_EPISODES}")
            log_say("Guide the arm using zero-force teaching …")

            robot.enable_free_driver()

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

            robot.disable_free_driver()

            if not events["stop_recording"] and (
                recorded_episodes < NUM_EPISODES - 1 or events["rerecord_episode"]
            ):
                log_say("Reset the environment, then press Enter to continue")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
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
            recorded_episodes += 1

    finally:
        log_say("Stopping recording")
        robot.disconnect()
        listener.stop()

        dataset.finalize()
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
