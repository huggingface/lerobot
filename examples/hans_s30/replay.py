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

"""Replay a recorded episode on the Hans Robot S30.

Loads a LeRobotDataset and plays back a single episode by replaying the
stored joint-position actions.

Edit the constants below to match your setup before running::

    python examples/hans_s30/replay.py

Key settings to edit:
- ``ROBOT_IP``     – IPv4 address of the Hans controller.
- ``HF_REPO_ID``   – Dataset repository used during recording.
- ``EPISODE_INDEX``– Index of the episode to replay (0-based).
"""

import time

from lerobot.datasets import LeRobotDataset
from lerobot.robots.hans_s30 import HansS30, HansS30RobotConfig
from lerobot.utils.robot_utils import precise_sleep

# ── User settings ────────────────────────────────────────────────────────────
ROBOT_IP = "192.168.115.11"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"
EPISODE_INDEX = 0
# ─────────────────────────────────────────────────────────────────────────────


def main():
    robot_config = HansS30RobotConfig(
        ip=ROBOT_IP,
        port=10003,
        id="my_hans_s30",
    )
    robot = HansS30(robot_config)
    robot.connect()

    dataset = LeRobotDataset(HF_REPO_ID, episodes=[EPISODE_INDEX])

    from_idx = dataset.meta.episodes["dataset_from_index"][EPISODE_INDEX]
    to_idx = dataset.meta.episodes["dataset_to_index"][EPISODE_INDEX]
    fps = dataset.fps

    print(f"Replaying episode {EPISODE_INDEX} ({to_idx - from_idx} frames @ {fps} Hz) …")

    try:
        for idx in range(from_idx, to_idx):
            t0 = time.perf_counter()
            frame = dataset[idx]

            action = {key: frame[key].item() for key in robot.action_features}
            robot.send_action(action)

            precise_sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

        print("Replay finished.")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
