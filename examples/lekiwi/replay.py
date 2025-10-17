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

import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

EPISODE_IDX = 0

# Initialize the robot config
robot_config = LeKiwiClientConfig(remote_ip="172.18.134.136", id="lekiwi")

# Initialize the robot
robot = LeKiwiClient(robot_config)

# Fetch the dataset to replay
dataset = LeRobotDataset("<hf_username>/<dataset_repo_id>", episodes=[EPISODE_IDX])
# Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == EPISODE_IDX)
actions = episode_frames.select_columns(ACTION)

# Connect to the robot
robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

print("Starting replay loop...")
log_say(f"Replaying episode {EPISODE_IDX}")
for idx in range(len(episode_frames)):
    t0 = time.perf_counter()

    # Get recorded action from dataset
    action = {
        name: float(actions[idx][ACTION][i]) for i, name in enumerate(dataset.features[ACTION]["names"])
    }

    # Send action to robot
    _ = robot.send_action(action)

    busy_wait(max(1.0 / dataset.fps - (time.perf_counter() - t0), 0.0))

robot.disconnect()
