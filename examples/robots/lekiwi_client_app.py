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

import logging
import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import hw_to_dataset_features
from lerobot.common.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.common.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.common.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig

NB_CYCLES_CLIENT_CONNECTION = 250


def main():
    logging.info("Configuring Teleop Devices")
    leader_arm_config = SO100LeaderConfig(port="/dev/tty.usbmodem58760433331")
    leader_arm = SO100Leader(leader_arm_config)

    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)

    logging.info("Configuring LeKiwi Client")
    robot_config = LeKiwiClientConfig(remote_ip="172.18.134.136", id="lekiwi")
    robot = LeKiwiClient(robot_config)

    logging.info("Creating LeRobot Dataset")

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id="user/lekiwi" + str(int(time.time())),
        fps=10,
        features=dataset_features,
        robot_type=robot.name,
    )

    logging.info("Connecting Teleop Devices")
    leader_arm.connect()
    keyboard.connect()

    logging.info("Connecting remote LeKiwi")
    robot.connect()

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        logging.error("Failed to connect to all devices")
        return

    logging.info("Starting LeKiwi teleoperation")
    i = 0
    while i < NB_CYCLES_CLIENT_CONNECTION:
        arm_action = leader_arm.get_action()
        base_action = keyboard.get_action()
        action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

        action_sent = robot.send_action(action)
        observation = robot.get_observation()

        frame = {**action_sent, **observation}
        task = "Dummy Example Task Dataset"

        logging.info("Saved a frame into the dataset")
        dataset.add_frame(frame, task)
        i += 1

    logging.info("Disconnecting Teleop Devices and LeKiwi Client")
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()

    logging.info("Uploading dataset to the hub")
    dataset.save_episode()
    dataset.push_to_hub()

    logging.info("Finished LeKiwi cleanly")


if __name__ == "__main__":
    main()
