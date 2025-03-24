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

import rerun as rr

from lerobot.common.robots.dummy.configuration_dummy import DummyConfig
from lerobot.common.robots.dummy.dummy import Dummy
from lerobot.common.teleoperators.so100 import SO100Teleop, SO100TeleopConfig


def main():
    logging.info("Configuring Devices")
    leader_arm_config = SO100TeleopConfig(port="/dev/tty.usbmodem58760434171")
    leader_arm = SO100Teleop(leader_arm_config)

    robot_config = DummyConfig()
    robot = Dummy(robot_config)

    logging.info("Connecting SO100 Devices")
    leader_arm.connect()

    logging.info("Connecting Dummy")
    robot.connect()

    rr.init("rerun_dummy_data")
    rr.spawn(memory_limit="50%")  # add a bluprint config
    logging.info("Starting...")
    i = 0
    while i < 10000:
        arm_action = leader_arm.get_action()
        observation = robot.get_observation()

        for j in range(arm_action.size):
            rr.log(f"arm_action_{j}", rr.Scalar(arm_action[j]))

        for k, v in observation.items():
            rr.log(k, rr.Image(v))

        i += 1

    robot.disconnect()
    leader_arm.disconnect()


if __name__ == "__main__":
    main()
