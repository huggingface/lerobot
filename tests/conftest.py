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
import pytest

from lerobot.common.utils.utils import init_hydra_config

from .utils import DEVICE, KOCH_ROBOT_CONFIG_PATH


def pytest_collection_finish():
    print(f"\nTesting with {DEVICE=}")


@pytest.fixture(scope="session")
def is_koch_available():
    try:
        from lerobot.common.robot_devices.robots.factory import make_robot

        robot_cfg = init_hydra_config(KOCH_ROBOT_CONFIG_PATH)
        robot = make_robot(robot_cfg)
        robot.connect()
        del robot
        return True
    except Exception as e:
        print("A koch robot is not available.")
        print(e)
        return False
