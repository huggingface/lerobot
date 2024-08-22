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
import traceback

import pytest

from lerobot.common.utils.utils import init_hydra_config

from .utils import DEVICE, ROBOT_CONFIG_PATH_TEMPLATE


def pytest_collection_finish():
    print(f"\nTesting with {DEVICE=}")


@pytest.fixture
def is_robot_available(robot_type):
    try:
        from lerobot.common.robot_devices.robots.factory import make_robot

        config_path = ROBOT_CONFIG_PATH_TEMPLATE.format(robot=robot_type)
        robot_cfg = init_hydra_config(config_path)
        robot = make_robot(robot_cfg)
        robot.connect()
        del robot
        return True
    except Exception:
        traceback.print_exc()
        print(f"\nA {robot_type} robot is not available.")
        return False
