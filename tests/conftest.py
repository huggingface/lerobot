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
from serial import SerialException

from lerobot import available_cameras, available_motors, available_robots
from lerobot.common.robot_devices.robots.utils import make_robot
from tests.utils import DEVICE, make_camera, make_motors_bus

# Import fixture modules as plugins
pytest_plugins = [
    "tests.fixtures.dataset_factories",
    "tests.fixtures.files",
    "tests.fixtures.hub",
    "tests.fixtures.optimizers",
]


def pytest_collection_finish():
    print(f"\nTesting with {DEVICE=}")


def _check_component_availability(component_type, available_components, make_component):
    """Generic helper to check if a hardware component is available"""
    if component_type not in available_components:
        raise ValueError(
            f"The {component_type} type is not valid. Expected one of these '{available_components}'"
        )

    try:
        component = make_component(component_type)
        component.connect()
        del component
        return True

    except Exception as e:
        print(f"\nA {component_type} is not available.")

        if isinstance(e, ModuleNotFoundError):
            print(f"\nInstall module '{e.name}'")
        elif isinstance(e, SerialException):
            print("\nNo physical device detected.")
        elif isinstance(e, ValueError) and "camera_index" in str(e):
            print("\nNo physical camera detected.")
        else:
            traceback.print_exc()

        return False


@pytest.fixture
def is_robot_available(robot_type):
    return _check_component_availability(robot_type, available_robots, make_robot)


@pytest.fixture
def is_camera_available(camera_type):
    return _check_component_availability(camera_type, available_cameras, make_camera)


@pytest.fixture
def is_motor_available(motor_type):
    return _check_component_availability(motor_type, available_motors, make_motors_bus)


@pytest.fixture
def patch_builtins_input(monkeypatch):
    def print_text(text=None):
        if text is not None:
            print(text)

    monkeypatch.setattr("builtins.input", print_text)
