#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lerobot.robots.hope_jr.hope_jr_arm import HopeJrArm
from lerobot.robots.hope_jr.hope_jr_hand import HopeJrHand
from lerobot.robots.koch_follower.koch_follower import KochFollower
from lerobot.robots.lekiwi.lekiwi import LeKiwi
from lerobot.robots.omx_follower.omx_follower import OmxFollower
from lerobot.robots.openarm_follower.openarm_follower import OpenArmFollower
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.teleoperators.koch_leader.koch_leader import KochLeader
from lerobot.teleoperators.omx_leader.omx_leader import OmxLeader
from lerobot.teleoperators.openarm_leader.openarm_leader import OpenArmLeader
from lerobot.teleoperators.openarm_mini.openarm_mini import OpenArmMini
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.utils.errors import DeviceNotConnectedError


class FakeBus:
    def __init__(self, *, connected: bool = False, disconnect_error: Exception | None = None):
        self.is_connected = connected
        self.is_calibrated = True
        self.disconnect_error = disconnect_error
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.enable_torque_calls = 0
        self.set_zero_position_calls = 0

    def connect(self, *args, **kwargs) -> None:
        self.connect_calls += 1
        self.is_connected = True

    def disconnect(self, *args, **kwargs) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("bus is not connected")
        self.disconnect_calls += 1
        if self.disconnect_error is not None:
            raise self.disconnect_error
        self.is_connected = False

    def enable_torque(self) -> None:
        self.enable_torque_calls += 1

    def set_zero_position(self) -> None:
        self.set_zero_position_calls += 1


class FakeCamera:
    def __init__(self, *, connected: bool = False):
        self.is_connected = connected
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("camera is not connected")
        self.disconnect_calls += 1
        self.is_connected = False


def make_follower(cls, *, bus_connected: bool = False, camera_connected: bool = False):
    robot = object.__new__(cls)
    robot.id = "test"
    robot.config = SimpleNamespace(port="/dev/null", disable_torque_on_disconnect=True)
    robot.bus = FakeBus(connected=bus_connected)
    robot.cameras = {"camera": FakeCamera(connected=camera_connected)}
    robot.configure = MagicMock()
    if isinstance(robot, LeKiwi):
        robot.stop_base = MagicMock()
    return robot


FOLLOWER_CLASSES = [
    SOFollower,
    OpenArmFollower,
    OmxFollower,
    KochFollower,
    HopeJrArm,
    HopeJrHand,
    LeKiwi,
]


@pytest.mark.parametrize("cls", FOLLOWER_CLASSES)
def test_composite_follower_connect_and_disconnect_are_idempotent(cls):
    robot = make_follower(cls)

    robot.connect(calibrate=False)
    robot.connect(calibrate=False)

    assert robot.bus.connect_calls == 1
    assert robot.cameras["camera"].connect_calls == 1
    robot.configure.assert_called_once_with()

    robot.disconnect()
    robot.disconnect()

    assert robot.bus.disconnect_calls == 1
    assert robot.cameras["camera"].disconnect_calls == 1


@pytest.mark.parametrize("cls", FOLLOWER_CLASSES)
def test_composite_follower_repairs_a_partial_connection(cls):
    robot = make_follower(cls, bus_connected=True)

    robot.connect(calibrate=False)

    assert robot.bus.connect_calls == 0
    assert robot.cameras["camera"].connect_calls == 1
    assert robot.is_connected


@pytest.mark.parametrize("cls", FOLLOWER_CLASSES)
def test_composite_follower_rolls_back_resources_started_by_connect(cls):
    robot = make_follower(cls)
    configure_error = RuntimeError("configuration failed")
    robot.configure.side_effect = configure_error

    with pytest.raises(RuntimeError, match="configuration failed") as exc_info:
        robot.connect(calibrate=False)

    assert exc_info.value is configure_error
    assert not robot.bus.is_connected
    assert not robot.cameras["camera"].is_connected


@pytest.mark.parametrize("cls", FOLLOWER_CLASSES)
def test_composite_follower_disconnect_attempts_cameras_after_bus_failure(cls):
    robot = make_follower(cls, bus_connected=True, camera_connected=True)
    disconnect_error = OSError("bus failed")
    robot.bus.disconnect_error = disconnect_error

    with pytest.raises(OSError, match="bus failed") as exc_info:
        robot.disconnect()

    assert exc_info.value is disconnect_error
    assert robot.cameras["camera"].disconnect_calls == 1


@pytest.mark.parametrize("cls", FOLLOWER_CLASSES)
def test_composite_follower_disconnect_attempts_resources_in_unknown_partial_state(cls):
    robot = make_follower(cls)
    robot.bus.disconnect = MagicMock(side_effect=DeviceNotConnectedError("not connected"))
    robot.cameras["camera"].disconnect = MagicMock(side_effect=DeviceNotConnectedError("not connected"))

    robot.disconnect()

    robot.bus.disconnect.assert_called_once()
    robot.cameras["camera"].disconnect.assert_called_once_with()


def make_teleoperator(cls):
    teleop = object.__new__(cls)
    teleop.id = "test"
    teleop.config = SimpleNamespace(port="/dev/null", manual_control=True)
    teleop.bus = FakeBus()
    teleop.configure = MagicMock()
    return teleop


TELEOPERATOR_CLASSES = [SOLeader, OpenArmLeader, OpenArmMini, OmxLeader, KochLeader]


@pytest.mark.parametrize("cls", TELEOPERATOR_CLASSES)
def test_bimanual_arm_teleoperator_connect_and_disconnect_are_idempotent(cls):
    teleop = make_teleoperator(cls)

    teleop.connect(calibrate=False)
    teleop.connect(calibrate=False)

    assert teleop.bus.connect_calls == 1
    teleop.configure.assert_called_once_with()

    teleop.disconnect()
    teleop.disconnect()

    assert teleop.bus.disconnect_calls == 1


@pytest.mark.parametrize("cls", TELEOPERATOR_CLASSES)
def test_bimanual_arm_teleoperator_rolls_back_a_failed_connect(cls):
    teleop = make_teleoperator(cls)
    configure_error = RuntimeError("configuration failed")
    teleop.configure.side_effect = configure_error

    with pytest.raises(RuntimeError, match="configuration failed") as exc_info:
        teleop.connect(calibrate=False)

    assert exc_info.value is configure_error
    assert not teleop.bus.is_connected


@pytest.mark.parametrize("cls", TELEOPERATOR_CLASSES)
def test_bimanual_arm_teleoperator_disconnect_attempts_an_unknown_partial_state(cls):
    teleop = make_teleoperator(cls)
    teleop.bus.disconnect = MagicMock(side_effect=DeviceNotConnectedError("not connected"))

    teleop.disconnect()

    teleop.bus.disconnect.assert_called_once()
