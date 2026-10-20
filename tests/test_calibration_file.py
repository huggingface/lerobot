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

"""The calibration file a robot or a teleoperator offers to keep instead of a new calibration."""

from unittest.mock import MagicMock, call

import pytest

from lerobot.motors import MotorCalibration
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

CALIBRATION = {
    "motor_1": MotorCalibration(id=1, drive_mode=0, homing_offset=10, range_min=100, range_max=200)
}


@pytest.fixture(params=["robot", "teleoperator"])
def device(request, tmp_path):
    if request.param == "robot":
        return MockRobot(MockRobotConfig(id="arm", calibration_dir=tmp_path))
    return MockTeleop(MockTeleopConfig(id="arm", calibration_dir=tmp_path))


def test_without_a_file_nothing_is_asked(device, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("a calibration file was offered"))
    bus = MagicMock()

    assert not device._keep_calibration_file(bus)
    assert bus.mock_calls == []


def test_a_kept_file_is_written_with_the_torque_off(device, monkeypatch):
    device.calibration = CALIBRATION
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    bus = MagicMock()

    assert device._keep_calibration_file(bus)
    assert bus.mock_calls == [call.disable_torque(), call.write_calibration(CALIBRATION)]


def test_typing_c_asks_for_a_new_calibration(device, monkeypatch):
    device.calibration = CALIBRATION
    monkeypatch.setattr("builtins.input", lambda prompt: "c")
    bus = MagicMock()

    assert not device._keep_calibration_file(bus)
    assert bus.mock_calls == []


def test_a_file_is_not_written_when_the_torque_stays_on(device, monkeypatch):
    device.calibration = CALIBRATION
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    bus = MagicMock()
    bus.disable_torque.side_effect = ConnectionError("motor 1 does not answer")

    with pytest.raises(ConnectionError):
        device._keep_calibration_file(bus)

    bus.write_calibration.assert_not_called()
