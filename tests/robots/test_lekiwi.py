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

from unittest.mock import MagicMock

import pytest

pytest.importorskip("rustypot", reason="rustypot is required (install lerobot[serial-motors])")

from lerobot.motors import MotorCalibration
from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig


def test_lekiwi_calibrates_its_arm_by_hand_and_leaves_its_wheels_at_the_whole_encoder(tmp_path, monkeypatch):
    robot = LeKiwi(LeKiwiConfig(port="/dev/null", id="kiwi", calibration_dir=tmp_path, cameras={}))
    arm = {
        motor: MotorCalibration(
            id=robot.bus.motors[motor].id, drive_mode=0, homing_offset=10, range_min=100, range_max=200
        )
        for motor in robot.arm_motors
    }
    monkeypatch.setattr(robot.bus, "record_calibration", MagicMock(return_value=arm))
    monkeypatch.setattr(robot.bus, "write_calibration", MagicMock())

    robot.calibrate()

    robot.bus.record_calibration.assert_called_once_with(
        "position", full_turn_motors=["arm_wrist_roll"], motors=robot.arm_motors
    )
    assert robot.calibration == arm | {
        motor: MotorCalibration(
            id=robot.bus.motors[motor].id, drive_mode=0, homing_offset=0, range_min=0, range_max=4095
        )
        for motor in robot.base_motors
    }
    robot.bus.write_calibration.assert_called_once_with(robot.calibration)
