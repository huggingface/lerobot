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


def test_lekiwi_teleop_base_only_only_maps_base_motors(tmp_path):
    config = LeKiwiConfig(port="/dev/null", id="kiwi", calibration_dir=tmp_path, cameras={})
    assert config.teleop_base_only is False

    robot = LeKiwi(config)
    assert set(robot.arm_motors) == {
        "arm_shoulder_pan",
        "arm_shoulder_lift",
        "arm_elbow_flex",
        "arm_wrist_flex",
        "arm_wrist_roll",
        "arm_gripper",
    }
    assert {motor.id for motor in robot.bus.motors.values()} == {1, 2, 3, 4, 5, 6, 7, 8, 9}

    base_only_config = LeKiwiConfig(
        port="/dev/null", id="kiwi", calibration_dir=tmp_path, cameras={}, teleop_base_only=True
    )
    base_only_robot = LeKiwi(base_only_config)
    assert base_only_robot.arm_motors == []
    assert sorted(base_only_robot.base_motors) == ["base_back_wheel", "base_left_wheel", "base_right_wheel"]
    assert {motor.id for motor in base_only_robot.bus.motors.values()} == {7, 8, 9}

    expected_state_ft = {"x.vel": float, "y.vel": float, "theta.vel": float}
    assert base_only_robot.observation_features == expected_state_ft
    assert base_only_robot.action_features == expected_state_ft
