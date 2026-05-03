#!/usr/bin/env python

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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import MotorCalibration
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def _make_bus_mock() -> MagicMock:
    """Return a bus mock with just the attributes used by the robot."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        bus_mock.is_calibrated = True
        return bus_mock

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO101Follower, "configure", lambda self: None),
    ):
        cfg = SO101FollowerConfig(port="/dev/null")
        robot = SO101Follower(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def _stub_calibration() -> dict[str, MotorCalibration]:
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    return {
        name: MotorCalibration(id=i, drive_mode=0, homing_offset=100, range_min=10, range_max=4000)
        for i, name in enumerate(names, 1)
    }


def verify_calibration_values(
    calibration: dict[str, MotorCalibration], expected_values: dict[str, tuple[int, int, int]]
):
    for motor, (expected_range_min, expected_homing_offset, expected_range_max) in expected_values.items():
        assert calibration[motor].range_min == expected_range_min, f"{motor} range_min mismatch"
        assert calibration[motor].homing_offset == expected_homing_offset, f"{motor} homing_offset mismatch"
        assert calibration[motor].range_max == expected_range_max, f"{motor} range_max mismatch"


def test_partial_calibration_merges_only_targeted(follower, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    follower.calibration = _stub_calibration()
    follower.bus.set_half_turn_homings.return_value = {"shoulder_pan": 900}
    follower.bus.record_ranges_of_motion.return_value = ({"shoulder_pan": 50}, {"shoulder_pan": 3500})

    follower.calibrate(motors=["shoulder_pan"])

    # the shoulder_pan should be updated, while the other motors should remain unchanged
    verify_calibration_values(
        follower.calibration,
        {
            "shoulder_pan": (50, 900, 3500),
            "shoulder_lift": (10, 100, 4000),
            "elbow_flex": (10, 100, 4000),
            "wrist_flex": (10, 100, 4000),
            "wrist_roll": (10, 100, 4000),
            "gripper": (10, 100, 4000),
        },
    )


def test_partial_calibration_multiple_motors_merges_only_targeted(follower, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    follower.calibration = _stub_calibration()
    follower.bus.set_half_turn_homings.return_value = {"shoulder_pan": 900, "elbow_flex": 800}
    follower.bus.record_ranges_of_motion.return_value = (
        {"shoulder_pan": 50, "elbow_flex": 60},
        {"shoulder_pan": 3500, "elbow_flex": 3600},
    )

    follower.calibrate(motors=["shoulder_pan", "elbow_flex"])

    # only hsoulder_pan and elbow_flex should be updated
    verify_calibration_values(
        follower.calibration,
        {
            "shoulder_pan": (50, 900, 3500),
            "shoulder_lift": (10, 100, 4000),
            "elbow_flex": (60, 800, 3600),
            "wrist_flex": (10, 100, 4000),
            "wrist_roll": (10, 100, 4000),
            "gripper": (10, 100, 4000),
        },
    )


def test_partial_calibrate_full_turn_motor(follower, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    follower.calibration = _stub_calibration()
    follower.bus.set_half_turn_homings.return_value = {"wrist_roll": 777}

    follower.calibrate(motors=["wrist_roll"])

    follower.bus.record_ranges_of_motion.assert_not_called()

    # the wrist_roll homing offset should be updated, but the range should remain unchanged [0, 4095]
    verify_calibration_values(
        follower.calibration,
        {
            "shoulder_pan": (10, 100, 4000),
            "shoulder_lift": (10, 100, 4000),
            "elbow_flex": (10, 100, 4000),
            "wrist_flex": (10, 100, 4000),
            "wrist_roll": (0, 777, 4095),
            "gripper": (10, 100, 4000),
        },
    )


def test_partial_calibration_requires_existing_calibration(follower, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    follower.calibration = {}
    with pytest.raises(ValueError):
        follower.calibrate(motors=["shoulder_pan"])


def test_partial_calibration_unknown_motor(follower, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    follower.calibration = _stub_calibration()
    with pytest.raises(ValueError):
        follower.calibrate(motors=["joint_x"])
