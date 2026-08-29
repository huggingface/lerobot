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

import logging
from unittest.mock import MagicMock

import pytest

from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.errors import DeviceNotConnectedError


class _Bimanual(BimanualMixin):
    def __init__(self, left_arm, right_arm):
        self.left_arm = left_arm
        self.right_arm = right_arm


def _arm() -> MagicMock:
    arm = MagicMock()
    arm.is_connected = False
    arm.is_calibrated = False
    return arm


def test_connect_rolls_back_when_left_arm_fails():
    left, right = _arm(), _arm()
    failure = RuntimeError("left failed")
    left.connect.side_effect = failure
    robot = _Bimanual(left, right)

    with pytest.raises(RuntimeError) as exc_info:
        robot.connect()

    assert exc_info.value is failure
    right.connect.assert_not_called()
    left.disconnect.assert_called_once()
    right.disconnect.assert_called_once()


def test_connect_rolls_back_when_right_arm_fails():
    left, right = _arm(), _arm()
    failure = RuntimeError("right failed")
    right.connect.side_effect = failure
    disconnected = []
    left.disconnect.side_effect = lambda: disconnected.append("left")
    right.disconnect.side_effect = lambda: disconnected.append("right")
    robot = _Bimanual(left, right)

    with pytest.raises(RuntimeError) as exc_info:
        robot.connect()

    assert exc_info.value is failure
    assert disconnected == ["right", "left"]


def test_connect_keeps_original_error_if_rollback_fails(caplog):
    left, right = _arm(), _arm()
    failure = RuntimeError("right failed")
    right.connect.side_effect = failure
    left.disconnect.side_effect = RuntimeError("left cleanup failed")
    right.disconnect.side_effect = RuntimeError("right cleanup failed")
    robot = _Bimanual(left, right)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as exc_info:
        robot.connect()

    assert exc_info.value is failure
    assert [record.getMessage() for record in caplog.records] == [
        "Failed to disconnect the right arm.",
        "Failed to disconnect the left arm.",
    ]


def test_disconnect_continues_when_one_arm_fails(caplog):
    left, right = _arm(), _arm()
    left.is_connected = True
    right.is_connected = True
    left.disconnect.side_effect = RuntimeError("left cleanup failed")
    robot = _Bimanual(left, right)

    with caplog.at_level(logging.ERROR):
        robot.disconnect()

    right.disconnect.assert_called_once()
    assert [record.getMessage() for record in caplog.records] == [
        "Failed to disconnect the left arm.",
    ]


def test_disconnect_handles_partial_and_repeated_cleanup():
    left, right = _arm(), _arm()
    left.is_connected = True
    right.disconnect.side_effect = DeviceNotConnectedError()
    robot = _Bimanual(left, right)

    robot.disconnect()
    robot.disconnect()

    assert left.disconnect.call_count == 2
    assert right.disconnect.call_count == 2


def test_failed_connect_uses_cleanup_hook_only_for_rollback():
    left, right = _arm(), _arm()
    right.connect.side_effect = RuntimeError("right failed")
    robot = _Bimanual(left, right)
    robot._disconnect_arm_after_failed_connect = MagicMock()

    with pytest.raises(RuntimeError):
        robot.connect()

    assert robot._disconnect_arm_after_failed_connect.call_count == 2

    robot._disconnect_arm_after_failed_connect.reset_mock()
    robot.disconnect()

    robot._disconnect_arm_after_failed_connect.assert_not_called()
    left.disconnect.assert_called_once()
    right.disconnect.assert_called_once()
