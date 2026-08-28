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


def _errors(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]


def test_left_connect_failure_releases_both_arms():
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


def test_right_connect_failure_releases_the_right_arm_first():
    left, right = _arm(), _arm()
    failure = RuntimeError("right failed")
    right.connect.side_effect = failure
    # The right arm may hold a half-initialized bus from the failed attempt.
    released: list[str] = []
    left.disconnect.side_effect = lambda: released.append("left")
    right.disconnect.side_effect = lambda: released.append("right")
    robot = _Bimanual(left, right)

    with pytest.raises(RuntimeError) as exc_info:
        robot.connect()

    assert exc_info.value is failure
    assert released == ["right", "left"]


def test_connect_reports_the_original_error_when_rollback_fails(caplog):
    left, right = _arm(), _arm()
    failure = RuntimeError("right failed")
    right.connect.side_effect = failure
    right.disconnect.side_effect = RuntimeError("right cleanup failed")
    left.disconnect.side_effect = RuntimeError("left cleanup failed")
    robot = _Bimanual(left, right)

    with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError) as exc_info:
        robot.connect()

    assert exc_info.value is failure
    assert _errors(caplog) == [
        "Failed to disconnect the right arm during cleanup.",
        "Failed to disconnect the left arm during cleanup.",
    ]


def test_rollback_stays_quiet_for_arms_that_never_connected(caplog):
    left, right = _arm(), _arm()
    left.connect.side_effect = RuntimeError("left failed")
    # Arms guarded by `check_if_not_connected` raise when they never came up.
    left.disconnect.side_effect = DeviceNotConnectedError()
    right.disconnect.side_effect = DeviceNotConnectedError()
    robot = _Bimanual(left, right)

    with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError):
        robot.connect()

    assert _errors(caplog) == []


def test_disconnect_attempts_both_arms_when_one_fails(caplog):
    left, right = _arm(), _arm()
    left.disconnect.side_effect = RuntimeError("left cleanup failed")
    robot = _Bimanual(left, right)

    with caplog.at_level(logging.DEBUG):
        robot.disconnect()

    right.disconnect.assert_called_once()
    assert _errors(caplog) == ["Failed to disconnect the left arm during cleanup."]


def test_repeated_disconnect_is_quiet_once_the_arms_are_released(caplog):
    left, right = _arm(), _arm()
    left.disconnect.side_effect = DeviceNotConnectedError()
    right.disconnect.side_effect = DeviceNotConnectedError()
    robot = _Bimanual(left, right)

    with caplog.at_level(logging.DEBUG):
        robot.disconnect()
        robot.disconnect()

    assert left.disconnect.call_count == 2
    assert right.disconnect.call_count == 2
    assert _errors(caplog) == []


def test_disconnect_uses_the_arms_configured_behavior():
    # The cleanup hook may force torque off, which must not leak into the
    # normal disconnect path.
    left, right = _arm(), _arm()
    robot = _Bimanual(left, right)
    robot._disconnect_arm_for_cleanup = MagicMock()

    robot.disconnect()

    robot._disconnect_arm_for_cleanup.assert_not_called()
    left.disconnect.assert_called_once()
    right.disconnect.assert_called_once()


def test_rollback_uses_the_cleanup_hook():
    left, right = _arm(), _arm()
    right.connect.side_effect = RuntimeError("right failed")
    robot = _Bimanual(left, right)
    robot._disconnect_arm_for_cleanup = MagicMock()

    with pytest.raises(RuntimeError):
        robot.connect()

    assert robot._disconnect_arm_for_cleanup.call_count == 2
    left.disconnect.assert_not_called()
    right.disconnect.assert_not_called()
