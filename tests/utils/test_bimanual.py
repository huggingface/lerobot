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

from lerobot.utils.bimanual import BimanualMixin


def _make_arm_mock(name: str, fails: bool = False) -> MagicMock:
    """Return an arm mock that tracks its own connection state."""
    arm = MagicMock(name=name)
    arm.is_connected = False

    def _connect(_calibrate=True):
        if fails:
            raise ConnectionError(f"Failed to open a camera on {name}.")
        arm.is_connected = True

    def _disconnect():
        arm.is_connected = False

    arm.connect.side_effect = _connect
    arm.disconnect.side_effect = _disconnect
    return arm


class _Bimanual(BimanualMixin):
    """Minimal concrete stand-in: the mixin only needs the two arms."""

    def __init__(self, left_arm, right_arm):
        self.left_arm = left_arm
        self.right_arm = right_arm


def test_connect_releases_left_arm_when_right_arm_fails():
    # The left arm holds a bus and cameras; leaving it connected strands a whole arm.
    left = _make_arm_mock("left_arm")
    right = _make_arm_mock("right_arm", fails=True)
    robot = _Bimanual(left, right)

    with pytest.raises(ConnectionError, match="right_arm"):
        robot.connect()

    left.disconnect.assert_called_once_with()
    assert not left.is_connected
    assert not robot.is_connected


def test_connect_reports_original_error_when_left_arm_cleanup_fails():
    left = _make_arm_mock("left_arm")
    left.disconnect.side_effect = RuntimeError("disconnect exploded")
    right = _make_arm_mock("right_arm", fails=True)
    robot = _Bimanual(left, right)

    with pytest.raises(ConnectionError, match="right_arm"):
        robot.connect()


def test_connect_leaves_both_arms_connected_on_success():
    left = _make_arm_mock("left_arm")
    right = _make_arm_mock("right_arm")
    robot = _Bimanual(left, right)

    robot.connect()

    assert robot.is_connected
    left.disconnect.assert_not_called()
    right.disconnect.assert_not_called()
