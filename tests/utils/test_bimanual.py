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

import pytest

from lerobot.utils.bimanual import BimanualMixin


class FakeArm:
    def __init__(
        self,
        *,
        connected: bool = False,
        calibrated: bool = False,
        connect_error: Exception | None = None,
        disconnect_error: Exception | None = None,
        calibrate_error: Exception | None = None,
        configure_error: Exception | None = None,
        acquire_before_connect_error: bool = False,
    ):
        self._is_connected = connected
        self._is_calibrated = calibrated
        self.connect_error = connect_error
        self.disconnect_error = disconnect_error
        self.calibrate_error = calibrate_error
        self.configure_error = configure_error
        self.acquire_before_connect_error = acquire_before_connect_error
        self.calls: list[tuple[str, bool] | tuple[str]] = []

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.calls.append(("connect", calibrate))
        if self.acquire_before_connect_error:
            self._is_connected = True
        if self.connect_error is not None:
            raise self.connect_error
        self._is_connected = True

    def disconnect(self) -> None:
        self.calls.append(("disconnect",))
        if self.disconnect_error is not None:
            raise self.disconnect_error
        self._is_connected = False

    def calibrate(self) -> None:
        self.calls.append(("calibrate",))
        if self.calibrate_error is not None:
            raise self.calibrate_error
        self._is_calibrated = True

    def configure(self) -> None:
        self.calls.append(("configure",))
        if self.configure_error is not None:
            raise self.configure_error


class FakeBimanual(BimanualMixin):
    def __init__(self, left_arm: FakeArm, right_arm: FakeArm):
        self.left_arm = left_arm
        self.right_arm = right_arm


def test_connect_is_noop_when_fully_connected():
    left = FakeArm(connected=True)
    right = FakeArm(connected=True)
    device = FakeBimanual(left, right)

    device.connect()

    assert left.calls == []
    assert right.calls == []


@pytest.mark.parametrize("connected_side", ["left", "right"])
def test_connect_repairs_a_partial_connection(connected_side):
    left = FakeArm(connected=connected_side == "left")
    right = FakeArm(connected=connected_side == "right")
    device = FakeBimanual(left, right)

    device.connect(calibrate=False)

    assert device.is_connected
    assert left.calls == ([] if connected_side == "left" else [("connect", False)])
    assert right.calls == ([] if connected_side == "right" else [("connect", False)])


def test_connect_failure_preserves_an_arm_connected_before_the_call():
    left = FakeArm(connected=True)
    connect_error = RuntimeError("right failed")
    right = FakeArm(connect_error=connect_error)
    device = FakeBimanual(left, right)

    with pytest.raises(RuntimeError, match="right failed") as exc_info:
        device.connect()

    assert left.is_connected
    assert left.calls == []
    assert any("right arm" in note for note in exc_info.value.__notes__)


def test_connect_failure_rolls_back_arms_started_by_the_call():
    left = FakeArm()
    connect_error = RuntimeError("right failed")
    right = FakeArm(connect_error=connect_error, acquire_before_connect_error=True)
    device = FakeBimanual(left, right)

    with pytest.raises(RuntimeError, match="right failed"):
        device.connect()

    assert not left.is_connected
    assert not right.is_connected
    assert left.calls == [("connect", True), ("disconnect",)]
    assert right.calls == [("connect", True), ("disconnect",)]


def test_connect_reports_primary_and_rollback_failures():
    rollback_error = OSError("left cleanup failed")
    left = FakeArm(disconnect_error=rollback_error)
    connect_error = RuntimeError("right failed")
    right = FakeArm(connect_error=connect_error)
    device = FakeBimanual(left, right)

    with pytest.raises(ExceptionGroup) as exc_info:
        device.connect()

    assert exc_info.value.exceptions == (connect_error, rollback_error)


def test_calibrate_explicitly_calibrates_both_arms():
    left = FakeArm(calibrated=True)
    right = FakeArm(calibrated=False)
    device = FakeBimanual(left, right)

    device.calibrate()

    assert left.calls == [("calibrate",)]
    assert right.calls == [("calibrate",)]


def test_calibrate_attempts_both_arms_and_preserves_a_single_error():
    calibrate_error = RuntimeError("left failed")
    left = FakeArm(calibrate_error=calibrate_error)
    right = FakeArm()
    device = FakeBimanual(left, right)

    with pytest.raises(RuntimeError, match="left failed") as exc_info:
        device.calibrate()

    assert exc_info.value is calibrate_error
    assert right.calls == [("calibrate",)]
    assert any("calibrating the left arm" in note for note in exc_info.value.__notes__)


def test_configure_groups_failures_from_both_arms():
    left_error = OSError("left failed")
    right_error = RuntimeError("right failed")
    left = FakeArm(configure_error=left_error)
    right = FakeArm(configure_error=right_error)
    device = FakeBimanual(left, right)

    with pytest.raises(ExceptionGroup) as exc_info:
        device.configure()

    assert left.calls == [("configure",)]
    assert right.calls == [("configure",)]
    assert exc_info.value.exceptions == (left_error, right_error)


def test_disconnect_is_noop_when_fully_disconnected():
    left = FakeArm()
    right = FakeArm()
    device = FakeBimanual(left, right)

    device.disconnect()

    assert left.calls == [("disconnect",)]
    assert right.calls == [("disconnect",)]


def test_disconnect_cleans_up_a_partial_connection():
    left = FakeArm()
    right = FakeArm(connected=True)
    device = FakeBimanual(left, right)

    device.disconnect()

    assert left.calls == [("disconnect",)]
    assert right.calls == [("disconnect",)]
    assert not device.is_connected


def test_disconnect_attempts_both_arms_and_preserves_a_single_error():
    disconnect_error = OSError("left failed")
    left = FakeArm(connected=True, disconnect_error=disconnect_error)
    right = FakeArm(connected=True)
    device = FakeBimanual(left, right)

    with pytest.raises(OSError, match="left failed") as exc_info:
        device.disconnect()

    assert exc_info.value is disconnect_error
    assert right.calls == [("disconnect",)]
    assert any("left arm" in note for note in exc_info.value.__notes__)


def test_disconnect_groups_failures_from_both_arms():
    left_error = OSError("left failed")
    right_error = RuntimeError("right failed")
    left = FakeArm(connected=True, disconnect_error=left_error)
    right = FakeArm(connected=True, disconnect_error=right_error)
    device = FakeBimanual(left, right)

    with pytest.raises(ExceptionGroup) as exc_info:
        device.disconnect()

    assert exc_info.value.exceptions == (left_error, right_error)
