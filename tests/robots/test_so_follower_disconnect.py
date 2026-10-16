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
from unittest.mock import patch

import pytest

from lerobot.motors import Motor, MotorNormMode
from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.robots.so_follower import SOFollower
from lerobot.utils.errors import DeviceNotConnectedError
from tests.mocks.mock_motors_bus import MockMotorsBus


class FakeComponent:
    def __init__(self, name: str, calls: list[str], failure: BaseException | None = None):
        self.name = name
        self.calls = calls
        self.failure = failure
        self.is_connected = True
        self.disconnect_args: tuple[object, ...] | None = None

    def disconnect(self, *args: object) -> None:
        self.calls.append(self.name)
        self.disconnect_args = args
        self.is_connected = False
        if self.failure is not None:
            raise self.failure


def make_follower(calls: list[str], failures: dict[str, BaseException], prefix: str = "") -> SOFollower:
    # Skip constructors to guarantee no hardware I/O or calibration-file writes.
    # Keep the actual disconnect method, decorators and is_connected property.
    robot = SOFollower.__new__(SOFollower)
    robot.id = "test"
    robot.config = SimpleNamespace(disable_torque_on_disconnect=True)
    robot.bus = FakeComponent(prefix + "bus", calls, failures.get(prefix + "bus"))
    robot.cameras = {
        name: FakeComponent(prefix + name, calls, failures.get(prefix + name))
        for name in ("camera1", "camera2")
    }
    return robot


def make_bimanual(calls: list[str], failures: dict[str, BaseException]) -> BiSOFollower:
    robot = BiSOFollower.__new__(BiSOFollower)
    robot.id = "test-bimanual"
    robot.left_arm = make_follower(calls, failures, "left.")
    robot.right_arm = make_follower(calls, failures, "right.")
    return robot


@pytest.mark.parametrize("disable_torque", [True, False])
@pytest.mark.parametrize("failed", [(), ("bus",), ("camera1",), ("camera2",), ("bus", "camera1", "camera2")])
def test_follower_attempts_all_cleanup_and_preserves_first_error(failed, disable_torque, caplog):
    calls = []
    failures = {name: (ConnectionError if name == "bus" else OSError)(f"{name} failed") for name in failed}
    robot = make_follower(calls, failures)
    robot.config.disable_torque_on_disconnect = disable_torque
    assert robot.is_connected

    if failures:
        first_error = failures[failed[0]]
        original_cause = OSError("original cause")
        first_error.__cause__ = original_cause
        with pytest.raises(type(first_error)) as exc_info:
            robot.disconnect()
        assert exc_info.value is first_error
        assert exc_info.value.__cause__ is original_cause
        for failure in failures.values():
            assert str(failure) in caplog.text
    else:
        robot.disconnect()

    assert calls == ["bus", "camera1", "camera2"]
    assert robot.bus.disconnect_args == (disable_torque,)
    assert all(not cam.is_connected for cam in robot.cameras.values())
    assert not robot.bus.is_connected


@pytest.mark.parametrize(
    "failed",
    [(), ("left.bus",), ("left.camera1",), ("right.bus",), ("left.bus", "right.bus", "right.camera1")],
)
def test_bimanual_attempts_both_arms_and_preserves_first_error(failed, caplog):
    calls = []
    failures = {
        name: (ConnectionError if name.endswith("bus") else OSError)(f"{name} failed") for name in failed
    }
    robot = make_bimanual(calls, failures)
    assert robot.is_connected

    if failures:
        with pytest.raises(type(failures[failed[0]])) as exc_info:
            robot.disconnect()
        assert exc_info.value is failures[failed[0]]
        for failure in failures.values():
            assert str(failure) in caplog.text
    else:
        robot.disconnect()

    assert calls == [
        "left.bus",
        "left.camera1",
        "left.camera2",
        "right.bus",
        "right.camera1",
        "right.camera2",
    ]
    for arm in (robot.left_arm, robot.right_arm):
        assert not arm.bus.is_connected
        assert all(not cam.is_connected for cam in arm.cameras.values())


@pytest.mark.parametrize("bimanual", [False, True])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_disconnect_finishes_cleanup_before_reraising_interrupt(bimanual, error_type):
    calls = []
    failure = error_type()
    failures = {"left.bus" if bimanual else "bus": failure}
    robot = make_bimanual(calls, failures) if bimanual else make_follower(calls, failures)

    with pytest.raises(error_type) as exc_info:
        robot.disconnect()

    assert exc_info.value is failure
    assert calls == (
        ["left.bus", "left.camera1", "left.camera2", "right.bus", "right.camera1", "right.camera2"]
        if bimanual
        else ["bus", "camera1", "camera2"]
    )


@pytest.mark.parametrize("bimanual", [False, True])
def test_disconnect_keeps_not_connected_guard(bimanual):
    calls = []
    robot = make_bimanual(calls, {}) if bimanual else make_follower(calls, {})
    arm = robot.left_arm if bimanual else robot
    arm.bus.is_connected = False

    with pytest.raises(DeviceNotConnectedError):
        robot.disconnect()

    assert not calls
    # Prevent the still-connected right arm's destructor from adding cleanup calls.
    if bimanual:
        robot.right_arm.bus.is_connected = False


@pytest.mark.parametrize("close_fails", [False, True])
def test_bimanual_cleanup_with_real_serial_disconnect_and_fake_transport(close_fails):
    calls = []
    robot = make_bimanual(calls, {})
    transports = []
    for arm in (robot.left_arm, robot.right_arm):
        arm.bus = MockMotorsBus(
            "/dev/dummy-port",
            {f"motor_{i}": Motor(i, "model_2", MotorNormMode.DEGREES) for i in (1, 2, 3)},
        )
        arm.bus.connect(handshake=False)
        transports.append(arm.bus._bus)
        for i in (1, 2, 3):
            arm.bus._bus.seed(i, "torque_enable", 1)
    transports[0].absent.add(2)

    close_failure = OSError("close failed") if close_fails else None
    with (
        patch.object(transports[0], "close", wraps=transports[0].close, side_effect=close_failure),
        pytest.raises(OSError, match="close failed" if close_fails else "motor_2") as exc_info,
    ):
        robot.disconnect()

    if close_fails:
        assert exc_info.value is close_failure
        assert isinstance(exc_info.value.__cause__, ConnectionError)
        assert "motor_2" in str(exc_info.value.__cause__)
    assert robot.left_arm.bus.is_connected == close_fails
    assert not robot.right_arm.bus.is_connected
    assert calls == ["left.camera1", "left.camera2", "right.camera1", "right.camera2"]
    for index, transport in enumerate(transports):
        assert transport.closed == (index != 0 or not close_fails)
        assert transport.torques == [([i], False, 5) for i in (1, 2, 3)]
    assert [transports[0].stored(i, "torque_enable") for i in (1, 2, 3)] == [0, 1, 0]
    assert [transports[1].stored(i, "torque_enable") for i in (1, 2, 3)] == [0, 0, 0]
