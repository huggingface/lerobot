from unittest.mock import Mock

import pytest

from lerobot.teleoperators.bi_so_leader.bi_so_leader import BiSOLeader
from lerobot.utils.errors import DeviceAlreadyConnectedError


class FakeArm:
    def __init__(self, fail_connect=False, fail_disconnect=False):
        self.is_connected = False
        self.fail_connect = fail_connect
        self.fail_disconnect = fail_disconnect
        self.calls = []

    def connect(self, calibrate=True):
        self.calls.append(("connect", calibrate))
        self.is_connected = True
        if self.fail_connect:
            raise RuntimeError("injected connect failure")

    def disconnect(self):
        self.calls.append(("disconnect",))
        if self.fail_disconnect:
            raise RuntimeError("injected disconnect failure")
        self.is_connected = False


def make_bimanual(left=None, right=None):
    instance = object.__new__(BiSOLeader)
    instance.left_arm = left or FakeArm()
    instance.right_arm = right or FakeArm()
    return instance


def test_normal_connect():
    robot = make_bimanual()

    robot.connect(calibrate=False)

    assert robot.left_arm.is_connected
    assert robot.right_arm.is_connected


def test_right_connect_failure_rolls_back_both_arms():
    left = FakeArm()
    right = FakeArm(fail_connect=True)
    robot = make_bimanual(left, right)

    with pytest.raises(RuntimeError, match="injected connect failure"):
        robot.connect(calibrate=False)

    assert not left.is_connected
    assert not right.is_connected
    assert ("disconnect",) in left.calls
    assert ("disconnect",) in right.calls


def test_partial_connection_is_rejected():
    left = FakeArm()
    left.is_connected = True
    robot = make_bimanual(left, FakeArm())

    with pytest.raises(DeviceAlreadyConnectedError):
        robot.connect()

    assert left.is_connected
    assert not robot.right_arm.is_connected


def test_disconnect_handles_partial_connection():
    left = FakeArm()
    left.is_connected = True
    right = FakeArm()
    robot = make_bimanual(left, right)

    robot.disconnect()

    assert not left.is_connected
    assert not right.is_connected
    assert ("disconnect",) in left.calls
    assert not right.calls


def test_disconnect_attempts_both_arms_after_failure():
    left = FakeArm(fail_disconnect=True)
    right = FakeArm()
    left.is_connected = True
    right.is_connected = True
    robot = make_bimanual(left, right)

    with pytest.raises(RuntimeError, match="injected disconnect failure"):
        robot.disconnect()

    assert left.is_connected
    assert not right.is_connected
    assert ("disconnect",) in right.calls


def test_disconnect_is_safe_when_both_disconnected():
    robot = make_bimanual()

    robot.disconnect()

    assert not robot.left_arm.calls
    assert not robot.right_arm.calls
