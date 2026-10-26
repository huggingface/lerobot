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

"""Tests for the LimX TRON1 robot class.

These drive the driver against a stand-in for the vendor SDK, so they need no robot, no network
and no ``limxsdk`` install. The stand-in reproduces the parts of the SDK contract the driver
relies on -- ``init`` returning a boolean, a ``subscribeRobotState`` push callback, and
``publishRobotCmd`` taking an attribute bag -- so the real vendor package is never imported.
"""

import os
import sys
import threading
import time
import types

import pytest

from lerobot.robots.limx_tron1 import LimxTron1, LimxTron1Config
from lerobot.robots.limx_tron1.joints import pos_key


class FakeRobotCmd:
    """Stand-in for ``limxsdk.datatypes.RobotCmd`` -- an attribute bag with no behaviour."""

    def __init__(self):
        self.stamp = 0
        self.mode = []
        self.q = []
        self.dq = []
        self.tau = []
        self.Kp = []
        self.Kd = []
        self.motor_names = []
        self.parallel_solve_required = []


class FakeDatatypes:
    """Stand-in for the ``limxsdk.datatypes`` module."""

    RobotCmd = FakeRobotCmd


class FakeRobotState:
    """Stand-in for ``limxsdk.datatypes.RobotState``."""

    def __init__(self, q):
        self.stamp = 0
        self.tau = [0.0] * len(q)
        self.q = list(q)
        self.dq = [0.0] * len(q)
        self.motor_names = []


class FakeVendorRobot:
    """Records published commands and lets a test push a state sample."""

    def __init__(self, init_ok: bool = True):
        self._init_ok = init_ok
        self.init_calls: list[str] = []
        self.published: list[FakeRobotCmd] = []
        self._state_callback = None

    def init(self, robot_ip: str) -> bool:
        self.init_calls.append(robot_ip)
        return self._init_ok

    def subscribeRobotState(self, callback) -> None:  # noqa: N802 - vendor method name
        self._state_callback = callback

    def publishRobotCmd(self, cmd: FakeRobotCmd) -> None:  # noqa: N802 - vendor method name
        self.published.append(cmd)

    def push_state(self, q) -> None:
        """Deliver a state sample the way the SDK's subscriber would."""
        assert self._state_callback is not None, "subscribeRobotState was never called"
        self._state_callback(FakeRobotState(q))


class HarnessRobot(LimxTron1):
    """``LimxTron1`` wired to a fake SDK handle instead of the vendor package."""

    def __init__(self, config: LimxTron1Config, vendor: FakeVendorRobot) -> None:
        super().__init__(config)
        self._vendor = vendor

    def _make_robot(self):
        # Mirror what the real method does with the SDK handle, so tests exercise the same
        # subscribe-then-publish contract.
        self._vendor.subscribeRobotState(self._on_state)
        return self._vendor, FakeDatatypes


def make_config(**overrides) -> LimxTron1Config:
    base = {"robot_ip": "127.0.0.1", "publish_rate": 100.0}
    base.update(overrides)
    return LimxTron1Config(**base)


def install_fake_limxsdk(monkeypatch, recorded: dict):
    """Put fake ``limxsdk`` modules on the import path, recording the env at import time."""

    def vendor_factory(robot_type):
        recorded["robot_type"] = robot_type
        recorded["env"] = os.environ.get("ROBOT_TYPE")
        return FakeVendorRobot()

    robot_module = types.ModuleType("limxsdk.robot")
    robot_module.Robot = vendor_factory
    robot_module.RobotType = types.SimpleNamespace(PointFoot="PointFoot")

    top_module = types.ModuleType("limxsdk")
    top_module.datatypes = FakeDatatypes

    monkeypatch.setitem(sys.modules, "limxsdk", top_module)
    monkeypatch.setitem(sys.modules, "limxsdk.robot", robot_module)
    monkeypatch.setattr("lerobot.robots.limx_tron1.limx_tron1.require_package", lambda *args, **kwargs: None)


# ------------------------------------------------------------------ features


def test_observation_features_match_get_observation():
    vendor = FakeVendorRobot()
    robot = HarnessRobot(make_config(), vendor)
    robot.connect()
    try:
        vendor.push_state(range(len(robot.config.joint_names)))
        assert set(robot.get_observation()) == set(robot.observation_features)
        assert set(robot.action_features) == set(robot.observation_features)
    finally:
        robot.disconnect()


def test_observation_reports_measured_state():
    vendor = FakeVendorRobot()
    robot = HarnessRobot(make_config(robot_type="SF_TRON1A"), vendor)
    robot.connect()
    try:
        dim = len(robot.config.joint_names)
        positions = [0.1 * index for index in range(dim)]
        vendor.push_state(positions)
        observation = robot.get_observation()
        for name, expected in zip(robot.config.joint_names, positions, strict=True):
            assert observation[pos_key(name)] == pytest.approx(expected)
    finally:
        robot.disconnect()


# ------------------------------------------------------------------- actions


def test_action_reaches_the_robot_as_a_point_foot_command():
    vendor = FakeVendorRobot()
    robot = HarnessRobot(make_config(robot_type="PF_TRON1A", publish_rate=0.0001), vendor)
    robot.connect()
    try:
        action = {pos_key(name): 0.25 for name in robot.config.joint_names}
        robot.send_action(action)
    finally:
        robot.disconnect()

    assert vendor.published, "send_action published no command"
    command = vendor.published[-1]
    dim = len(robot.config.joint_names)
    assert dim == 6
    assert command.q == pytest.approx([0.25] * dim)
    assert len(command.Kp) == dim
    assert len(command.Kd) == dim
    # The legged builds route commands by index and have no parallel-linkage conversion, so the
    # vendor's own controller leaves both fields empty -- and so does this driver.
    assert command.motor_names == []
    assert command.parallel_solve_required == []
    # mode 0 is the SDK's torque-position hybrid mode, matching the vendor examples.
    assert command.mode == [0] * dim


def test_action_reaches_the_robot_as_a_sole_foot_command():
    vendor = FakeVendorRobot()
    robot = HarnessRobot(make_config(robot_type="SF_TRON1A", publish_rate=0.0001), vendor)
    robot.connect()
    try:
        action = {pos_key(name): 0.25 for name in robot.config.joint_names}
        robot.send_action(action)
    finally:
        robot.disconnect()

    command = vendor.published[-1]
    assert len(command.q) == 8
    assert command.q == pytest.approx([0.25] * 8)
    assert command.Kp == pytest.approx([45.0] * 8)
    assert command.Kd == pytest.approx([3.0, 3.0, 3.0, 1.5, 3.0, 3.0, 3.0, 1.5])


def test_action_reaches_the_robot_as_a_wheel_foot_command():
    vendor = FakeVendorRobot()
    robot = HarnessRobot(make_config(robot_type="WF_TRON1A", publish_rate=0.0001), vendor)
    robot.connect()
    try:
        action = {pos_key(name): 0.25 for name in robot.config.joint_names}
        robot.send_action(action)
    finally:
        robot.disconnect()

    command = vendor.published[-1]
    assert len(command.q) == 8
    assert command.q == pytest.approx([0.25] * 8)
    assert command.Kd == pytest.approx([2.5, 2.5, 2.5, 0.8, 2.5, 2.5, 2.5, 0.8])


def test_action_order_follows_joint_names():
    vendor = FakeVendorRobot()
    robot = HarnessRobot(make_config(robot_type="SF_TRON1A", publish_rate=0.0001), vendor)
    robot.connect()
    try:
        action = {pos_key(name): float(index) for index, name in enumerate(robot.config.joint_names)}
        robot.send_action(action)
    finally:
        robot.disconnect()

    assert vendor.published[-1].q == pytest.approx(list(range(8)))


def test_send_action_returns_only_the_commanded_setpoint():
    robot = HarnessRobot(make_config(robot_type="SF_TRON1A", publish_rate=0.0001), FakeVendorRobot())
    robot.connect()
    try:
        action = {pos_key(name): float(index) for index, name in enumerate(robot.config.joint_names)}
        sent = robot.send_action({**action, "unrelated.pos": 1.0})
    finally:
        robot.disconnect()

    assert sent == action
    assert set(sent) == set(robot.action_features)


def test_missing_joint_key_is_reported():
    robot = HarnessRobot(make_config(), FakeVendorRobot())
    robot.connect()
    try:
        with pytest.raises(KeyError, match="missing joint keys"):
            robot.send_action({pos_key("abad_L_Joint"): 0.0})
    finally:
        robot.disconnect()


# ------------------------------------------------------------ publish loop


def test_start_pose_is_published_once_connected():
    vendor = FakeVendorRobot()
    robot = HarnessRobot(make_config(start_pose=[0.5] * 6, publish_rate=200.0), vendor)
    robot.connect()
    try:
        deadline = time.monotonic() + 1.0
        while not vendor.published and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        robot.disconnect()

    assert vendor.published, "the publish loop never sent the start pose"
    assert vendor.published[0].q == pytest.approx([0.5] * 6)


def test_nothing_is_published_without_a_target():
    """With no start pose and no action, the driver must leave the robot alone."""
    vendor = FakeVendorRobot()
    robot = HarnessRobot(make_config(publish_rate=200.0), vendor)
    robot.connect()
    try:
        time.sleep(0.15)
        assert not vendor.published, "the driver commanded the robot before being asked to"
    finally:
        robot.disconnect()


def test_publish_loop_keeps_sending_the_latest_target():
    vendor = FakeVendorRobot()
    robot = HarnessRobot(make_config(publish_rate=200.0), vendor)
    robot.connect()
    try:
        action = {pos_key(name): 0.125 for name in robot.config.joint_names}
        robot.send_action(action)
        time.sleep(0.2)
    finally:
        robot.disconnect()

    assert len(vendor.published) > 1, "the target was sent once but never republished"


class OverlapDetectingVendorRobot(FakeVendorRobot):
    """Records whether two threads were ever inside ``publishRobotCmd`` at once."""

    def __init__(self) -> None:
        super().__init__()
        self._active = 0
        self.max_concurrent = 0

    def publishRobotCmd(self, cmd: FakeRobotCmd) -> None:  # noqa: N802 - vendor method name
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        time.sleep(0.0005)  # widen the window an unsynchronised caller would race in
        super().publishRobotCmd(cmd)
        self._active -= 1


def test_send_action_and_publish_loop_never_publish_concurrently():
    vendor = OverlapDetectingVendorRobot()
    robot = HarnessRobot(make_config(publish_rate=2000.0), vendor)
    robot.connect()
    try:
        action = {pos_key(name): 0.0 for name in robot.config.joint_names}
        for _ in range(200):
            robot.send_action(action)
    finally:
        robot.disconnect()

    assert vendor.max_concurrent == 1, "the SDK handle was published to from two threads at once"


# ------------------------------------------------------- robot_type export


def test_robot_type_is_exported_before_the_vendor_import(monkeypatch):
    recorded: dict = {}
    install_fake_limxsdk(monkeypatch, recorded)
    monkeypatch.delenv("ROBOT_TYPE", raising=False)

    robot = LimxTron1(make_config(robot_type="SF_TRON1A"))
    robot._make_robot()

    assert recorded["env"] == "SF_TRON1A"
    assert recorded["robot_type"] == "PointFoot"


def test_vendor_default_is_left_alone_when_robot_type_is_unset(monkeypatch):
    recorded: dict = {}
    install_fake_limxsdk(monkeypatch, recorded)
    monkeypatch.delenv("ROBOT_TYPE", raising=False)

    robot = LimxTron1(make_config())
    robot._make_robot()

    assert "ROBOT_TYPE" not in os.environ
    assert recorded["env"] is None


# ----------------------------------------------------------------- lifecycle


def test_operations_require_a_connection():
    robot = HarnessRobot(make_config(), FakeVendorRobot())
    assert not robot.is_connected
    with pytest.raises(RuntimeError, match="not connected"):
        robot.send_action({})
    with pytest.raises(RuntimeError, match="not connected"):
        robot.get_observation()


def test_connect_reports_a_failed_initialisation():
    robot = HarnessRobot(make_config(), FakeVendorRobot(init_ok=False))
    with pytest.raises(ConnectionError, match="failed to initialise"):
        robot.connect()
    assert not robot.is_connected


def test_context_manager_connects_and_disconnects():
    vendor = FakeVendorRobot()
    robot = HarnessRobot(make_config(), vendor)
    with robot as connected:
        assert connected.is_connected
    assert not robot.is_connected
    assert vendor.init_calls == ["127.0.0.1"]


def test_disconnect_is_idempotent():
    robot = HarnessRobot(make_config(), FakeVendorRobot())
    robot.connect()
    robot.disconnect()
    robot.disconnect()
    assert not robot.is_connected


def test_observation_times_out_without_state():
    robot = HarnessRobot(make_config(observation_timeout=0.05), FakeVendorRobot())
    robot.connect()
    try:
        with pytest.raises(RuntimeError, match="no fresh state sample"):
            robot.get_observation()
    finally:
        robot.disconnect()


def test_stale_state_is_rejected():
    vendor = FakeVendorRobot()
    robot = HarnessRobot(make_config(observation_timeout=0.05, state_max_age=0.02), vendor)
    robot.connect()
    try:
        vendor.push_state(range(len(robot.config.joint_names)))
        robot.get_observation()  # fresh: accepted
        time.sleep(0.08)
        with pytest.raises(RuntimeError, match="no fresh state sample"):
            robot.get_observation()
    finally:
        robot.disconnect()


# ------------------------------------------- command ordering and reconnects


class OwnedLock:
    """A lock that records which thread holds it.

    ``threading.Lock`` cannot answer "is the current thread the one holding this?", which is what
    the ordering test needs: the publish loop has to read the target while holding the lock that
    `send_action` also takes, and there is no other way to observe that from outside.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owner: int | None = None

    def __enter__(self):
        self._lock.acquire()
        self._owner = threading.get_ident()
        return self

    def __exit__(self, *exc_info):
        self._owner = None
        self._lock.release()
        return False

    def owned_by_current_thread(self) -> bool:
        return self._owner == threading.get_ident()


class LockScopeHarness(HarnessRobot):
    """``HarnessRobot`` that flags every read of the target made outside the publish lock."""

    def __init__(self, config: LimxTron1Config, vendor: FakeVendorRobot) -> None:
        super().__init__(config, vendor)
        self.target_reads_outside_lock = 0
        self._publish_lock = OwnedLock()

    @property
    def _target(self):
        lock = self.__dict__.get("_publish_lock")
        if lock is not None and not lock.owned_by_current_thread():
            self.target_reads_outside_lock += 1
        return self.__dict__.get("_target_storage")

    @_target.setter
    def _target(self, value):
        self.__dict__["_target_storage"] = value


def test_the_publish_loop_reads_the_target_inside_the_publish_lock():
    """Reading the target outside the lock is what let an older setpoint overtake a newer one.

    With the read outside, the loop could snapshot target A, then `send_action` could install and
    publish B and return, and only then would the loop publish A -- the robot receives B followed
    by the older A.  The read and the publish have to be one critical section.
    """
    vendor = FakeVendorRobot()
    config = make_config(publish_rate=1000.0)
    config.start_pose = [0.1] * len(config.joint_names)
    robot = LockScopeHarness(config, vendor)

    robot.connect()
    try:
        names = robot.config.joint_names
        for step in range(20):
            robot.send_action({pos_key(name): 0.2 + step * 0.01 for name in names})
            time.sleep(0.01)
    finally:
        robot.disconnect()

    assert robot.target_reads_outside_lock == 0, (
        f"the publish loop read the target {robot.target_reads_outside_lock} times without the "
        "lock, so a stale target can be published after a newer one"
    )


class BlockingVendorRobot(FakeVendorRobot):
    """Holds the first ``publishRobotCmd`` open so a disconnect can time out joining it."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def publishRobotCmd(self, cmd: FakeRobotCmd) -> None:  # noqa: N802 - vendor method name
        self.entered.set()
        self.release.wait(timeout=10)
        super().publishRobotCmd(cmd)


class SwappableHarness(LimxTron1):
    """``LimxTron1`` that hands out a different vendor handle on each connection."""

    def __init__(self, config: LimxTron1Config, vendors: list) -> None:
        super().__init__(config)
        self._vendors = list(vendors)

    def _make_robot(self):
        vendor = self._vendors.pop(0)
        vendor.subscribeRobotState(self._on_state)
        return vendor, FakeDatatypes


def test_a_publisher_stranded_by_disconnect_does_not_outlive_that_connection():
    """A thread still blocked inside the SDK when ``disconnect`` times out must not survive.

    ``disconnect`` only joins for two seconds.  When it gives up, the old loop is still inside
    ``publishRobotCmd``; the next ``connect`` used to clear the shared stop event, which let that
    thread resume and publish to the freshly opened handle.
    """
    first = BlockingVendorRobot()
    second = FakeVendorRobot()
    config = make_config(publish_rate=1000.0)
    # A target has to exist before the loop will call the SDK at all.
    config.start_pose = [0.1] * len(config.joint_names)
    robot = SwappableHarness(config, [first, second])

    robot.connect()
    assert first.entered.wait(5), "the publish loop never reached the SDK"
    stranded = robot._publisher

    robot.disconnect()  # join times out: the loop is still blocked in the SDK call
    assert stranded.is_alive(), "expected the publisher to be stranded inside publishRobotCmd"

    robot.connect()  # second connection, second stop event
    first.release.set()  # let the stranded thread return from the SDK

    stranded.join(timeout=5)
    try:
        assert not stranded.is_alive(), "a publisher from the previous connection is still running"
    finally:
        robot.disconnect()


@pytest.mark.parametrize("field", ["start_pose", "kp", "kd"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_tables_are_rejected_before_anything_reaches_the_sdk(field, bad):
    dim = len(make_config().joint_names)
    with pytest.raises(ValueError, match="non-finite"):
        make_config(**{field: [bad] * dim})
