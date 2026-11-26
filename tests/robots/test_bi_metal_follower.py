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

import time
from unittest.mock import MagicMock

import pytest

from lerobot.robots.bi_metal_follower import BiMetalFollower, BiMetalFollowerConfig
from lerobot.robots.metal_follower import MetalFollowerConfig
from tests.mocks.mock_metal import (
    fake_can,
    make_ready,
    mock_hardware as mock_arm_hardware,
    raw_states,
    without_hardware_packages,
)


def arm_config(port, **kwargs):
    return MetalFollowerConfig(port=port, gravity_compensation=False, **kwargs)


@pytest.fixture
def robot(tmp_path, monkeypatch):
    without_hardware_packages(monkeypatch)
    robot = BiMetalFollower(
        BiMetalFollowerConfig(
            id="test",
            calibration_dir=tmp_path,
            left_arm_config=arm_config("can0"),
            right_arm_config=arm_config("can1"),
        )
    )
    yield robot
    for arm in robot.arms.values():
        arm._connected = False


def mock_hardware(robot, monkeypatch):
    for arm in robot.arms.values():
        mock_arm_hardware(arm, monkeypatch)


def ready(robot):
    for arm in robot.arms.values():
        make_ready(arm)
    return dict.fromkeys(robot.action_features, 0.0)


def test_arms_need_distinct_interfaces():
    with pytest.raises(ValueError, match="distinct CAN interfaces"):
        BiMetalFollowerConfig(left_arm_config=arm_config("can0"), right_arm_config=arm_config("can0"))


def test_actions_and_observations_are_side_prefixed(robot):
    action = ready(robot)
    assert len(action) == 14 and "left_shoulder_pan.pos" in action and "right_gripper.pos" in action
    sent = robot.send_action(action)
    assert sent == pytest.approx(action)
    assert set(robot.get_observation()) == set(action)


def test_top_level_cameras_keep_their_names_with_depth(robot):
    robot._top_level_cam_keys = {"top"}
    robot.left_arm.config.cameras = {"top": MagicMock(height=4, width=6, use_rgb=True, use_depth=True)}
    robot.right_arm.config.cameras = {"wrist": MagicMock(height=4, width=6, use_rgb=True, use_depth=False)}
    assert {k for k, v in robot._cameras_ft.items()} == {"top", "top_depth", "right_wrist"}


def test_stale_feedback_on_either_arm_stops_both(robot):
    action = ready(robot)
    robot.right_arm.controller._updated_at -= 10
    with pytest.raises(ConnectionError, match="stale"):
        robot.send_action(action)
    with pytest.raises(ConnectionError, match="stopped"):
        robot.left_arm.controller.latest()


def test_bad_second_arm_position_never_enables_first(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.right_arm.bus.read_states.return_value.position[1] = -1
    with pytest.raises(ValueError, match="initial_position_deg"):
        robot.connect()
    for arm in robot.arms.values():
        arm.bus.enable.assert_not_called()
        arm.bus.disconnect.assert_called_once()
    assert not any(arm.is_connected for arm in robot.arms.values())


def test_failed_second_bus_releases_the_first(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.right_arm.bus.read_states.side_effect = ConnectionError("missing or stale motor feedback")
    with pytest.raises(ConnectionError, match="stale"):
        robot.connect()
    robot.left_arm.bus.disconnect.assert_called_once()
    assert not any(arm.is_connected for arm in robot.arms.values())


def test_second_arm_refuses_torque_after_a_fault_on_the_first(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    left = robot.left_arm.controller
    start_left = left.start

    def start_then_fault():
        start_left()
        robot._fault.set()  # what the left control loop does when it faults

    monkeypatch.setattr(left, "start", start_then_fault)
    with pytest.raises(ConnectionError):
        robot.connect()
    robot.right_arm.bus.enable.assert_not_called()
    assert not robot.is_connected


def test_reconnects_after_disconnect(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    for _ in range(2):
        robot.connect()
        assert robot.is_connected
        robot.disconnect()
        assert not robot.is_connected
    for arm in robot.arms.values():
        assert arm.bus.enable.call_count == 2


def test_partial_disconnect_can_be_retried(robot):
    for arm in robot.arms.values():
        arm._connected = True
    robot.left_arm.disconnect = MagicMock(side_effect=RuntimeError("controller did not stop"))
    robot.right_arm.disconnect = MagicMock(side_effect=lambda: setattr(robot.right_arm, "_connected", False))
    with pytest.raises(RuntimeError, match="controller did not stop"):
        robot.disconnect()
    assert not robot.is_connected
    robot.left_arm.disconnect.side_effect = lambda: setattr(robot.left_arm, "_connected", False)
    robot.disconnect()
    assert robot.left_arm.disconnect.call_count == 2
    robot.right_arm.disconnect.assert_called_once()


def test_partial_connection_must_be_disconnected_before_reconnecting(robot):
    robot.left_arm._connected = True
    with pytest.raises(RuntimeError, match="Disconnect both"):
        robot.connect()


def test_fault_on_one_arm_leaves_both_holding(robot):
    fakes = {}
    for side, arm in robot.arms.items():
        fakes[side] = fake_can(arm, feedback_timeout_s=0.02)
        for motor in fakes[side].motors.values():
            motor["status"] = 1
        arm.bus.enabled, arm.bus._enabled_at = True, time.monotonic()
        arm.controller._set_initial_state(arm.controller.read_state())
    fakes["left"].motors["elbow_flex"]["status"] = 0xC  # rotor overtemperature
    robot.left_arm.controller._run()
    robot.right_arm.controller._run()  # sees the shared fault at once
    assert robot.left_arm.controller._failure is not None and robot.right_arm.controller._failure is None
    for side, fake in fakes.items():
        hold = fake.batches[-1]
        assert hold["shoulder_lift"].kp == 100 and (hold["gripper"].kp, hold["gripper"].kd) == (0.0, 0.0)
        assert ("elbow_flex" in hold) == (side == "right")


def test_enable_rechecks_both_initial_positions(robot, monkeypatch):
    """An arm moved between the pair's position check and its own enable refuses torque; both disconnect."""
    mock_hardware(robot, monkeypatch)
    prepare = robot.right_arm.prepare_controller

    def prepare_then_move():
        prepare()
        robot.right_arm.bus.read_states.return_value = raw_states()
        robot.right_arm.bus.read_states.return_value.position[0] = 1.0

    monkeypatch.setattr(robot.right_arm, "prepare_controller", prepare_then_move)
    with pytest.raises(ValueError, match="fold it"):
        robot.connect()
    robot.right_arm.bus.enable.assert_not_called()
    robot.left_arm.bus.disconnect.assert_called_once()
    assert not any(arm.is_connected for arm in robot.arms.values())
