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

import math
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

_ADAPTER = "lerobot.motors.damiao.motorbridge_bus"
_FOLLOWER = "lerobot.robots.openarm_follower.openarm_follower"

# joint -> (send_id, recv_id, motorbridge model) from the default motor_config.
_EXPECTED_MOTORS = {
    "joint_1": (0x01, 0x11, "8009"),
    "joint_2": (0x02, 0x12, "8009"),
    "joint_3": (0x03, 0x13, "4340"),
    "joint_4": (0x04, 0x14, "4340"),
    "joint_5": (0x05, 0x15, "4310"),
    "joint_6": (0x06, 0x16, "4310"),
    "joint_7": (0x07, 0x17, "4310"),
    "gripper": (0x08, 0x18, "4310"),
}
_WIDE_LIMITS = dict.fromkeys(_EXPECTED_MOTORS, (-180.0, 180.0))


def _make_controller_mock(positions_deg=None):
    """Controller whose motors report a known position (degrees), in creation order."""
    controller = MagicMock(name="MotorBridgeControllerMock")
    controller._count = 0

    def _add_motor(motor_id, feedback_id, model):
        index = controller._count
        controller._count += 1
        pos_deg = index + 1 if positions_deg is None else positions_deg[index]
        handle = MagicMock(name=f"MotorHandle{index}")
        handle.model = model
        state = MagicMock()
        state.pos = math.radians(pos_deg)
        state.vel = 0.0
        state.torq = 0.0
        state.t_mos = 0.0
        state.t_rotor = 0.0
        handle.get_state.return_value = state
        return handle

    controller.add_damiao_motor.side_effect = _add_motor
    return controller


@contextmanager
def _connected(*, positions_deg=None, use_can_fd=True, macos=False, **config_kwargs):
    controller = _make_controller_mock(positions_deg)
    config_kwargs.setdefault("joint_limits", _WIDE_LIMITS)
    with (
        patch(f"{_FOLLOWER}.require_package", lambda *a, **kw: None),
        patch(f"{_ADAPTER}.require_package", lambda *a, **kw: None),
        patch(f"{_ADAPTER}._is_macos", return_value=macos),
        patch(f"{_ADAPTER}.MotorBridgeController") as controller_cls,
        patch(f"{_ADAPTER}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_socketcanfd.return_value = controller
        controller_cls.return_value = controller
        config = OpenArmFollowerConfig(port="can0", use_can_fd=use_can_fd, **config_kwargs)
        robot = OpenArmFollower(config)
        robot.connect(calibrate=False)
        try:
            yield robot, controller, controller_cls
        finally:
            if robot.is_connected:
                robot.disconnect()


def test_features_match_joints():
    with patch(f"{_FOLLOWER}.require_package", lambda *a, **kw: None):
        robot = OpenArmFollower(OpenArmFollowerConfig(port="can0"))
    expected = {f"{motor}.pos" for motor in _EXPECTED_MOTORS}
    assert set(robot.action_features) == expected
    assert set(robot.observation_features) == expected


def test_connect_maps_models_and_ids():
    with _connected() as (_robot, controller, _cls):
        # add_damiao_motor is called positionally as (send_id, recv_id, model).
        got = {tuple(c.args) for c in controller.add_damiao_motor.call_args_list}
        assert got == set(_EXPECTED_MOTORS.values())


@pytest.mark.parametrize(
    ("use_can_fd", "uses_fd"),
    [(True, True), (False, False)],
)
def test_connect_selects_transport(use_can_fd, uses_fd):
    with _connected(use_can_fd=use_can_fd) as (_robot, _controller, controller_cls):
        if uses_fd:
            controller_cls.from_socketcanfd.assert_called_once_with("can0")
            controller_cls.assert_not_called()
        else:
            controller_cls.assert_called_once_with(channel="can0")
            controller_cls.from_socketcanfd.assert_not_called()


@pytest.mark.parametrize("use_can_fd", [True, False])
def test_macos_uses_libusb_classic_backend(use_can_fd):
    # On macOS, the libusb PCAN backend is selected via the ``pcanfd:`` channel
    # prefix and classic CAN is forced regardless of the configured use_can_fd.
    with _connected(use_can_fd=use_can_fd, macos=True) as (_robot, _controller, controller_cls):
        controller_cls.assert_called_once_with(channel="pcanfd:can0")
        controller_cls.from_socketcanfd.assert_not_called()


def test_observation_is_reported_in_degrees():
    positions = [10, 20, 30, 40, 50, 60, 70, -30]
    with _connected(positions_deg=positions) as (robot, _controller, _cls):
        obs = robot.get_observation()
    for (motor, _ids), expected in zip(_EXPECTED_MOTORS.items(), positions, strict=True):
        assert obs[f"{motor}.pos"] == pytest.approx(expected)


def test_send_action_converts_degrees_to_radians_with_gains():
    with _connected() as (robot, _controller, _cls):
        sent = robot.send_action({"joint_1.pos": 30.0})
        handle = robot.bus._handles["joint_1"]
        pos, vel, kp, kd, tau = handle.send_mit.call_args.args
        assert pos == pytest.approx(math.radians(30.0))
        assert vel == pytest.approx(0.0)
        # position_kp[0]=240.0, position_kd[0]=5.0 from the default config.
        assert (kp, kd, tau) == (240.0, 5.0, 0.0)
        assert sent["joint_1.pos"] == pytest.approx(30.0)


def test_send_action_clips_to_joint_limits():
    with _connected(joint_limits={**_WIDE_LIMITS, "joint_1": (-5.0, 5.0)}) as (robot, _controller, _cls):
        sent = robot.send_action({"joint_1.pos": 30.0})
        handle = robot.bus._handles["joint_1"]
        pos = handle.send_mit.call_args.args[0]
        assert pos == pytest.approx(math.radians(5.0))
        assert sent["joint_1.pos"] == pytest.approx(5.0)


def test_disconnect_disables_and_closes():
    with _connected(disable_torque_on_disconnect=True) as (robot, controller, _cls):
        handles = list(robot.bus._handles.values())
        robot.disconnect()
        for handle in handles:
            handle.disable.assert_called_once()
            handle.close.assert_called_once()
        controller.close.assert_called_once()
        assert not robot.is_connected
