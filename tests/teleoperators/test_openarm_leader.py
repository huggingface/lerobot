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

from lerobot.teleoperators.openarm_leader import OpenArmLeader, OpenArmLeaderConfig

_ADAPTER = "lerobot.motors.damiao.motorbridge_bus"
_LEADER = "lerobot.teleoperators.openarm_leader.openarm_leader"

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


def _make_controller_mock(positions_deg):
    controller = MagicMock(name="MotorBridgeControllerMock")
    controller._count = 0

    def _add_motor(motor_id, feedback_id, model):
        index = controller._count
        controller._count += 1
        handle = MagicMock(name=f"MotorHandle{index}")
        state = MagicMock()
        state.pos = math.radians(positions_deg[index])
        state.vel = 0.0
        state.torq = 0.0
        state.t_mos = 0.0
        state.t_rotor = 0.0
        handle.get_state.return_value = state
        return handle

    controller.add_damiao_motor.side_effect = _add_motor
    return controller


@contextmanager
def _connected(*, positions_deg, use_can_fd=True, macos=False, **config_kwargs):
    controller = _make_controller_mock(positions_deg)
    with (
        patch(f"{_LEADER}.require_package", lambda *a, **kw: None),
        patch(f"{_ADAPTER}.require_package", lambda *a, **kw: None),
        patch(f"{_ADAPTER}._is_macos", return_value=macos),
        patch(f"{_ADAPTER}.MotorBridgeController") as controller_cls,
        patch(f"{_ADAPTER}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_socketcanfd.return_value = controller
        controller_cls.return_value = controller
        teleop = OpenArmLeader(OpenArmLeaderConfig(port="can0", use_can_fd=use_can_fd, **config_kwargs))
        teleop.connect(calibrate=False)
        try:
            yield teleop, controller, controller_cls
        finally:
            if teleop.is_connected:
                teleop.disconnect()


def test_connect_maps_models_and_ids():
    with _connected(positions_deg=list(range(8))) as (_teleop, controller, _cls):
        got = {tuple(c.args) for c in controller.add_damiao_motor.call_args_list}
        assert got == set(_EXPECTED_MOTORS.values())


def test_manual_control_disables_torque_on_connect():
    with _connected(positions_deg=list(range(8)), manual_control=True) as (_teleop, controller, _cls):
        controller.disable_all.assert_called()


def test_get_action_is_reported_in_degrees():
    positions = [10, 20, 30, 40, 50, 60, 70, -30]
    with _connected(positions_deg=positions) as (teleop, _controller, _cls):
        action = teleop.get_action()
    for motor, expected in zip(_EXPECTED_MOTORS, positions, strict=True):
        assert action[f"{motor}.pos"] == pytest.approx(expected)


def test_macos_uses_libusb_classic_backend():
    with _connected(positions_deg=list(range(8)), macos=True) as (_teleop, _controller, controller_cls):
        controller_cls.assert_called_once_with(channel="pcanfd:can0")
        controller_cls.from_socketcanfd.assert_not_called()
