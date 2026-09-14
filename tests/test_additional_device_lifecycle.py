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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.earthrover_mini_plus.robot_earthrover_mini_plus import EarthRoverMiniPlus
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadTeleop
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.reachy2_teleoperator.reachy2_teleoperator import Reachy2Teleoperator
from lerobot.utils.errors import DeviceNotConnectedError


class FakeKeyboardListener:
    instances = []

    def __init__(self, **kwargs):
        self.alive = False
        self.stop_calls = 0
        self.instances.append(self)

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def stop(self):
        self.stop_calls += 1
        self.alive = False


class FailingKeyboardListener(FakeKeyboardListener):
    def start(self):
        self.alive = True
        raise RuntimeError("listener startup failed")


def test_earthrover_connect_and_disconnect_are_idempotent():
    robot = object.__new__(EarthRoverMiniPlus)
    robot.id = "test"
    robot._is_connected = False
    robot.sdk_base_url = "http://localhost:8000"
    robot.calibrate = MagicMock()
    robot._send_command_to_sdk = MagicMock()

    response = SimpleNamespace(status_code=200)
    with patch(
        "lerobot.robots.earthrover_mini_plus.robot_earthrover_mini_plus.requests.get",
        return_value=response,
    ) as request:
        robot.connect()
        robot.connect()

    request.assert_called_once()
    robot.calibrate.assert_called_once_with()

    robot.disconnect()
    robot.disconnect()

    robot._send_command_to_sdk.assert_called_once_with(0.0, 0.0)


def test_lekiwi_client_disconnect_cleans_partial_handles_and_is_idempotent():
    robot = object.__new__(LeKiwiClient)
    robot.id = "test"
    robot._is_connected = False
    robot.zmq_observation_socket = MagicMock()
    robot.zmq_cmd_socket = MagicMock()
    robot.zmq_context = MagicMock()
    observation_socket = robot.zmq_observation_socket
    command_socket = robot.zmq_cmd_socket
    context = robot.zmq_context

    robot.disconnect()
    robot.disconnect()

    observation_socket.close.assert_called_once_with()
    command_socket.close.assert_called_once_with()
    context.term.assert_called_once_with()


def test_gamepad_connect_noops_when_connected_and_disconnect_is_idempotent():
    teleop = object.__new__(GamepadTeleop)
    teleop.id = "test"
    teleop.gamepad = MagicMock()
    gamepad = teleop.gamepad

    teleop.connect()
    assert teleop.gamepad is gamepad

    teleop.disconnect()
    teleop.disconnect()

    gamepad.stop.assert_called_once_with()


def test_reachy_teleoperator_connect_and_disconnect_are_idempotent():
    teleop = object.__new__(Reachy2Teleoperator)
    teleop.id = "test"
    teleop.config = SimpleNamespace(ip_address="192.0.2.1")
    teleop.reachy = None
    sdk = MagicMock()
    sdk.is_connected.return_value = True

    with patch(
        "lerobot.teleoperators.reachy2_teleoperator.reachy2_teleoperator.ReachySDK",
        return_value=sdk,
    ) as sdk_factory:
        teleop.connect()
        teleop.connect()

    sdk_factory.assert_called_once_with("192.0.2.1")

    teleop.disconnect()
    teleop.disconnect()

    sdk.disconnect.assert_called_once_with()


def test_keyboard_connect_and_disconnect_are_idempotent(monkeypatch):
    import lerobot.teleoperators.keyboard.teleop_keyboard as keyboard_module

    FakeKeyboardListener.instances.clear()
    monkeypatch.setattr(keyboard_module, "PYNPUT_AVAILABLE", True)
    monkeypatch.setattr(keyboard_module, "pynput_can_capture", lambda: True)
    monkeypatch.setattr(
        keyboard_module,
        "keyboard",
        SimpleNamespace(Listener=FakeKeyboardListener, Key=SimpleNamespace(esc="esc")),
    )
    teleop = object.__new__(KeyboardTeleop)
    teleop.id = "test"
    teleop.listener = None

    teleop.connect()
    teleop.connect()

    assert len(FakeKeyboardListener.instances) == 1
    listener = FakeKeyboardListener.instances[0]

    teleop.disconnect()
    teleop.disconnect()

    assert listener.stop_calls == 1


def test_keyboard_connect_raises_when_capture_is_unavailable(monkeypatch):
    import lerobot.teleoperators.keyboard.teleop_keyboard as keyboard_module

    monkeypatch.setattr(keyboard_module, "PYNPUT_AVAILABLE", True)
    monkeypatch.setattr(keyboard_module, "pynput_can_capture", lambda: False)
    teleop = object.__new__(KeyboardTeleop)
    teleop.id = "test"
    teleop.listener = None

    with pytest.raises(DeviceNotConnectedError, match="Keyboard teleoperation is unavailable"):
        teleop.connect()

    assert not teleop.is_connected


def test_keyboard_connect_rolls_back_a_listener_startup_failure(monkeypatch):
    import lerobot.teleoperators.keyboard.teleop_keyboard as keyboard_module

    FailingKeyboardListener.instances.clear()
    monkeypatch.setattr(keyboard_module, "PYNPUT_AVAILABLE", True)
    monkeypatch.setattr(keyboard_module, "pynput_can_capture", lambda: True)
    monkeypatch.setattr(
        keyboard_module,
        "keyboard",
        SimpleNamespace(Listener=FailingKeyboardListener, Key=SimpleNamespace(esc="esc")),
    )
    teleop = object.__new__(KeyboardTeleop)
    teleop.id = "test"
    teleop.listener = None

    with pytest.raises(RuntimeError, match="listener startup failed"):
        teleop.connect()

    assert teleop.listener is None
    assert FailingKeyboardListener.instances[0].stop_calls == 1
