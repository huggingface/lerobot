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

from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop


class _DummyListener:
    def is_alive(self):
        return True


def test_keyboard_ee_get_action_outputs_task_space_command(monkeypatch):
    key = SimpleNamespace(
        space="space",
        left="left",
        right="right",
        up="up",
        down="down",
        shift="shift",
        shift_r="shift_r",
        ctrl_l="ctrl_l",
        ctrl_r="ctrl_r",
    )
    keyboard_stub = SimpleNamespace(Listener=_DummyListener, Key=key)

    monkeypatch.setattr("lerobot.teleoperators.keyboard.teleop_keyboard.PYNPUT_AVAILABLE", True)
    monkeypatch.setattr("lerobot.teleoperators.keyboard.teleop_keyboard.keyboard", keyboard_stub)
    monkeypatch.setattr("lerobot.teleoperators.keyboard.teleop_keyboard.require_package", lambda *_a, **_k: None)

    teleop = KeyboardEndEffectorTeleop(KeyboardEndEffectorTeleopConfig())
    teleop.listener = _DummyListener()
    teleop.current_pressed = {
        key.space: True,
        key.left: True,
        "i": True,
        "x": True,
    }

    action = teleop.get_action()

    assert action["enabled"] == 1.0
    assert action["target_x"] > 0.0
    assert action["target_wx"] > 0.0
    assert action["gripper_vel"] > 0.0


def test_keyboard_ee_deadman_blocks_motion(monkeypatch):
    key = SimpleNamespace(
        space="space",
        left="left",
        right="right",
        up="up",
        down="down",
        shift="shift",
        shift_r="shift_r",
        ctrl_l="ctrl_l",
        ctrl_r="ctrl_r",
    )
    keyboard_stub = SimpleNamespace(Listener=_DummyListener, Key=key)

    monkeypatch.setattr("lerobot.teleoperators.keyboard.teleop_keyboard.PYNPUT_AVAILABLE", True)
    monkeypatch.setattr("lerobot.teleoperators.keyboard.teleop_keyboard.keyboard", keyboard_stub)
    monkeypatch.setattr("lerobot.teleoperators.keyboard.teleop_keyboard.require_package", lambda *_a, **_k: None)

    teleop = KeyboardEndEffectorTeleop(KeyboardEndEffectorTeleopConfig(require_deadman=True))
    teleop.listener = _DummyListener()
    teleop.current_pressed = {key.left: True, "i": True, "x": True}

    action = teleop.get_action()

    assert action["enabled"] == 0.0
    assert action["target_x"] == 0.0
    assert action["target_wx"] == 0.0
    assert action["gripper_vel"] == 0.0
