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

from unittest.mock import patch, PropertyMock
import numpy as np
import pytest

from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardReachyMiniTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardReachyMiniTeleopConfig

@pytest.fixture
def teleop():
    with patch("lerobot.teleoperators.keyboard.teleop_keyboard.KeyboardTeleop.is_connected", new_callable=PropertyMock) as mock_is_connected:
        mock_is_connected.return_value = True
        config = KeyboardReachyMiniTeleopConfig()
        teleop = KeyboardReachyMiniTeleop(config)
        
        # Manually initialize the event queue elements since we bypass connect
        teleop.current_pressed = {}
        
        yield teleop

def test_keyboard_reachy_mini_initial_action(teleop):
    action = teleop.action
    assert action["ee.x"] == 0.0
    assert action["ee.y"] == 0.0
    assert action["ee.z"] == 0.0
    assert action["ee.wx"] == 0.0
    assert action["ee.wy"] == 0.0
    assert action["ee.wz"] == 0.0
    assert action["right_antenna.pos"] == 0.0
    assert action["left_antenna.pos"] == 0.0

def test_keyboard_reachy_mini_get_action_pitch(teleop):
    # Simulate 'w' press (Pitch forward)
    teleop.current_pressed = {"w": True}
    action = teleop.get_action()
    assert action["ee.wy"] == np.deg2rad(teleop.config.head_speed_deg)

    # Simulate 's' press (Pitch backward)
    teleop.current_pressed = {"s": True}
    action = teleop.get_action()
    assert action["ee.wy"] == 0.0 # Back to zero after w then s

def test_keyboard_reachy_mini_get_action_roll(teleop):
    # Simulate 'a' press (Roll left)
    teleop.current_pressed = {"a": True}
    action = teleop.get_action()
    assert action["ee.wx"] == np.deg2rad(teleop.config.head_speed_deg)

    # Simulate 'd' press (Roll right)
    teleop.current_pressed = {"d": True}
    action = teleop.get_action()
    assert action["ee.wx"] == 0.0

def test_keyboard_reachy_mini_get_action_yaw(teleop):
    # Simulate 'q' press (Yaw left)
    teleop.current_pressed = {"q": True}
    action = teleop.get_action()
    assert action["ee.wz"] == np.deg2rad(teleop.config.body_speed_deg)

    # Simulate 'e' press (Yaw right)
    teleop.current_pressed = {"e": True}
    action = teleop.get_action()
    assert action["ee.wz"] == 0.0

def test_keyboard_reachy_mini_get_action_z(teleop):
    # Simulate 'up' press
    teleop.current_pressed = {"up": True}
    action = teleop.get_action()
    assert action["ee.z"] == 0.001 * teleop.config.head_speed_deg

    # Simulate 'down' press
    teleop.current_pressed = {"down": True}
    action = teleop.get_action()
    assert action["ee.z"] == 0.0

def test_keyboard_reachy_mini_reset(teleop):
    teleop.action["ee.x"] = 1.0
    teleop.current_pressed = {"r": True}
    action = teleop.get_action()
    assert all(v == 0.0 for v in action.values())
