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

from unittest.mock import PropertyMock, patch

import pytest

from lerobot.teleoperators.keyboard import (
    KeyboardMicroDuckTeleop,
    KeyboardMicroDuckTeleopConfig,
)
from lerobot.utils.import_utils import _pynput_available

pytestmark = pytest.mark.skipif(not _pynput_available, reason="pynput is not installed")


@pytest.fixture
def teleop():
    teleop = KeyboardMicroDuckTeleop(KeyboardMicroDuckTeleopConfig())
    with patch.object(KeyboardMicroDuckTeleop, "is_connected", new_callable=PropertyMock, return_value=True):
        yield teleop


def press(teleop, *keys):
    for key in keys:
        teleop.event_queue.put((key, True))


def release(teleop, *keys):
    for key in keys:
        teleop.event_queue.put((key, False))


def test_idle_keyboard_commands_nothing(teleop):
    action = teleop.get_action()
    assert set(action) == set(teleop.action_features)
    assert all(value == 0.0 for value in action.values())


def test_held_keys_drive_the_twist(teleop):
    press(teleop, "w", "a", "x")
    action = teleop.get_action()
    assert action["vx"] == pytest.approx(0.3)
    assert action["vy"] == pytest.approx(0.3)
    assert action["vyaw"] == pytest.approx(-1.5)

    release(teleop, "w", "a", "x")
    assert teleop.get_action()["vx"] == 0.0


def test_speed_presets_step_once_per_press(teleop):
    press(teleop, "-")
    press(teleop, "-")  # auto-repeat of a held key must not step again
    press(teleop, "w")
    assert teleop.get_action()["vx"] == pytest.approx(0.3 * 0.66)


def test_head_keys_integrate_and_clamp(teleop):
    teleop.get_action()
    press(teleop, "j")
    teleop._last_t -= 0.05
    yaw = teleop.get_action()["head_yaw.pos"]
    assert 0.0 < yaw <= 0.8 * 0.1 + 1e-6

    for _ in range(50):
        teleop._last_t -= 0.1
        teleop.get_action()
    assert teleop.get_action()["head_yaw.pos"] == pytest.approx(0.8)

    press(teleop, "c")
    assert teleop.get_action()["head_yaw.pos"] == pytest.approx(0.0, abs=0.1)
