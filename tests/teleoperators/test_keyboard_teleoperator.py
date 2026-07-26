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

"""Tests for KeyboardTeleop's key-state tracking. Meant to be run where pynput is installed."""

import pytest

from lerobot.utils.import_utils import _pynput_available

if not _pynput_available:
    pytest.skip("pynput not available", allow_module_level=True)

from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop


def _make_keyboard_teleop() -> KeyboardTeleop:
    return KeyboardTeleop(KeyboardTeleopConfig())


def test_released_key_is_removed_not_stored_as_false():
    """Regression test: `_drain_pressed_keys` used to store released keys as
    `current_pressed[key] = False` forever. Subclasses (e.g.
    KeyboardEndEffectorTeleop.get_action) iterate current_pressed and assign
    per-axis deltas by overwriting as they go, so a stale False entry from a
    released key kept overwriting the value of the opposite (still pressed)
    key on the same axis.
    """
    teleop = _make_keyboard_teleop()

    teleop.event_queue.put(("a", True))
    teleop._drain_pressed_keys()
    assert teleop.current_pressed == {"a": True}

    teleop.event_queue.put(("a", False))
    teleop._drain_pressed_keys()
    assert teleop.current_pressed == {}


def test_tap_then_opposite_key_is_not_shadowed():
    """Reproduces the exact ratchet bug: tap one key (press+release), then
    press the logically opposite key - the second key must be the only
    entry left in current_pressed, not masked by a stale False.
    """
    teleop = _make_keyboard_teleop()

    # Tap "down" once.
    teleop.event_queue.put(("down", True))
    teleop._drain_pressed_keys()
    teleop.event_queue.put(("down", False))
    teleop._drain_pressed_keys()

    # Now press and hold "up".
    teleop.event_queue.put(("up", True))
    teleop._drain_pressed_keys()

    assert teleop.current_pressed == {"up": True}
