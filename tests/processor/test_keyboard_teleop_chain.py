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

from lerobot.processor import make_default_processors


def test_default_processors_do_not_translate_keyboard_actions():
    teleop_action_processor, robot_action_processor, _ = make_default_processors()

    raw_keyboard_action = {"w": None, "a": None}
    obs = {"joint1.pos": 0.1, "joint2.pos": -0.2}

    teleop_action = teleop_action_processor((raw_keyboard_action, obs))
    robot_action = robot_action_processor((teleop_action, obs))

    assert teleop_action == raw_keyboard_action
    assert robot_action == raw_keyboard_action
    assert all(not key.endswith(".pos") for key in robot_action)
