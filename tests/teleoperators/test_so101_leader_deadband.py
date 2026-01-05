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

from lerobot.teleoperators.so101_leader.so101_leader import _apply_action_deadband


def test_deadband_no_last_action_noop():
    action = {"shoulder_pan.pos": 10.0}
    out = _apply_action_deadband(action, None, 1.0)
    assert out == action


def test_deadband_float_holds_small_changes():
    last = {"shoulder_pan.pos": 10.0}
    action = {"shoulder_pan.pos": 10.5}
    out = _apply_action_deadband(action, last, 1.0)
    assert out["shoulder_pan.pos"] == 10.0


def test_deadband_float_allows_large_changes():
    last = {"shoulder_pan.pos": 10.0}
    action = {"shoulder_pan.pos": 11.1}
    out = _apply_action_deadband(action, last, 1.0)
    assert out["shoulder_pan.pos"] == 11.1


def test_deadband_dict_filters_only_specified_keys():
    last = {"shoulder_pan.pos": 10.0, "gripper.pos": 50.0}
    action = {"shoulder_pan.pos": 10.5, "gripper.pos": 50.1}

    # Only filter shoulder_pan; gripper should pass through unfiltered.
    out = _apply_action_deadband(action, last, {"shoulder_pan.pos": 1.0})
    assert out["shoulder_pan.pos"] == 10.0
    assert out["gripper.pos"] == 50.1


