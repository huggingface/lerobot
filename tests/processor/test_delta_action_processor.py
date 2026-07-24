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

"""Tests for MapDeltaActionToRobotActionStep's optional orientation (delta_pitch/delta_roll) support."""

from lerobot.processor.delta_action_processor import MapDeltaActionToRobotActionStep


def _base_action(**overrides):
    action = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1}
    action.update(overrides)
    return action


def test_enabled_gate_fires_on_rotation_only():
    """A pure rotation command (no position delta) must still enable the robot.

    Regression test: previously `enabled` was computed from position magnitude
    alone, so a keyboard/gamepad orientation-only command (e.g. pitch with no
    x/y/z) was silently dropped downstream.
    """
    step = MapDeltaActionToRobotActionStep(rotation_scale=0.5)

    result = step.action(_base_action(delta_pitch=1.0))
    assert result["enabled"] is True

    result = step.action(_base_action(delta_roll=1.0))
    assert result["enabled"] is True


def test_enabled_gate_false_when_everything_below_noise_threshold():
    step = MapDeltaActionToRobotActionStep(noise_threshold=1e-3)

    result = step.action(_base_action())
    assert result["enabled"] is False


def test_rotation_scale_applied_to_pitch_and_roll():
    step = MapDeltaActionToRobotActionStep(rotation_scale=0.25)

    result = step.action(_base_action(delta_pitch=2.0, delta_roll=-4.0))

    assert result["target_wy"] == 0.5  # delta_pitch * rotation_scale
    assert result["target_wz"] == -1.0  # delta_roll * rotation_scale
    assert result["target_wx"] == 0.0  # no roll-axis input is exposed for wx


def test_orientation_keys_are_optional_for_backward_compatibility():
    """Devices that never send delta_pitch/delta_roll (e.g. the phone teleoperator)
    must keep working unchanged."""
    step = MapDeltaActionToRobotActionStep()

    result = step.action(_base_action(delta_x=1.0))

    assert result["target_wy"] == 0.0
    assert result["target_wz"] == 0.0
    assert result["enabled"] is True
