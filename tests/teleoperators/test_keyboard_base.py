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

"""Keyboard mapping and configuration without hardware."""

from dataclasses import dataclass, field

import draccus
import pytest

from lerobot.processor.keyboard_base_processor import KeyboardToBaseActionProcessorStep


@dataclass
class _Config:
    mapping: KeyboardToBaseActionProcessorStep = field(default_factory=KeyboardToBaseActionProcessorStep)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("w", {"x.vel": 0.1, "y.vel": 0.0, "theta.vel": 0.0}),
        ("s", {"x.vel": -0.1, "y.vel": 0.0, "theta.vel": 0.0}),
        ("a", {"x.vel": 0.0, "y.vel": 0.1, "theta.vel": 0.0}),
        ("d", {"x.vel": 0.0, "y.vel": -0.1, "theta.vel": 0.0}),
        ("z", {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 30.0}),
        ("x", {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": -30.0}),
    ],
)
def test_mapping_directions(key, expected):
    step = KeyboardToBaseActionProcessorStep()
    assert step.action({key: None}) == expected


def test_opposite_keys_cancel_and_empty_input_stops():
    step = KeyboardToBaseActionProcessorStep()
    stopped = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    assert step.action(dict.fromkeys("wsadzx")) == stopped
    assert step.action({}) == stopped


def test_speed_changes_once_per_press_is_bounded_and_resets():
    step = KeyboardToBaseActionProcessorStep()
    for _ in range(3):
        assert step.action({"w": None, "+": None})["x.vel"] == 0.2
    step.action({})
    assert step.action({"w": None, "+": None})["x.vel"] == 0.3
    for _ in range(3):
        step.action({})
        assert step.action({"w": None, "+": None})["x.vel"] == 0.3
    for _ in range(4):
        step.action({})
        step.action({"-": None})
    assert step.action({"w": None})["x.vel"] == 0.1
    step.action({"+": None})
    step.reset()
    assert step.action({"w": None})["x.vel"] == 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"linear_speeds": ()},
        {"angular_speeds": (1.0,)},
        {"linear_speeds": (-1, 2, 3)},
        {"linear_speeds": (float("nan"), 2, 3)},
        {"angular_speeds": (1, 2, float("inf"))},
        {"key_bindings": {}},
    ],
)
def test_invalid_mapping_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        KeyboardToBaseActionProcessorStep(**kwargs)


def test_processor_config_supports_custom_speeds():
    config = draccus.parse(
        _Config, args=["--mapping.linear_speeds=[0.4,0.8]", "--mapping.angular_speeds=[10,20]"]
    )
    assert config.mapping.action({"w": None})["x.vel"] == 0.4
    assert config.mapping.action({"w": None, "+": None})["x.vel"] == 0.8
