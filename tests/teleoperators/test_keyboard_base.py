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

"""Keyboard mapping, configuration, and pipeline persistence without hardware."""

from dataclasses import dataclass
from unittest.mock import PropertyMock, patch

import draccus
import pytest

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.processor.keyboard_base_processor import (
    KeyboardToBaseActionProcessorStep,
    default_base_key_bindings,
    make_keyboard_base_processor,
)
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.keyboard import KeyboardBaseTeleop, KeyboardBaseTeleopConfig, KeyboardTeleop
from lerobot.teleoperators.utils import make_teleoperator_from_config


@dataclass
class _Config:
    teleop: TeleoperatorConfig


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


def test_config_and_factory_support_custom_bindings_and_speeds():
    config = draccus.parse(
        _Config,
        args=[
            "--teleop.type=keyboard_base",
            '--teleop.key_bindings={"forward":"i","backward":"k","left":"j","right":"l","rotate_left":"u","rotate_right":"o","speed_up":"+","speed_down":"-"}',
            "--teleop.linear_speeds=[0.4,0.8]",
            "--teleop.angular_speeds=[10,20]",
        ],
    )
    with patch("lerobot.teleoperators.keyboard.teleop_keyboard.require_package"):
        teleop = make_teleoperator_from_config(config.teleop)
    assert isinstance(teleop, KeyboardBaseTeleop)
    with patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=True):
        teleop.event_queue.put(("i", True))
        teleop.event_queue.put(("u", True))
        assert teleop.get_action() == {"x.vel": 0.4, "y.vel": 0.0, "theta.vel": 10.0}
        teleop.event_queue.put(("i", False))
        assert teleop.get_action()["x.vel"] == 0.0
    assert set(teleop.action_features) == {"x.vel", "y.vel", "theta.vel"}


def test_reconnect_resets_speed_and_stale_keys():
    with patch("lerobot.teleoperators.keyboard.teleop_keyboard.require_package"):
        teleop = KeyboardBaseTeleop(KeyboardBaseTeleopConfig())
    with patch.object(KeyboardTeleop, "is_connected", new_callable=PropertyMock, return_value=True):
        teleop.event_queue.put(("w", True))
        teleop.event_queue.put(("+", True))
        assert teleop.get_action()["x.vel"] == 0.2
        teleop.event_queue.put(("w", True))
        with patch.object(KeyboardTeleop, "connect"):
            teleop.connect()
        assert teleop.get_action() == {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
        teleop.event_queue.put(("w", True))
        assert teleop.get_action()["x.vel"] == 0.1


def test_pipeline_roundtrip_and_feature_mapping(tmp_path):
    keys = {**default_base_key_bindings(), "forward": "i"}
    pipeline = make_keyboard_base_processor(KeyboardToBaseActionProcessorStep(key_bindings=keys))
    raw = {"i": None}
    obs = {"arm_shoulder.pos": 10.0}
    expected = {"x.vel": 0.1, "y.vel": 0.0, "theta.vel": 0.0}
    assert pipeline((raw, obs)) == expected
    assert raw == {"i": None} and obs == {"arm_shoulder.pos": 10.0}
    pipeline.save_pretrained(tmp_path, config_filename="processor.json")
    restored = RobotProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="processor.json",
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    assert restored((raw, obs)) == expected
    features = {
        PipelineFeatureType.ACTION: {},
        PipelineFeatureType.OBSERVATION: {
            "arm_shoulder.pos": PolicyFeature(type=FeatureType.STATE, shape=(1,))
        },
    }
    output = pipeline.steps[0].transform_features(features)
    assert set(output[PipelineFeatureType.ACTION]) == set(expected)
    assert output[PipelineFeatureType.OBSERVATION] == features[PipelineFeatureType.OBSERVATION]
    assert features[PipelineFeatureType.ACTION] == {}
