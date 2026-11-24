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

from dataclasses import dataclass, field, replace

import draccus
import pytest

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.processor.hold_joints_processor import HoldUncommandedJointsProcessorStep
from lerobot.processor.keyboard_base_processor import (
    KeyboardToBaseActionProcessorStep,
    default_base_key_bindings,
)


def make_pipeline(*steps):
    return RobotProcessorPipeline(
        steps=list(steps),
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


@pytest.mark.parametrize(
    "keys,expected",
    [
        ("w", (0.1, 0.0, 0.0)),
        ("s", (-0.1, 0.0, 0.0)),
        ("a", (0.0, 0.1, 0.0)),
        ("d", (0.0, -0.1, 0.0)),
        ("z", (0.0, 0.0, 30.0)),
        ("x", (0.0, 0.0, -30.0)),
        ("wsadzx", (0.0, 0.0, 0.0)),
        ("", (0.0, 0.0, 0.0)),
    ],
)
def test_mapping_directions(keys, expected):
    action = KeyboardToBaseActionProcessorStep().action(dict.fromkeys(keys))
    assert (action["x.vel"], action["y.vel"], action["theta.vel"]) == expected


def test_speed_changes_once_per_press_is_bounded_and_resets():
    step = KeyboardToBaseActionProcessorStep()
    for _ in range(3):  # Holding "+" only counts once.
        assert step.action({"w": None, "+": None})["x.vel"] == 0.2
    for _ in range(2):
        step.action({})
        assert step.action({"w": None, "+": None})["x.vel"] == 0.3
    step.reset()
    assert step.action({"w": None})["x.vel"] == 0.1


@pytest.mark.parametrize("kwargs", [{"angular_speeds": (1.0,)}, {"key_bindings": {}}, {"layout": "unknown"}])
def test_invalid_mapping_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        KeyboardToBaseActionProcessorStep(**kwargs)


def test_holding_latches_positions_until_reset():
    pipeline = make_pipeline(HoldUncommandedJointsProcessorStep(("joint.pos",)))
    assert pipeline(({}, {"joint.pos": 10.0})) == {"joint.pos": 10.0}
    assert pipeline(({}, {"joint.pos": 8.0})) == {"joint.pos": 10.0}
    assert pipeline(({"joint.pos": 20.0}, {"joint.pos": 8.0})) == {"joint.pos": 20.0}
    pipeline.reset()
    assert pipeline(({}, {"joint.pos": 30.0})) == {"joint.pos": 30.0}


def test_features_and_serialization_preserve_arm_and_held_channels(tmp_path):
    pipeline = make_pipeline(
        KeyboardToBaseActionProcessorStep(
            layout="azerty", key_bindings={**default_base_key_bindings("azerty"), "forward": "i"}
        ),
        HoldUncommandedJointsProcessorStep(("arm_joint.pos", "gripper.pos")),
    )
    feature = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
    features = pipeline.transform_features({PipelineFeatureType.ACTION: {"arm_joint.pos": feature}})
    # Existing action features keep their position in the dataset layout.
    assert list(features[PipelineFeatureType.ACTION]) == [
        "arm_joint.pos",
        "x.vel",
        "y.vel",
        "theta.vel",
        "gripper.pos",
    ]
    pipeline.save_pretrained(tmp_path, config_filename="processor.json")
    restored = RobotProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename="processor.json",
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    sample = ({"i": None, "arm_joint.pos": 4.0}, {"gripper.pos": 8.0})
    expected = {"arm_joint.pos": 4.0, "x.vel": 0.1, "y.vel": 0.0, "theta.vel": 0.0, "gripper.pos": 8.0}
    assert restored(sample) == pipeline(sample) == expected


@pytest.mark.parametrize("layout,keys", [("qwerty", "waz"), ("azerty", "zqw"), ("qwertz", "way")])
def test_layout_cli_mapping_and_session_copy(layout, keys):
    @dataclass
    class Config:
        base_mapping: KeyboardToBaseActionProcessorStep = field(
            default_factory=KeyboardToBaseActionProcessorStep
        )

    config = draccus.parse(Config, args=[f"--base_mapping.layout={layout}"])
    mapping = replace(config.base_mapping)
    action = mapping.action({**dict.fromkeys(keys), "arm_joint.pos": 12.0})
    assert action == {"x.vel": 0.1, "y.vel": 0.1, "theta.vel": 30.0, "arm_joint.pos": 12.0}
    assert KeyboardToBaseActionProcessorStep(**mapping.get_config()).action(dict.fromkeys(keys)) == {
        "x.vel": 0.1,
        "y.vel": 0.1,
        "theta.vel": 30.0,
    }
