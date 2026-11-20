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

"""A policy reads one state key and predicts one action key; the rest are dropped.

`dataset_to_policy_features` types every non-visual `observation.*` column as STATE, but
`PreTrainedConfig.robot_state_feature` returns only the one named `observation.state`, and
`action_feature` only the one named `action`. The dropped keys leave no trace: they are
normalised, loaded and moved to the device every step, and they are written to the saved
`config.json` under `input_features`. `check_state` and `check_action` are what make that
audible.

The first test pins the behaviour the warnings describe. If the selectors ever start reading
more than one key, it fails, and the warnings become wrong rather than merely stale.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import lerobot.policies.factory as policy_factory
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_policy_config
from lerobot.policies.utils import check_action, check_state
from lerobot.utils.constants import ACTION, OBS_STATE

STATE = PolicyFeature(type=FeatureType.STATE, shape=(6,))
EFFORT = PolicyFeature(type=FeatureType.STATE, shape=(7,))
CAMERA = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8))
MOVE = PolicyFeature(type=FeatureType.ACTION, shape=(6,))


def _config(input_features, output_features=None):
    return make_policy_config(
        "act",
        input_features=dict(input_features),
        output_features=dict(output_features if output_features is not None else {ACTION: MOVE}),
        device="cpu",
    )


def test_only_the_state_key_named_observation_state_is_read():
    """The premise of both warnings. Not a restatement of the code: it asserts the
    *consequence*, on a config holding two STATE features and two ACTION features."""
    cfg = _config(
        {OBS_STATE: STATE, "observation.effort": EFFORT, "observation.images.cam": CAMERA},
        {ACTION: MOVE, "action.gripper": PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
    )

    assert cfg.robot_state_feature is STATE
    assert cfg.action_feature is MOVE
    # Order is irrelevant -- the selectors match on the name, not on position.
    reversed_cfg = _config({"observation.effort": EFFORT, OBS_STATE: STATE})
    assert reversed_cfg.robot_state_feature is STATE


def test_the_dropped_features_never_reach_the_model():
    """The strong form of the test above. Two STATE inputs and two ACTION outputs go in; the
    built policy's own layers are sized for one of each, so a dropped key has no entry into
    the model at all -- a narrow model, not merely a narrow selector."""
    cfg = _config(
        {OBS_STATE: STATE, "observation.effort": EFFORT, "observation.images.cam": CAMERA},
        {ACTION: MOVE, "action.gripper": PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
    )
    # As tests/training/test_ema.py does: no torchvision download for a shape check.
    cfg.pretrained_backbone_weights = None
    policy = ACTPolicy(cfg)

    # 6, not 6 + 7: the effort column is given no input width whatsoever.
    assert policy.model.encoder_robot_state_input_proj.in_features == STATE.shape[0]
    assert policy.model.action_head.out_features == MOVE.shape[0]


def test_an_extra_state_feature_is_reported(caplog):
    cfg = _config({OBS_STATE: STATE, "observation.effort": EFFORT})

    with caplog.at_level(logging.WARNING):
        check_state(cfg, dict(cfg.input_features))

    assert "observation.effort" in caplog.text
    assert OBS_STATE in caplog.text


def test_a_lone_observation_state_is_not_reported(caplog):
    cfg = _config({OBS_STATE: STATE, "observation.images.cam": CAMERA})

    with caplog.at_level(logging.WARNING):
        check_state(cfg, dict(cfg.input_features))

    assert caplog.text == ""


def test_a_vision_only_policy_is_not_reported(caplog):
    """No state features at all is a supported configuration, not a mistake."""
    cfg = _config({"observation.images.cam": CAMERA})

    with caplog.at_level(logging.WARNING):
        check_state(cfg, dict(cfg.input_features))

    assert caplog.text == ""


def test_a_misnamed_state_feature_is_reported_as_no_proprioception(caplog):
    """The expensive case: training converges, nothing fails, and the policy never
    knew where the arm was."""
    cfg = _config({"observation.states": STATE, "observation.images.cam": CAMERA})

    with caplog.at_level(logging.WARNING):
        check_state(cfg, dict(cfg.input_features))

    assert "without proprioception" in caplog.text
    assert "observation.states" in caplog.text
    assert "rename_map" in caplog.text


def test_an_extra_action_feature_is_reported(caplog):
    cfg = _config(
        {OBS_STATE: STATE},
        {ACTION: MOVE, "action.gripper": PolicyFeature(type=FeatureType.ACTION, shape=(1,))},
    )

    with caplog.at_level(logging.WARNING):
        check_action(cfg, dict(cfg.output_features))

    assert "action.gripper" in caplog.text


def test_a_lone_action_is_not_reported(caplog):
    cfg = _config({OBS_STATE: STATE}, {ACTION: MOVE})

    with caplog.at_level(logging.WARNING):
        check_action(cfg, dict(cfg.output_features))

    assert caplog.text == ""


def test_an_empty_config_is_not_reported(caplog):
    """`input_features` is unresolved until `make_policy` fills it in."""
    cfg = _config({}, {})

    with caplog.at_level(logging.WARNING):
        check_state(cfg, {})
        check_action(cfg, {})

    assert caplog.text == ""


@pytest.mark.parametrize("rename_map", [None, {"observation.current": "observation.effort"}])
def test_make_policy_reports_a_dataset_column_the_policy_will_drop(monkeypatch, caplog, rename_map):
    """With and without a rename map: the checks read the resolved config, so a renamed
    key is reported under the name the policy actually sees."""
    cfg = SimpleNamespace(
        type="mock",
        device="cpu",
        pretrained_path=None,
        use_peft=False,
        input_features={},
        output_features={},
    )
    ds_meta = SimpleNamespace(
        features={
            "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.current": {"dtype": "float32", "shape": (7,), "names": None},
            "action": {"dtype": "float32", "shape": (6,), "names": None},
        },
        stats={},
    )
    monkeypatch.setattr(
        policy_factory, "get_policy_class", lambda _: MagicMock(return_value=torch.nn.Linear(1, 1))
    )

    with caplog.at_level(logging.WARNING):
        policy_factory.make_policy(cfg, ds_meta=ds_meta, rename_map=rename_map)

    expected = "observation.effort" if rename_map else "observation.current"
    assert expected in caplog.text
    assert "ignored" in caplog.text
