#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import pytest
import torch

from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors


def test_fastwam_is_registered_in_policy_factory():
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

    cfg = make_policy_config("fastwam", action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2)

    assert isinstance(cfg, FastWAMConfig)
    assert cfg.type == "fastwam"
    assert get_policy_class("fastwam") is FastWAMPolicy


def test_fastwam_pre_post_processors_are_available():
    cfg = make_policy_config("fastwam", action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2)

    preprocessor, postprocessor = make_pre_post_processors(cfg)

    assert preprocessor.name == "policy_preprocessor"
    assert postprocessor.name == "policy_postprocessor"


def test_fastwam_postprocessor_only_adds_action_inversion_when_configured():
    from lerobot.policies.fastwam.processor_fastwam import (
        FastWAMActionInversionProcessorStep,
        FastWAMActionToggleProcessorStep,
    )

    default_cfg = make_policy_config(
        "fastwam", action_dim=3, proprio_dim=2, action_horizon=4, n_action_steps=2
    )
    _, default_postprocessor = make_pre_post_processors(default_cfg)

    assert any(isinstance(step, FastWAMActionToggleProcessorStep) for step in default_postprocessor.steps)
    assert not any(
        isinstance(step, FastWAMActionInversionProcessorStep) for step in default_postprocessor.steps
    )

    inverted_cfg = make_policy_config(
        "fastwam",
        action_dim=3,
        proprio_dim=2,
        action_horizon=4,
        n_action_steps=2,
        toggle_action_dimensions=[],
        invert_dimensions=[-1],
    )
    _, inverted_postprocessor = make_pre_post_processors(inverted_cfg)

    assert any(isinstance(step, FastWAMActionInversionProcessorStep) for step in inverted_postprocessor.steps)


def test_fastwam_action_inversion_processor_flips_configured_dimensions():
    from lerobot.policies.fastwam.processor_fastwam import FastWAMActionInversionProcessorStep

    processor = FastWAMActionInversionProcessorStep(invert_dimensions=[0, -1])
    action = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    processed = processor.action(action)

    assert torch.equal(processed, torch.tensor([[-1.0, 2.0, -3.0], [-4.0, 5.0, -6.0]]))
    assert torch.equal(action, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_fastwam_rejects_non_wan22_hub_model_ids():
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig

    with pytest.raises(ValueError, match="model_id"):
        FastWAMConfig(model_id="somebody/other-model")
