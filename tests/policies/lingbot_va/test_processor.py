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

from __future__ import annotations

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.processor_lingbot_va import (
    LingBotVAActionUnnormalizeStep,
    make_lingbot_va_pre_post_processors,
)
from lerobot.utils.constants import (
    OBS_IMAGES,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def _make_config() -> LingBotVAConfig:
    cfg = LingBotVAConfig(device="cpu")
    cfg.input_features = {f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))}
    cfg.output_features = {}
    cfg.validate_features()
    return cfg


def test_action_unnormalize_inverts_quantile_norm() -> None:
    q01 = [-1.0, -0.5, 0.0]
    q99 = [1.0, 0.5, 2.0]
    step = LingBotVAActionUnnormalizeStep(action_q01=q01, action_q99=q99)

    # Forward (the policy-side) quantile normalization: (x - q01) / (q99 - q01 + eps) * 2 - 1.
    q01_t = torch.tensor(q01)
    q99_t = torch.tensor(q99)
    raw = torch.tensor([[0.3, 0.1, 1.0]])
    normed = (raw - q01_t) / (q99_t - q01_t + 1e-6) * 2.0 - 1.0

    recovered = step.action(normed)
    assert torch.allclose(recovered, raw, atol=1e-4)


def test_action_unnormalize_config_roundtrip() -> None:
    step = LingBotVAActionUnnormalizeStep(action_q01=[0.0, 1.0], action_q99=[2.0, 3.0])
    cfg = step.get_config()
    assert cfg == {"action_q01": [0.0, 1.0], "action_q99": [2.0, 3.0]}
    rebuilt = LingBotVAActionUnnormalizeStep(**cfg)
    assert rebuilt.action_q01 == step.action_q01
    assert rebuilt.action_q99 == step.action_q99


def test_make_pre_post_processors_names_and_steps() -> None:
    cfg = _make_config()
    pre, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats=None)
    assert pre.name == POLICY_PREPROCESSOR_DEFAULT_NAME
    assert post.name == POLICY_POSTPROCESSOR_DEFAULT_NAME
    # The postprocessor must contain the dedicated quantile unnormalize step.
    assert any(isinstance(s, LingBotVAActionUnnormalizeStep) for s in post.steps)


def test_postprocessor_applies_unnormalization() -> None:
    cfg = _make_config()
    _, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats=None)
    # A normalized action of all -1 should map back to q01.
    normed = torch.full((1, len(cfg.used_action_channel_ids)), -1.0)
    out = post(normed)
    assert torch.allclose(out, torch.tensor(cfg.action_q01).unsqueeze(0), atol=1e-4)
