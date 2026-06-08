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
from lerobot.policies.lingbot_va.processor_lingbot_va import make_lingbot_va_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline, UnnormalizerProcessorStep
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    ACTION,
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


def test_make_pre_post_processors_names_and_steps() -> None:
    cfg = _make_config()
    pre, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats=None)
    assert pre.name == POLICY_PREPROCESSOR_DEFAULT_NAME
    assert post.name == POLICY_POSTPROCESSOR_DEFAULT_NAME
    # Actions are unnormalized by the standard built-in quantile unnormalizer.
    assert any(isinstance(s, UnnormalizerProcessorStep) for s in post.steps)


def test_freshly_built_postprocessor_is_identity() -> None:
    # Without action stats the quantile unnormalizer is a no-op (identity passthrough): the real
    # per-benchmark q01/q99 are restored from the saved checkpoint on load, not hardcoded here.
    cfg = _make_config()
    _, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats=None)
    normed = torch.tensor([[0.3, -0.5, 1.0, -1.0, 0.0, 0.7, -0.2]])
    assert torch.allclose(post(normed), normed, atol=1e-6)


def test_postprocessor_quantile_unnormalization() -> None:
    # QUANTILES unnormalize maps [-1, 1] -> [q01, q99]: -1 -> q01, +1 -> q99.
    cfg = _make_config()
    q01 = [-1.0, -0.5, 0.0, -1.0, -1.0, -1.0, -1.0]
    q99 = [1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0]
    stats = {ACTION: {"q01": q01, "q99": q99}}
    _, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats=stats)
    out_lo = post(torch.full((1, 7), -1.0))
    out_hi = post(torch.full((1, 7), 1.0))
    assert torch.allclose(out_lo, torch.tensor(q01).unsqueeze(0), atol=1e-4)
    assert torch.allclose(out_hi, torch.tensor(q99).unsqueeze(0), atol=1e-4)


def test_postprocessor_stats_survive_save_load(tmp_path) -> None:
    # Regression guard for the Hub mechanism: the q01/q99 stats live in the saved post-processor
    # state and must round-trip through save_pretrained / from_pretrained.
    cfg = _make_config()
    q01 = [-0.6, -0.8, -0.9, -0.1, -0.15, -0.25, -1.0]
    q99 = [0.9, 0.85, 0.9, 0.17, 0.18, 0.34, 1.0]
    _, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats={ACTION: {"q01": q01, "q99": q99}})
    post.save_pretrained(tmp_path)
    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    out = loaded(torch.full((1, 7), -1.0))
    assert torch.allclose(out, torch.tensor(q01).unsqueeze(0), atol=1e-4)
