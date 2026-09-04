# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the raw-upstream -> LeRobot checkpoint conversion helpers."""

import json

import pytest

# Importing the config pulls in the policy package __init__, which imports the
# modeling module and its heavy deps (transformers, einops, ...). Skip on CI
# tiers where those optional extras are not installed.
pytest.importorskip("transformers")

from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config
from lerobot.policies.lingbot_vla_v2.scripts.convert_upstream_checkpoint import (
    ALLOWED_SKIPPED_PREFIXES,
    DEFAULT_ALIGN_PARAMS_FILE,
    UPSTREAM_CONFIG_OVERRIDES,
    split_upstream_loading_keys,
)


def test_upstream_config_overrides_match_released_6b_architecture():
    config = LingbotVLAV2Config(
        token_moe_intermediate_size=1,
        token_shared_intermediate_size=1,
        router_activation="softmax",
        routed_scaling_factor=1.0,
        use_shared_expert_gate=True,
        moe_implementation=None,
    )

    for key, value in UPSTREAM_CONFIG_OVERRIDES.items():
        setattr(config, key, value)
    config._moe_implementation = config.moe_implementation

    for key, value in UPSTREAM_CONFIG_OVERRIDES.items():
        assert getattr(config, key) == value
    assert config._moe_implementation == "fused"


def test_bundled_align_params_recipe_passes_config_validation():
    """The bundled RoboTwin recipe must survive LingbotVLAV2Config construction —
    it is what --include-depth-heads feeds the converter."""
    params = json.loads(DEFAULT_ALIGN_PARAMS_FILE.read_text())
    params.pop("_comment", None)
    config = LingbotVLAV2Config(align_params=params)
    assert config.use_depth_align is True
    assert config.use_future_image is True


def test_split_keys_allows_disabled_distillation_heads():
    missing, skipped, unexpected = split_upstream_loading_keys(
        model_keys={"model.action_out_proj.weight"},
        checkpoint_keys={
            "model.action_out_proj.weight",
            "model.depth_align_embs",
            "model.future_video_align_head.projector.proj_out.weight",
            "model.current_shared_task_proj.weight",
        },
    )

    assert missing == []
    assert len(skipped) == 3
    assert unexpected == []
    assert all(any(k.startswith(p) for p in ALLOWED_SKIPPED_PREFIXES) for k in skipped)


def test_split_keys_covers_future_video_cls_prefix():
    """A future checkpoint with use_cls_loss=true must not hard-fail action-only conversion."""
    _, skipped, unexpected = split_upstream_loading_keys(
        model_keys=set(),
        checkpoint_keys={"model.future_video_cls_align_head.projector.proj_out.weight"},
    )
    assert skipped and not unexpected


def test_split_keys_strict_when_depth_heads_included():
    """--include-depth-heads: heads are in the model, so nothing may be skipped and
    checkpoint-only tensors fail instead of being silently dropped."""
    missing, skipped, unexpected = split_upstream_loading_keys(
        model_keys={"model.action_out_proj.weight", "model.depth_align_head.projector.proj_out.weight"},
        checkpoint_keys={
            "model.action_out_proj.weight",
            "model.depth_align_head.projector.proj_out.weight",
        },
        allow_skipped=False,
    )
    assert missing == []
    assert skipped == []
    assert unexpected == []

    # Checkpoint carries a distillation tensor the model did not build (e.g. the
    # align_params omitted a head): must be hard-unexpected, not allowed-skipped.
    _, skipped, unexpected = split_upstream_loading_keys(
        model_keys={"model.action_out_proj.weight"},
        checkpoint_keys={"model.action_out_proj.weight", "model.depth_align_embs"},
        allow_skipped=False,
    )
    assert skipped == []
    assert unexpected == ["model.depth_align_embs"]

    # Checkpoint lacks a head the model built: missing, i.e. no random-init heads.
    missing, _, _ = split_upstream_loading_keys(
        model_keys={"model.depth_align_head.projector.proj_out.weight"},
        checkpoint_keys=set(),
        allow_skipped=False,
    )
    assert missing == ["model.depth_align_head.projector.proj_out.weight"]


def test_split_keys_flags_unknown_unexpected_key():
    missing, skipped, unexpected = split_upstream_loading_keys(
        model_keys=set(),
        checkpoint_keys={"model.qwenvl_with_expert.qwen_expert.bad.weight"},
    )

    assert missing == []
    assert skipped == []
    assert unexpected == ["model.qwenvl_with_expert.qwen_expert.bad.weight"]


def test_split_keys_flags_missing_key():
    missing, skipped, unexpected = split_upstream_loading_keys(
        model_keys={"model.action_out_proj.weight"},
        checkpoint_keys=set(),
    )

    assert missing == ["model.action_out_proj.weight"]
    assert skipped == []
    assert unexpected == []
