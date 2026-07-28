# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json

import pytest

from lerobot.configs import PreTrainedConfig
from lerobot.policies.lingbot_vla_v2.checkpoint_lingbot_vla_v2 import (
    LINGBOT_VLA_V2_UPSTREAM_CONFIG_OVERRIDES,
    LINGBOT_VLA_V2_UPSTREAM_REPO_ID,
    apply_lingbot_vla_v2_upstream_config,
    is_raw_lingbot_vla_v2_checkpoint,
    remap_lingbot_vla_v2_upstream_key,
    validate_lingbot_vla_v2_upstream_loading_keys,
)
from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config


def test_raw_upstream_checkpoint_detection_accepts_repo_id():
    assert is_raw_lingbot_vla_v2_checkpoint(LINGBOT_VLA_V2_UPSTREAM_REPO_ID)


def test_raw_upstream_checkpoint_detection_accepts_local_sharded_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"vlm_family": "qwen3_vl"}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))

    assert is_raw_lingbot_vla_v2_checkpoint(tmp_path)


def test_raw_upstream_checkpoint_detection_rejects_lerobot_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "lingbot_vla_v2"}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))

    assert not is_raw_lingbot_vla_v2_checkpoint(tmp_path)


def test_pretrained_config_from_raw_upstream_checkpoint_uses_lingbot_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"vlm_family": "qwen3_vl"}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))

    config = PreTrainedConfig.from_pretrained(
        tmp_path,
        cli_overrides=[
            "--robot_config_path=robot.yaml",
            "--norm_stats_path=stats.json",
            "--device=cpu",
        ],
    )

    assert isinstance(config, LingbotVLAV2Config)
    assert config.robot_config_path == "robot.yaml"
    assert config.norm_stats_path == "stats.json"
    assert config.device == "cpu"
    assert config.moe_implementation == "fused"


def test_upstream_config_profile_matches_released_6b_architecture():
    config = LingbotVLAV2Config(
        token_moe_intermediate_size=1,
        token_shared_intermediate_size=1,
        router_activation="softmax",
        routed_scaling_factor=1.0,
        use_shared_expert_gate=True,
        moe_implementation=None,
    )

    apply_lingbot_vla_v2_upstream_config(config)

    for key, value in LINGBOT_VLA_V2_UPSTREAM_CONFIG_OVERRIDES.items():
        assert getattr(config, key) == value
    assert config._moe_implementation == "fused"


def test_upstream_key_remap_is_explicit_identity_for_released_6b():
    key = "model.qwenvl_with_expert.qwen_expert.model.layers.0.mlp.experts.down_proj"
    assert remap_lingbot_vla_v2_upstream_key(key) == key


def test_upstream_loading_validation_allows_disabled_distillation_heads():
    _, allowed = validate_lingbot_vla_v2_upstream_loading_keys(
        missing_keys=[],
        unexpected_keys=[
            "model.depth_align_embs",
            "model.future_video_align_head.projector.proj_out.weight",
            "model.current_shared_task_proj.weight",
        ],
        use_depth=False,
    )

    assert len(allowed) == 3


def test_upstream_loading_validation_rejects_unknown_unexpected_key():
    with pytest.raises(RuntimeError, match="unexpected non-whitelisted keys"):
        validate_lingbot_vla_v2_upstream_loading_keys(
            missing_keys=[],
            unexpected_keys=["model.qwenvl_with_expert.qwen_expert.bad.weight"],
            use_depth=False,
        )


def test_upstream_loading_validation_rejects_missing_key():
    with pytest.raises(RuntimeError, match="missing required keys"):
        validate_lingbot_vla_v2_upstream_loading_keys(
            missing_keys=["model.action_out_proj.weight"],
            unexpected_keys=[],
            use_depth=False,
        )
