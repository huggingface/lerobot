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

"""Config tests for the Faster-WAM (arXiv 2608.02365) options on FastWAMConfig.

Covers serialization round-trips, option validation, module construction effects, and
checkpoint compatibility (a default config must keep the exact legacy state-dict key
layout so existing Fast-WAM checkpoints load strictly).
"""

import pytest

pytest.importorskip("transformers", reason="fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="fastwam requires the `fastwam` extra (diffusers)")

from lerobot.configs import PreTrainedConfig
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.wan import ActionDiT, LiteKVFusion


def _fasterwam_config(**overrides) -> FastWAMConfig:
    fields = {
        "action_expert_layers": 1,
        "lite_kv_fusion": True,
        "retracted_1d_rope": True,
        "world_model_only_conditioning": True,
    }
    fields.update(overrides)
    return FastWAMConfig(**fields)


def test_default_config_preserves_fastwam_behavior():
    config = FastWAMConfig()
    assert config.action_expert_layers is None
    assert not config.lite_kv_fusion
    assert not config.retracted_1d_rope
    assert not config.world_model_only_conditioning
    assert config.action_dit_config["num_layers"] == 30
    assert (
        "use_text_cross_attention" not in config.action_dit_config
        or config.action_dit_config["use_text_cross_attention"]
    )


def test_fasterwam_config_serialization_round_trip(tmp_path):
    config = _fasterwam_config()
    config.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path, policy_class="fastwam")
    assert isinstance(loaded, FastWAMConfig)
    assert loaded.action_expert_layers == 1
    assert loaded.lite_kv_fusion
    assert loaded.retracted_1d_rope
    assert loaded.world_model_only_conditioning
    assert loaded.action_dit_config["num_layers"] == 1
    assert loaded.action_dit_config["use_text_cross_attention"] is False


def test_shallow_expert_requires_lite_kv_fusion():
    with pytest.raises(ValueError, match="lite_kv_fusion"):
        FastWAMConfig(action_expert_layers=1)


def test_action_expert_layers_cannot_exceed_video_depth():
    with pytest.raises(ValueError, match="cannot exceed"):
        FastWAMConfig(action_expert_layers=31, lite_kv_fusion=True)


def test_retracted_rope_requires_lite_kv_fusion():
    with pytest.raises(ValueError, match="retracted_1d_rope"):
        FastWAMConfig(retracted_1d_rope=True)


def test_action_expert_layers_override_num_layers():
    config = _fasterwam_config(action_dit_config={"num_layers": 30})
    assert config.action_dit_config["num_layers"] == 1


def test_one_layer_action_expert_builds_single_block_without_cross_attention():
    config = _fasterwam_config()
    action_expert = ActionDiT(**config.action_dit_config)
    assert len(action_expert.blocks) == 1
    assert not action_expert.use_text_cross_attention
    assert action_expert.text_embedding is None
    # World-model-only conditioning: no cross-attention tensors anywhere in the expert.
    assert not any("cross_attn" in name or "text_embedding" in name for name in action_expert.state_dict())

    default_expert = ActionDiT(**FastWAMConfig().action_dit_config)
    assert default_expert.use_text_cross_attention
    assert default_expert.text_embedding is not None


def test_lite_kv_fusion_module_matches_config():
    config = _fasterwam_config()
    fusion = LiteKVFusion(
        num_action_layers=config.action_dit_config["num_layers"],
        num_heads=config.action_dit_config["num_heads"],
        num_video_layers=config.video_dit_config["num_layers"],
    )
    assert fusion.logits.shape == (1, config.action_dit_config["num_heads"], 30)
    assert sum(p.numel() for p in fusion.parameters()) == 720


def test_default_config_state_dict_layout_is_unchanged():
    """Checkpoint compatibility: with all Faster-WAM options off, the MoT must not gain
    any new parameter keys (existing Fast-WAM checkpoints load strictly), and the action
    expert keys must stay under mot.layers.*.blocks.action.*."""
    from lerobot.policies.fastwam.wan import MoT

    expert_kwargs = {
        "hidden_dim": 8,
        "action_dim": 3,
        "ffn_dim": 16,
        "text_dim": 8,
        "freq_dim": 8,
        "eps": 1e-6,
        "num_heads": 2,
        "attn_head_dim": 12,
        "fp32_attention": False,
    }
    mot = MoT(
        mixtures={
            "video": ActionDiT(**expert_kwargs, num_layers=2),
            "action": ActionDiT(**expert_kwargs, num_layers=2),
        },
        mot_checkpoint_mixed_attn=False,
    )
    keys = set(mot.state_dict().keys())
    assert not any("lite_kv_fusion" in key for key in keys)
    assert "layers.0.blocks.action.self_attn.q.weight" in keys
    assert "layers.1.blocks.action.self_attn.q.weight" in keys

    dot_mot = MoT(
        mixtures={
            "video": ActionDiT(**expert_kwargs, num_layers=2),
            "action": ActionDiT(**expert_kwargs, num_layers=1),
        },
        mot_checkpoint_mixed_attn=False,
        lite_kv_fusion=LiteKVFusion(1, 2, 2),
    )
    dot_keys = set(dot_mot.state_dict().keys())
    assert "lite_kv_fusion.logits" in dot_keys
    # The single action block lives in layer 0 only; deeper layers own video only.
    assert "layers.0.blocks.action.self_attn.q.weight" in dot_keys
    assert not any(key.startswith("layers.1.blocks.action") for key in dot_keys)
