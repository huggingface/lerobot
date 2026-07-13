#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

import hashlib

import pytest
import torch
from transformers import StaticCache
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig as TransformersQwen2_5_VLConfig,
    Qwen2_5_VLTextConfig as TransformersQwen2_5_VLTextConfig,
)

from lerobot.policies.wall_x.configuration_wall_x import WallXConfig
from lerobot.policies.wall_x.qwen_model.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from lerobot.policies.wall_x.qwen_model.qwen2_5_vl_moe import Qwen2_5_VLMoEModel


def make_tiny_config(**overrides) -> Qwen2_5_VLConfig:
    kwargs = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "rope_scaling": {"type": "mrope", "mrope_section": [2, 1, 1]},
        "num_experts": 2,
        "dim_inputs": [16, 16],
        "mlp_moe": False,
    }
    kwargs.update(overrides)
    return Qwen2_5_VLConfig(**kwargs)


@pytest.fixture
def tiny_model() -> Qwen2_5_VLMoEModel:
    torch.manual_seed(0)
    return Qwen2_5_VLMoEModel(make_tiny_config()).eval()


def test_native_config_accepts_legacy_flat_layout_and_defines_pad_token_id():
    config = make_tiny_config()

    assert isinstance(config, TransformersQwen2_5_VLConfig)
    assert isinstance(config.text_config, TransformersQwen2_5_VLTextConfig)
    assert config.pad_token_id is None
    assert config.hidden_size == 16
    assert config.num_experts == 2
    assert config.dim_inputs == (16, 16)

    serialized = config.to_dict()
    assert serialized["text_config"]["hidden_size"] == 16
    assert serialized["text_config"]["num_experts"] == 2

    legacy_flat_config = {
        "model_type": "qwen2_5_vl",
        "vocab_size": 48,
        "hidden_size": 24,
        "intermediate_size": 48,
        "num_hidden_layers": 1,
        "num_attention_heads": 3,
        "num_key_value_heads": 3,
        "rope_scaling": {"type": "mrope", "mrope_section": [2, 1, 1]},
        "num_experts": 3,
        "dim_inputs": [24, 24, 24],
        "mlp_moe": False,
    }
    restored = Qwen2_5_VLConfig.from_dict(legacy_flat_config)
    assert restored.hidden_size == 24
    assert restored.num_experts == 3
    assert restored.dim_inputs == (24, 24, 24)
    assert hasattr(restored.text_config, "pad_token_id")


def test_tiny_model_state_dict_schema_is_stable(tiny_model):
    schema = "\n".join(
        f"{key}:{tuple(value.shape)}" for key, value in sorted(tiny_model.state_dict().items())
    )

    assert len(tiny_model.state_dict()) == 26
    assert hashlib.sha256(schema.encode()).hexdigest() == (
        "1038565029879e42a02485814bbf3f86e4e6e1a8117851f32e68e05c9d737cf0"
    )


def test_action_tokens_form_a_bidirectional_island(tiny_model):
    captured = {}

    def capture_attention_inputs(module, args, kwargs):
        captured["attention_mask"] = kwargs["attention_mask"].detach()

    handle = tiny_model.layers[0].self_attn.register_forward_pre_hook(
        capture_attention_inputs, with_kwargs=True
    )
    try:
        tiny_model(
            input_ids=torch.tensor([[2, 3, 4, 5]]),
            moe_token_types=torch.tensor([[0, 1, 1, 0]]),
            use_cache=False,
        )
    finally:
        handle.remove()

    allowed = captured["attention_mask"][0, 0] == 0
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, True, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    torch.testing.assert_close(allowed, expected)


def test_packed_text_position_ids_reach_attention_and_bound_action_islands(tiny_model):
    captured = {}

    def capture_attention_inputs(module, args, kwargs):
        captured["attention_mask"] = kwargs["attention_mask"].detach()
        captured["position_ids"] = kwargs["position_ids"].detach()

    handle = tiny_model.layers[0].self_attn.register_forward_pre_hook(
        capture_attention_inputs, with_kwargs=True
    )
    text_position_ids = torch.tensor([[0, 1, 0, 1]])
    multimodal_position_ids = torch.stack([text_position_ids] * 3)
    packed_position_ids = torch.cat([text_position_ids.unsqueeze(0), multimodal_position_ids], dim=0)
    try:
        tiny_model(
            input_ids=torch.tensor([[2, 3, 4, 5]]),
            position_ids=packed_position_ids,
            moe_token_types=torch.tensor([[0, 1, 1, 0]]),
            use_cache=False,
        )
    finally:
        handle.remove()

    torch.testing.assert_close(captured["position_ids"], text_position_ids)
    # Positions 1 and 2 are action tokens, but they belong to different packed
    # sequences and therefore must not be connected by the action overlay.
    assert captured["attention_mask"][0, 0, 1, 2] < 0
    assert captured["attention_mask"][0, 0, 2, 1] < 0


def test_native_output_capture_and_return_dict_behavior(tiny_model):
    model_inputs = {
        "input_ids": torch.tensor([[2, 3, 4, 5]]),
        "moe_token_types": torch.tensor([[0, 1, 1, 0]]),
        "use_cache": False,
        "output_attentions": True,
        "output_hidden_states": True,
    }
    outputs = tiny_model(**model_inputs)

    assert len(outputs.hidden_states) == tiny_model.config.num_hidden_layers + 1
    assert len(outputs.attentions) == tiny_model.config.num_hidden_layers
    assert outputs.attentions[0].shape == (1, tiny_model.config.num_attention_heads, 4, 4)

    tuple_outputs = tiny_model(**model_inputs, return_dict=False)
    assert isinstance(tuple_outputs, tuple)
    assert len(tuple_outputs) == 3
    torch.testing.assert_close(tuple_outputs[0], outputs.last_hidden_state)


def test_dynamic_cache_matches_full_sequence(tiny_model):
    input_ids = torch.tensor([[2, 3, 4, 5]])
    token_types = torch.zeros_like(input_ids)

    with torch.no_grad():
        full_output = tiny_model(
            input_ids=input_ids,
            moe_token_types=token_types,
            use_cache=False,
        ).last_hidden_state
        prefill_output = tiny_model(
            input_ids=input_ids[:, :3],
            moe_token_types=token_types[:, :3],
            use_cache=True,
        )
        decode_output = tiny_model(
            input_ids=input_ids[:, 3:],
            attention_mask=torch.ones_like(input_ids),
            moe_token_types=token_types[:, 3:],
            past_key_values=prefill_output.past_key_values,
            use_cache=True,
        )

    assert decode_output.past_key_values.get_seq_length() == input_ids.shape[1]
    torch.testing.assert_close(
        decode_output.last_hidden_state,
        full_output[:, -1:],
        rtol=1e-5,
        atol=1e-6,
    )


def test_static_cache_matches_full_sequence(tiny_model):
    input_ids = torch.tensor([[2, 3, 4, 5]])
    token_types = torch.zeros_like(input_ids)
    cache = StaticCache(config=tiny_model.config, max_cache_len=input_ids.shape[1])

    with torch.no_grad():
        full_output = tiny_model(
            input_ids=input_ids,
            moe_token_types=token_types,
            use_cache=False,
        ).last_hidden_state
        tiny_model(
            input_ids=input_ids[:, :3],
            moe_token_types=token_types[:, :3],
            past_key_values=cache,
            cache_position=torch.arange(3),
            use_cache=True,
        )
        decode_output = tiny_model(
            input_ids=input_ids[:, 3:],
            attention_mask=torch.ones_like(input_ids),
            moe_token_types=token_types[:, 3:],
            past_key_values=cache,
            cache_position=torch.tensor([3]),
            use_cache=True,
        )

    assert torch.as_tensor(cache.get_seq_length()).item() == input_ids.shape[1]
    torch.testing.assert_close(
        decode_output.last_hidden_state,
        full_output[:, -1:],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("attn_implementation", ["sdpa", "flash_attention_2"])
def test_non_eager_attention_backends_are_rejected(attn_implementation):
    config = make_tiny_config()
    config._attn_implementation = attn_implementation

    with pytest.raises(ValueError, match="supports only attn_implementation='eager'"):
        Qwen2_5_VLMoEModel(config)

    with pytest.raises(ValueError, match="supports only attn_implementation='eager'"):
        WallXConfig(attn_implementation=attn_implementation)
