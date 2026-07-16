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

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VLVisionAttention,
)

from lerobot.policies.wall_x.qwen_model import vision_attention  # noqa: E402
from lerobot.policies.wall_x.qwen_model.vision_attention import (  # noqa: E402
    WallXVisionAttention,
    configure_wall_x_vision_attention,
)


def make_tiny_vision_config(attn_implementation: str) -> Qwen2_5_VLVisionConfig:
    config = Qwen2_5_VLVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        depth=1,
    )
    config._attn_implementation = attn_implementation
    return config


def make_attention_inputs():
    hidden_states = torch.randn(7, 16)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)
    cos = torch.ones(7, 8)
    sin = torch.zeros(7, 8)
    return hidden_states, cu_seqlens, (cos, sin)


def fake_varlen_attention(
    query,
    key,
    value,
    cu_seq_q,
    cu_seq_k,
    max_q,
    max_k,
    *,
    return_aux=None,
    scale=None,
    window_size=(-1, -1),
):
    del cu_seq_k, max_q, max_k, return_aux
    assert window_size == (-1, -1)

    outputs = []
    boundaries = zip(cu_seq_q[:-1].tolist(), cu_seq_q[1:].tolist(), strict=True)
    for start, end in boundaries:
        query_chunk = query[start:end].transpose(0, 1)
        key_chunk = key[start:end].transpose(0, 1)
        value_chunk = value[start:end].transpose(0, 1)
        attention = torch.matmul(query_chunk, key_chunk.transpose(-2, -1)) * scale
        attention = torch.softmax(attention, dim=-1, dtype=torch.float32).to(query.dtype)
        outputs.append(torch.matmul(attention, value_chunk).transpose(0, 1))
    return torch.cat(outputs)


def test_auto_backend_falls_back_to_native_sdpa_on_cpu():
    torch.manual_seed(0)
    config = make_tiny_vision_config("sdpa")
    native_attention = Qwen2_5_VLVisionAttention(config).eval()
    wall_x_attention = WallXVisionAttention(config, backend="auto").eval()
    wall_x_attention.load_state_dict(native_attention.state_dict())
    hidden_states, cu_seqlens, position_embeddings = make_attention_inputs()

    native_output = native_attention(
        hidden_states,
        cu_seqlens,
        position_embeddings=position_embeddings,
    )
    wall_x_output = wall_x_attention(
        hidden_states,
        cu_seqlens,
        position_embeddings=position_embeddings,
    )

    torch.testing.assert_close(wall_x_output, native_output)


def test_packed_varlen_matches_eager_outputs_and_gradients(monkeypatch):
    monkeypatch.setattr(vision_attention, "_varlen_attn", fake_varlen_attention)
    monkeypatch.setattr(vision_attention, "_VARLEN_USES_WINDOW_SIZE", True)
    monkeypatch.setattr(vision_attention, "_supports_varlen_attention", lambda *_args: True)

    torch.manual_seed(0)
    eager_config = make_tiny_vision_config("eager")
    sdpa_config = make_tiny_vision_config("sdpa")
    eager_attention = Qwen2_5_VLVisionAttention(eager_config).eval()
    varlen_attention = WallXVisionAttention(sdpa_config, backend="varlen").eval()
    varlen_attention.load_state_dict(eager_attention.state_dict())

    hidden_states, cu_seqlens, position_embeddings = make_attention_inputs()
    eager_hidden_states = hidden_states.detach().clone().requires_grad_()
    varlen_hidden_states = hidden_states.detach().clone().requires_grad_()

    eager_output = eager_attention(
        eager_hidden_states,
        cu_seqlens,
        position_embeddings=position_embeddings,
    )
    varlen_output = varlen_attention(
        varlen_hidden_states,
        cu_seqlens,
        position_embeddings=position_embeddings,
    )
    torch.testing.assert_close(varlen_output, eager_output)

    eager_output.square().mean().backward()
    varlen_output.square().mean().backward()
    torch.testing.assert_close(varlen_hidden_states.grad, eager_hidden_states.grad)
    for (eager_name, eager_param), (varlen_name, varlen_param) in zip(
        eager_attention.named_parameters(),
        varlen_attention.named_parameters(),
        strict=True,
    ):
        assert varlen_name == eager_name
        torch.testing.assert_close(varlen_param.grad, eager_param.grad)


def test_configure_vision_attention_preserves_state_dict_schema_and_values(monkeypatch):
    monkeypatch.setattr(vision_attention, "_varlen_attn", fake_varlen_attention)
    torch.manual_seed(0)
    config = make_tiny_vision_config("sdpa")
    vision_model = nn.Module()
    block = nn.Module()
    block.attn = Qwen2_5_VLVisionAttention(config)
    vision_model.blocks = nn.ModuleList([block])
    original_state = {key: value.clone() for key, value in vision_model.state_dict().items()}

    configure_wall_x_vision_attention(vision_model, backend="auto")

    assert isinstance(vision_model.blocks[0].attn, WallXVisionAttention)
    assert vision_model.state_dict().keys() == original_state.keys()
    for key, original_value in original_state.items():
        torch.testing.assert_close(vision_model.state_dict()[key], original_value)


def test_explicit_varlen_reports_unsupported_runtime():
    config = make_tiny_vision_config("sdpa")
    attention = WallXVisionAttention(config, backend="varlen")
    hidden_states, cu_seqlens, position_embeddings = make_attention_inputs()

    with pytest.raises(RuntimeError, match="vision_attn_implementation='varlen' cannot be used"):
        attention(
            hidden_states,
            cu_seqlens,
            position_embeddings=position_embeddings,
        )
