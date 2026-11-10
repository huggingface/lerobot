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

"""Knowledge insulation and its hooks in PI0.5's shared joint layer, on a tiny backbone."""

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.policies.pi05.modeling_pi05 import (  # noqa: E402
    GemmaConfig,
    PaliGemmaWithExpertModel,
    _joint_attention,
)

PREFIX, SUFFIX, WIDTH = 10, 6, 64


@pytest.fixture(scope="module")
def backbone() -> PaliGemmaWithExpertModel:
    torch.manual_seed(0)
    tiny = GemmaConfig(width=WIDTH, depth=2, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    model = PaliGemmaWithExpertModel(tiny, tiny, use_adarms=[False, True], precision=torch.float32)
    model.paligemma.model.vision_tower = None  # the joint layers never use it
    model.train()
    return model


def _forward(model, *, knowledge_insulation, **kwargs):
    generator = torch.Generator().manual_seed(1)
    prefix = torch.randn(2, PREFIX, WIDTH, generator=generator)
    suffix = torch.randn(2, SUFFIX, WIDTH, generator=generator)
    cond = torch.randn(2, WIDTH, generator=generator)
    causal = torch.zeros(2, PREFIX + SUFFIX, dtype=torch.long)
    causal[:, PREFIX] = 1
    blocks = causal.cumsum(1)
    allowed = blocks[:, None, :] <= blocks[:, :, None]
    mask = torch.where(allowed[:, None], 0.0, torch.finfo(torch.float32).min)
    position_ids = torch.arange(PREFIX + SUFFIX).expand(2, -1)
    model.zero_grad(set_to_none=True)
    model.knowledge_insulation = knowledge_insulation
    try:
        (prefix_out, suffix_out), _ = model.forward(
            attention_mask=mask,
            position_ids=position_ids,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            adarms_cond=[None, cond],
            **kwargs,
        )
    finally:
        model.knowledge_insulation = False
    return prefix_out, suffix_out


def _vlm_grads(model):
    return {
        name: param.grad
        for name, param in model.named_parameters()
        if "paligemma.model.language_model" in name
    }


def test_knowledge_insulation_keeps_forward_values(backbone):
    with torch.no_grad():
        reference = _forward(backbone, knowledge_insulation=False)
        insulated = _forward(backbone, knowledge_insulation=True)
    torch.testing.assert_close(insulated[0], reference[0])
    torch.testing.assert_close(insulated[1], reference[1])


def test_knowledge_insulation_blocks_action_gradients_into_the_vlm(backbone):
    _, suffix_out = _forward(backbone, knowledge_insulation=False)
    suffix_out.pow(2).mean().backward()
    assert any(grad is not None and grad.abs().max() > 0 for grad in _vlm_grads(backbone).values())

    _, suffix_out = _forward(backbone, knowledge_insulation=True)
    suffix_out.pow(2).mean().backward()
    assert all(grad is None or grad.abs().max() == 0 for grad in _vlm_grads(backbone).values())


def test_suppress_prefix_grads_leaves_the_vlm_out_of_autograd(backbone):
    prefix_out, suffix_out = _forward(backbone, knowledge_insulation=True, suppress_prefix_grads=True)
    assert not prefix_out.requires_grad
    suffix_out.pow(2).mean().backward()
    assert all(grad is None for grad in _vlm_grads(backbone).values())


def test_attention_fn_receives_each_query_group(backbone):
    parts = []

    def attention_fn(self_attn, query, key, value, mask, scaling, part):
        parts.append((part, query.shape[2]))
        return _joint_attention(self_attn, query, key, value, mask, scaling, fp32_attention=False)

    with torch.no_grad():
        reference = _forward(backbone, knowledge_insulation=True)
        hooked = _forward(backbone, knowledge_insulation=True, attention_fn=attention_fn)
    assert parts == [("vlm", PREFIX), ("action", SUFFIX)] * 2  # two layers
    torch.testing.assert_close(hooked[1], reference[1])
