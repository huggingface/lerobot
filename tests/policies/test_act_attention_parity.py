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

from copy import deepcopy

import pytest
import torch

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTDecoderLayer, ACTEncoderLayer
from lerobot.utils.random_utils import seeded_context


@pytest.mark.parametrize("pre_norm", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_act_sdpa_matches_explicit_attention_outputs_and_gradients(pre_norm, masked):
    config = ACTConfig(dim_model=16, n_heads=2, dim_feedforward=32, dropout=0.0, pre_norm=pre_norm)
    with seeded_context(42):
        optimized = torch.nn.ModuleList([ACTEncoderLayer(config), ACTDecoderLayer(config)]).double().train()
        source = torch.randn(5, 2, 16, dtype=torch.float64)
        target = torch.randn(3, 2, 16, dtype=torch.float64)
        source_pos = torch.randn(5, 1, 16, dtype=torch.float64)
        target_pos = torch.randn(3, 1, 16, dtype=torch.float64)
        loss_weights = torch.randn_like(target)
    reference = deepcopy(optimized)
    for attention in reference.modules():
        if isinstance(attention, torch.nn.MultiheadAttention):
            forward = attention.forward

            def explicit_forward(*args, _forward=forward, **kwargs):
                kwargs["need_weights"] = True
                return _forward(*args, **kwargs)

            attention.forward = explicit_forward
    mask = (
        torch.tensor([[False, False, False, True, True], [False, False, False, False, True]])
        if masked
        else None
    )
    results = []
    for model in (optimized, reference):
        x, y = source.clone().requires_grad_(), target.clone().requires_grad_()
        memory = model[0](x, pos_embed=source_pos, key_padding_mask=mask)
        output = model[1](y, memory, decoder_pos_embed=target_pos, encoder_pos_embed=source_pos)
        (output * loss_weights).sum().backward()
        results.append((output.detach(), x.grad, y.grad, dict(model.named_parameters())))
    for actual, expected in zip(results[0][:3], results[1][:3], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)
    for name, parameter in results[0][3].items():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, results[1][3][name].grad, atol=1e-10, rtol=1e-8)
