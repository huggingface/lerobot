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

"""Unit tests for Lite KV-Fusion (Faster-WAM, arXiv 2608.02365, Eq. 1).

The paper config is 24 heads x 30 video layers x 1 action layer = 720 mixing logits.
These tests use tiny equivalents and pin the (action layer, head, video layer) indexing,
softmax normalization, K/V weight sharing, cross-layer mixing and gradient reachability.
"""

import pytest
import torch

pytest.importorskip("transformers", reason="fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="fastwam requires the `fastwam` extra (diffusers)")

from lerobot.policies.fastwam.wan import LiteKVFusion


def _cache(
    num_video_layers: int, batch: int, seq: int, heads: int, head_dim: int
) -> list[dict[str, torch.Tensor]]:
    """Per-layer cache where layer l has constant key e_{l+1} and value e_{2(l+1)+1}."""
    width = heads * head_dim
    cache = []
    for layer in range(num_video_layers):
        k = torch.zeros(batch, seq, width)
        v = torch.zeros(batch, seq, width)
        # spread distinct scalars across heads so per-head mixing is observable
        for h in range(heads):
            k[:, :, h * head_dim] = float(layer + 1)
            k[:, :, h * head_dim + 1] = 10.0 * float(layer + 1) + h
            v[:, :, h * head_dim] = 100.0 * float(layer + 1) + h
        cache.append({"k": k, "v": v})
    return cache


def test_mixing_weights_indexing_is_action_layer_head_video_layer():
    """Logits are indexed [action layer j, head h, video layer l]; a wrong transposition
    (e.g. [h, l, j] or [j, l, h]) changes which video layer a head selects."""
    fusion = LiteKVFusion(num_action_layers=2, num_heads=2, num_video_layers=4)
    fusion.logits.data.zero_()
    fusion.logits.data[1, 0, 2] = 10.0  # action layer 1, head 0 -> video layer 2
    fusion.logits.data[1, 1, 3] = 10.0  # action layer 1, head 1 -> video layer 3
    fusion.logits.data[0, 0, 0] = 10.0  # action layer 0, head 0 -> video layer 0
    beta_1 = fusion.mixing_weights(1)
    assert beta_1[0].argmax() == 2
    assert beta_1[1].argmax() == 3
    beta_0 = fusion.mixing_weights(0)
    assert beta_0[0].argmax() == 0
    assert beta_0[1].argmax() == 0  # uniform row: logits only set for layer 1


def test_fused_keys_and_values_come_from_the_selected_layer():
    """With one-hot mixing at layer m, the fused K/V equal cache[m]; perturbing another
    layer's cache must not move the output (catches wrong layer indexing such as always
    reading cache[0] or a transposed logits lookup)."""
    num_video_layers, heads, head_dim = 4, 2, 4
    fusion = LiteKVFusion(num_action_layers=1, num_heads=heads, num_video_layers=num_video_layers)
    # One-hot at video layer 2 for every head; the logit margin keeps the softmax tail
    # far below float32 resolution so the fused tensors equal the selected layer's.
    fusion.logits.data[0, :, 2] = 40.0

    cache = _cache(num_video_layers, batch=2, seq=3, heads=heads, head_dim=head_dim)
    k_fused, v_fused = fusion(cache, action_layer_idx=0)
    torch.testing.assert_close(k_fused, cache[2]["k"])
    torch.testing.assert_close(v_fused, cache[2]["v"])

    # Perturbing a non-selected layer must not change the fused output.
    before_k = k_fused.clone()
    cache[0]["k"] = cache[0]["k"] + 5.0
    cache[1]["v"] = cache[1]["v"] + 5.0
    k_fused2, v_fused2 = fusion(cache, action_layer_idx=0)
    torch.testing.assert_close(k_fused2, before_k)
    torch.testing.assert_close(v_fused2, cache[2]["v"])


def test_keys_and_values_share_the_same_mixing_weights():
    """The paper fuses K and V with the same beta (they are strictly paired). With a
    uniform mixture the fused K and V must both be the mean over layers, per head."""
    num_video_layers, heads, head_dim = 3, 2, 4
    fusion = LiteKVFusion(num_action_layers=1, num_heads=heads, num_video_layers=num_video_layers)
    cache = _cache(num_video_layers, batch=1, seq=2, heads=heads, head_dim=head_dim)
    k_fused, v_fused = fusion(cache, action_layer_idx=0)
    k_mean = torch.stack([c["k"] for c in cache]).mean(dim=0)
    v_mean = torch.stack([c["v"] for c in cache]).mean(dim=0)
    torch.testing.assert_close(k_fused, k_mean)
    torch.testing.assert_close(v_fused, v_mean)


def test_gradients_reach_cache_and_logits():
    """Gradient reachability: a loss on the fused output must produce gradients on the
    fusion logits AND on the cached K/V (the path the video backbone's activations flow
    through during DoT training). A detached cache would zero the backbone gradients."""
    num_video_layers, heads, head_dim = 3, 2, 4
    fusion = LiteKVFusion(num_action_layers=1, num_heads=heads, num_video_layers=num_video_layers)
    cache = _cache(num_video_layers, batch=1, seq=2, heads=heads, head_dim=head_dim)
    for entry in cache:
        entry["k"].requires_grad_(True)
        entry["v"].requires_grad_(True)
    k_fused, v_fused = fusion(cache, action_layer_idx=0)
    loss = (k_fused + v_fused).pow(2).mean()
    loss.backward()
    assert fusion.logits.grad is not None and fusion.logits.grad.abs().sum() > 0
    for entry in cache:
        assert entry["k"].grad is not None and entry["k"].grad.abs().sum() > 0
        assert entry["v"].grad is not None and entry["v"].grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_fusion_preserves_float32_softmax_with_low_precision_storage(dtype):
    fusion = LiteKVFusion(1, 2, 3).to(dtype=dtype)
    with torch.no_grad():
        fusion.logits.copy_(torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]))
    expected = torch.softmax(fusion.logits[0].float(), dim=-1)
    assert fusion.mixing_weights(0).dtype == torch.float32
    torch.testing.assert_close(fusion.mixing_weights(0), expected, rtol=0, atol=0)
    cache = [
        {key: value.to(dtype).requires_grad_() for key, value in entry.items()}
        for entry in _cache(3, batch=1, seq=2, heads=2, head_dim=4)
    ]
    k, v = fusion(cache, 0)
    assert k.dtype == v.dtype == dtype
    (k.float().square().mean() + v.float().square().mean()).backward()
    assert torch.isfinite(fusion.logits.grad).all() and fusion.logits.grad.abs().sum() > 0
    assert all(torch.isfinite(entry[key].grad).all() for entry in cache for key in ("k", "v"))
