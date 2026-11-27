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

import pytest
import torch

pytest.importorskip("transformers")
from transformers import DynamicCache  # noqa: E402

from lerobot.policies.smolvla.smolvlm_with_expert import PrefixKVCache  # noqa: E402

PREFIX, SUFFIX, LAYERS = 7, 3, 4


def _kv(seq: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 5, seq, 8, generator=g), torch.randn(2, 5, seq, 8, generator=g)


def _prefill_step_crop(cache):
    """What SmolVLA does: fill the prefix, append one denoising step's K/V, crop back to the prefix."""
    for i in range(LAYERS):
        cache.update(*_kv(PREFIX, i), i)
    for i in range(LAYERS):
        cache.update(*_kv(SUFFIX, 100 + i), i)
    for layer in cache.layers:
        layer.crop(PREFIX - layer.get_seq_length())
    return cache


def test_matches_dynamic_cache():
    ours, ref = _prefill_step_crop(PrefixKVCache()), _prefill_step_crop(DynamicCache())
    assert len(ours.layers) == LAYERS
    for a, b in zip(ours.layers, ref.layers, strict=True):
        assert torch.equal(a.keys, b.keys)
        assert torch.equal(a.values, b.values)
        assert a.get_seq_length() == b.get_seq_length() == PREFIX


def test_update_returns_full_sequence():
    cache = PrefixKVCache()
    cache.update(*_kv(PREFIX, 0), 0)
    k, v = cache.update(*_kv(SUFFIX, 1), 0)
    assert k.shape[-2] == v.shape[-2] == PREFIX + SUFFIX
    assert cache.get_seq_length() == PREFIX + SUFFIX
    assert cache.get_seq_length(layer_idx=3) == 0


def test_crop_contract():
    cache = PrefixKVCache()
    cache.update(*_kv(PREFIX, 0), 0)
    layer = cache.layers[0]
    layer.crop(0)  # cross-attention layers are cropped by 0 every step: must be a no-op
    assert layer.get_seq_length() == PREFIX
    layer.crop(-2)
    assert layer.get_seq_length() == PREFIX - 2
    with pytest.raises(ValueError):
        layer.crop(1)


class _Prefill(torch.nn.Module):
    def forward(self, k, v):
        cache = PrefixKVCache()
        cache.update(k, v, 0)
        return cache.layers[0].keys, cache.layers[0].values


def test_prefill_exports_without_concat():
    """The first update must not concatenate onto an empty tensor (that op stops export, e.g. to OpenVINO)."""
    exported = torch.export.export(_Prefill(), _kv(PREFIX, 0))
    assert "aten.cat" not in str(exported.graph)
