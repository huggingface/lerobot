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

"""Exercise the prefix-cache contract against a real tiny Gemma on CPU."""

from copy import deepcopy

import pytest
import torch

pytest.importorskip("transformers")

from transformers import GemmaConfig, GemmaModel  # noqa: E402

from lerobot.policies.common.vla_utils import ReadOnlyPrefixCache  # noqa: E402


@pytest.mark.parametrize("batch_size", [1, 2])
@torch.no_grad()
def test_prefix_cache_repeated_suffix_matches_cloned_dynamic_cache(batch_size):
    torch.manual_seed(0)
    config = GemmaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    config._attn_implementation = "eager"
    model = GemmaModel(config).eval()
    prefix = model(input_ids=torch.randint(0, 32, (batch_size, 5)), use_cache=True).past_key_values
    saved = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
    shared = ReadOnlyPrefixCache.from_cache(prefix)
    for _ in range(3):
        suffix = torch.randn(batch_size, 3, config.hidden_size)
        kwargs = {
            "inputs_embeds": suffix,
            "attention_mask": torch.zeros(batch_size, 1, 3, 8),
            "position_ids": torch.arange(5, 8).unsqueeze(0).expand(batch_size, -1),
            "use_cache": False,
        }
        expected = model(past_key_values=deepcopy(prefix), **kwargs).last_hidden_state
        actual = model(past_key_values=shared, **kwargs).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert shared.get_seq_length() == prefix.get_seq_length() == 5
        for layer, (keys, values) in zip(prefix.layers, saved, strict=True):
            torch.testing.assert_close(layer.keys, keys, atol=0, rtol=0)
            torch.testing.assert_close(layer.values, values, atol=0, rtol=0)
