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

"""The precomputed cross-attention expert K/V must match the per-step projection."""

import copy
from unittest import mock

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.random_utils import set_seed
from tests.utils import skip_if_package_missing

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _make_model_and_inputs(dtype, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    set_seed(0)
    config = SmolVLAConfig(
        max_action_dim=7, max_state_dim=8, chunk_size=4, n_action_steps=4, num_steps=3, num_vlm_layers=2
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.images.camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}

    model = SmolVLAPolicy(config).to(device=device, dtype=dtype).eval().model
    # denoise_step upcasts the expert output to float32 before action_out_proj
    model.action_out_proj.to(dtype=torch.float32)

    inputs = {
        "images": [torch.rand(1, 3, 512, 512, device=device, dtype=dtype)],
        "img_masks": [torch.ones(1, dtype=torch.bool, device=device)],
        "lang_tokens": torch.randint(0, 100, (1, 8), device=device),
        "lang_masks": torch.ones(1, 8, dtype=torch.bool, device=device),
        "state": torch.randn(1, 8, device=device, dtype=dtype),
    }
    return model, inputs


@skip_if_package_missing("transformers")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_precomputed_cross_attn_kv_matches_per_step(dtype, device):
    model, inputs = _make_model_and_inputs(dtype, device)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(**inputs)
    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
        position_ids=torch.cumsum(prefix_pad_masks, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    cross_attn_kv = model.vlm_with_expert.precompute_cross_attn_kv(past_key_values)
    assert set(cross_attn_kv) == {1}
    frozen = tuple((layer.keys, layer.values) for layer in past_key_values.layers)
    from_tuple = model.vlm_with_expert.precompute_cross_attn_kv(frozen)
    assert from_tuple.keys() == cross_attn_kv.keys()
    for layer_idx, (key, value) in cross_attn_kv.items():
        tuple_key, tuple_value = from_tuple[layer_idx]
        assert torch.equal(key, tuple_key)
        assert torch.equal(value, tuple_value)

    x_t = torch.randn(1, model.config.chunk_size, model.config.max_action_dim, device=device, dtype=dtype)
    timestep = torch.full((1,), 0.4, device=device)
    kwargs = {"prefix_pad_masks": prefix_pad_masks, "x_t": x_t, "timestep": timestep}

    expected = model.denoise_step(past_key_values=copy.deepcopy(past_key_values), **kwargs)
    got = model.denoise_step(
        past_key_values=copy.deepcopy(past_key_values), cross_attn_kv=cross_attn_kv, **kwargs
    )

    assert torch.equal(got, expected)


@skip_if_package_missing("transformers")
@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_sample_actions_projects_prefix_kv_once_per_chunk(device):
    model, inputs = _make_model_and_inputs(torch.float32, device)
    vlm = model.vlm_with_expert
    noise = torch.randn(1, model.config.chunk_size, model.config.max_action_dim, device=device)

    with mock.patch.object(vlm, "_project_prefix_kv", wraps=vlm._project_prefix_kv) as project:
        got = model.sample_actions(**inputs, noise=noise)
    # The test model has one cross-attention layer.
    assert project.call_count == 1

    with (
        mock.patch.object(vlm, "precompute_cross_attn_kv", return_value=None),
        mock.patch.object(vlm, "_project_prefix_kv", wraps=vlm._project_prefix_kv) as project,
    ):
        expected = model.sample_actions(**inputs, noise=noise)
    assert project.call_count == model.config.num_steps

    assert torch.equal(got, expected)
