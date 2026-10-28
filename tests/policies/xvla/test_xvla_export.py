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

import io

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # noqa: E402
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_LANGUAGE_TOKENS, OBS_STATE  # noqa: E402

TINY_FLORENCE = {
    "model_type": "florence2",
    "bos_token_id": 0,
    "eos_token_id": 2,
    "pad_token_id": 1,
    "projection_dim": 64,
    "vocab_size": 51289,
    "text_config": {
        "vocab_size": 51289,
        "d_model": 64,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "num_hidden_layers": 1,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": 128,
        "decoder_ffn_dim": 128,
        "max_position_embeddings": 64,
    },
    "vision_config": {
        "model_type": "davit",
        "drop_path_rate": 0.0,
        "patch_size": [7, 3, 3, 3],
        "patch_stride": [4, 2, 2, 2],
        "patch_padding": [3, 1, 1, 1],
        "patch_prenorm": [False, True, True, True],
        "dim_embed": [16, 32, 64, 128],
        "num_heads": [2, 2, 4, 4],
        "num_groups": [2, 2, 4, 4],
        "depths": [1, 1, 1, 1],
        "window_size": 2,
        "projection_dim": 64,
        "visual_temporal_embedding": {"type": "COSINE", "max_temporal_embeddings": 100},
        "image_pos_embed": {"type": "learned_abs_2d", "max_pos_embeddings": 50},
        "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
    },
}


CAMS = [f"{OBS_IMAGES}.cam{i}" for i in range(3)]


def tiny_policy() -> XVLAPolicy:
    config = XVLAConfig(
        input_features={
            **{cam: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for cam in CAMS},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
        florence_config=TINY_FLORENCE,
        hidden_size=32,
        depth=1,
        num_heads=2,
        chunk_size=4,
        n_action_steps=4,
        num_image_views=3,
        resize_imgs_with_padding=None,
        num_denoising_steps=2,
        device="cpu",
    )
    torch.manual_seed(0)
    return XVLAPolicy(config).eval()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("present_views", [1, 2, 3])
def test_export_with_a_padding_view_has_fixed_shapes(batch_size, present_views):
    policy = tiny_policy()
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(3, 1000, (batch_size, 8), generator=gen)
    state = torch.randn(batch_size, 8, generator=gen)
    images = torch.rand(batch_size, present_views, 3, 64, 64, generator=gen)
    noise = torch.randn(batch_size, policy.config.chunk_size, policy.model.dim_action, generator=gen)

    class Predict(nn.Module):
        def forward(self, ids, state, images, noise):
            batch = {OBS_LANGUAGE_TOKENS: ids, OBS_STATE: state}
            batch.update({CAMS[i]: images[:, i] for i in range(present_views)})
            return policy.predict_action_chunk(batch, noise=noise)

    model = Predict()
    # Exported under no_grad, as for inference; a traced grad-mode switch does not load back on PyTorch 2.11.
    with torch.no_grad():
        ep = torch.export.export(model, (ids, state, images, noise), strict=False)
    # Fixed-shape backends such as Torch-TensorRT cannot run sizes or checks that depend on tensor values.
    assert not [n for n in ep.graph.nodes if "nonzero" in str(n.target) or "_assert_scalar" in str(n.target)]
    assert not ep.range_constraints
    expected = model(ids, state, images, noise)
    buffer = io.BytesIO()
    torch.export.save(ep, buffer)
    buffer.seek(0)
    for program in (ep, torch.export.load(buffer)):
        assert torch.equal(program.module()(ids, state, images, noise), expected)
    # Decomposed operators may round differently.
    torch.testing.assert_close(ep.run_decompositions().module()(ids, state, images, noise), expected)


def test_forward_vlm_checks_a_given_view_count_and_takes_any_mask_without_one():
    model = tiny_policy().model
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(3, 1000, (2, 8), generator=gen)
    images = torch.rand(2, 3, 3, 64, 64, generator=gen)
    prefix = torch.tensor([[1, 1, 0], [1, 1, 0]])
    counted = model.forward_vlm(ids, images, prefix, num_valid_views=2)
    read = model.forward_vlm(ids, images, prefix)
    assert counted.keys() == read.keys() and all(torch.equal(counted[k], read[k]) for k in read)
    model.forward_vlm(ids, images, torch.tensor([[1, 0, 1], [1, 1, 1]]))
    for mask in ([[1, 1, 1], [1, 1, 0]], [[1, 0, 1], [1, 0, 1]], [[0, 1, 1], [0, 1, 1]]):
        with pytest.raises(ValueError, match="first 2 views"):
            model.forward_vlm(ids, images, torch.tensor(mask), num_valid_views=2)
    for count in (0, 4):
        with pytest.raises(ValueError, match="between 1 and 3"):
            model.forward_vlm(ids, images, torch.ones(2, 3), num_valid_views=count)


def test_training_forward_with_the_view_count_matches_the_mask_path():
    policy = tiny_policy()
    gen = torch.Generator().manual_seed(0)
    batch = {
        OBS_LANGUAGE_TOKENS: torch.randint(3, 1000, (2, 8), generator=gen),
        OBS_STATE: torch.randn(2, 8, generator=gen),
        ACTION: torch.randn(2, policy.config.chunk_size, 20, generator=gen),
        **{cam: torch.rand(2, 3, 64, 64, generator=gen) for cam in CAMS[:2]},
    }
    inputs, targets = policy._build_model_inputs(batch), policy._prepare_action_targets(batch)
    assert inputs["num_valid_views"] == 2
    torch.manual_seed(0)
    counted = policy.model(action=targets, **inputs)
    torch.manual_seed(0)
    read = policy.model(action=targets, **{**inputs, "num_valid_views": None})
    assert counted.keys() == read.keys() and all(torch.equal(counted[k], read[k]) for k in read)
