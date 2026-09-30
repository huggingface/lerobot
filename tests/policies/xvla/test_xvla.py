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

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # noqa: E402
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE  # noqa: E402

BATCH_SIZE = 2
IMAGE_KEY = "observation.images.image"
TINY_FLORENCE_CONFIG = {
    "vision_config": {
        "depths": [1, 1],
        "patch_size": [7, 3],
        "patch_stride": [4, 2],
        "patch_padding": [3, 1],
        "patch_prenorm": [False, True],
        "embed_dim": [16, 32],
        "num_heads": [1, 2],
        "num_groups": [1, 2],
        "window_size": 2,
        "projection_dim": 32,
    },
    "text_config": {
        "vocab_size": 64,
        "d_model": 32,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": 64,
        "decoder_ffn_dim": 64,
        "max_position_embeddings": 64,
    },
}


@pytest.fixture(scope="module")
def policy() -> XVLAPolicy:
    config = XVLAConfig(
        input_features={
            IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
        florence_config=TINY_FLORENCE_CONFIG,
        hidden_size=32,
        depth=1,
        num_heads=2,
        len_soft_prompts=2,
        dim_time=8,
        max_len_seq=128,
        chunk_size=4,
        n_action_steps=4,
        num_denoising_steps=2,
        device="cpu",
    )
    torch.manual_seed(0)
    return XVLAPolicy(config).eval()


def make_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return {
        IMAGE_KEY: torch.rand(BATCH_SIZE, 3, 32, 32, generator=generator),
        OBS_STATE: torch.randn(BATCH_SIZE, 6, generator=generator),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 64, (BATCH_SIZE, 5), generator=generator),
    }


def make_noise(policy: XVLAPolicy, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH_SIZE, policy.config.chunk_size, policy.model.dim_action, generator=generator)


def test_same_noise_gives_same_actions(policy):
    batch = make_batch()
    noise = make_noise(policy, seed=2)
    noise_before = noise.clone()

    first = policy.predict_action_chunk(batch, noise=noise)
    second = policy.predict_action_chunk(batch, noise=noise)

    assert torch.equal(first, second)
    assert torch.equal(noise, noise_before)


def test_default_noise_matches_explicit_draw(policy):
    batch = make_batch()

    torch.manual_seed(4)
    default = policy.predict_action_chunk(batch)

    torch.manual_seed(4)
    noise = torch.randn(
        BATCH_SIZE, policy.config.chunk_size, policy.model.dim_action, device="cpu", dtype=torch.float32
    )
    explicit = policy.predict_action_chunk(batch, noise=noise)

    assert torch.equal(default, explicit)


def test_select_action_uses_noise_for_a_new_chunk(policy):
    batch = make_batch()
    noise = make_noise(policy, seed=5)
    expected = policy.predict_action_chunk(batch, noise=noise)

    policy.reset()
    action = policy.select_action(batch, noise=noise)

    assert torch.equal(action, expected[:, 0])
