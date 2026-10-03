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

pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot.utils.random_utils import seeded_context

HORIZON = 8
ACTION_DIM = 2


def _make_policy(noise_scheduler_type: str) -> DiffusionPolicy:
    config = DiffusionConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        n_obs_steps=2,
        horizon=HORIZON,
        n_action_steps=4,
        down_dims=(16, 32),
        diffusion_step_embed_dim=16,
        noise_scheduler_type=noise_scheduler_type,
        num_inference_steps=5,
        device="cpu",
    )
    policy = DiffusionPolicy(config)
    policy.eval()
    return policy


def _chunk_batch() -> dict[str, torch.Tensor]:
    return {OBS_STATE: torch.randn(2, 2, 3), OBS_ENV_STATE: torch.randn(2, 2, 4)}


def test_diffusion_predict_action_chunk_uses_given_noise():
    policy = _make_policy("DDIM")
    batch = _chunk_batch()
    noise = torch.randn(2, HORIZON, ACTION_DIM)
    noise_before = noise.clone()

    actions = policy.predict_action_chunk(batch, noise=noise)
    actions_again = policy.predict_action_chunk(batch, noise=noise)
    other_actions = policy.predict_action_chunk(batch, noise=torch.randn_like(noise))

    assert torch.equal(actions, actions_again)
    assert torch.equal(noise, noise_before)
    assert not torch.equal(actions, other_actions)


@pytest.mark.parametrize("noise_scheduler_type", ["DDPM", "DDIM"])
def test_diffusion_predict_action_chunk_default_noise_matches_seeded_draw(noise_scheduler_type):
    policy = _make_policy(noise_scheduler_type)
    batch = _chunk_batch()

    with seeded_context(0):
        actions = policy.predict_action_chunk(batch)
    with seeded_context(0):
        noise = torch.randn(size=(2, HORIZON, ACTION_DIM), dtype=torch.float32, device="cpu")
        actions_with_noise = policy.predict_action_chunk(batch, noise=noise)

    assert torch.equal(actions, actions_with_noise)


def test_diffusion_select_action_uses_given_noise_for_new_chunks():
    policy = _make_policy("DDIM")
    observation = {OBS_STATE: torch.randn(2, 3), OBS_ENV_STATE: torch.randn(2, 4)}
    noise = torch.randn(2, HORIZON, ACTION_DIM)

    def first_action(chunk_noise: torch.Tensor) -> torch.Tensor:
        policy.reset()
        return policy.select_action(dict(observation), noise=chunk_noise)

    action = first_action(noise)
    assert torch.equal(action, first_action(noise))
    assert not torch.equal(action, first_action(torch.randn_like(noise)))
