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

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.rlt.configuration_rlt import RLTActorConfig, RLTConfig, RLTokenConfig
from lerobot.policies.rlt.modeling_rlt import RLTPolicy
from lerobot.policies.rlt.vla_adapter import (
    OBS_REFERENCE_ACTION,
    OBS_RLT_STATE,
    OBS_VLA_EMBEDDINGS,
    PI05PrefixRLTAdapter,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def _make_rlt_config() -> RLTConfig:
    return RLTConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        device="cpu",
        chunk_size=3,
        rl_token=RLTokenConfig(
            input_dim=4,
            rl_token_dim=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            ff_dim=8,
            dropout=0.0,
        ),
        actor=RLTActorConfig(hidden_dims=[8], residual_scale=0.25, clamp_output=False),
    )


def test_rlt_policy_accepts_compact_rlt_state_and_reference_action():
    policy = RLTPolicy(_make_rlt_config())
    for param in policy.actor.net.parameters():
        param.data.zero_()

    reference_action = torch.linspace(-0.5, 0.5, policy._action_chunk_dim)
    action = policy.select_action(
        {
            OBS_RLT_STATE: torch.ones(policy.config.rl_token.rl_token_dim),
            OBS_REFERENCE_ACTION: reference_action,
            OBS_STATE: torch.zeros(3),
        }
    )

    assert action.shape == reference_action.shape
    assert torch.allclose(action, reference_action)


def test_rlt_policy_can_encode_vla_embeddings_when_no_compact_state_is_present():
    policy = RLTPolicy(_make_rlt_config())

    action = policy.select_action(
        {
            OBS_VLA_EMBEDDINGS: torch.randn(5, policy.config.rl_token.input_dim),
            OBS_REFERENCE_ACTION: torch.zeros(policy._action_chunk_dim),
            OBS_STATE: torch.zeros(3),
        }
    )

    assert action.shape == (policy._action_chunk_dim,)


def test_flatten_action_chunk_handles_single_flat_chunk():
    flat_chunk = torch.arange(6)
    assert PI05PrefixRLTAdapter.flatten_action_chunk(flat_chunk).shape == (1, 6)

    unbatched_chunk = torch.arange(6).reshape(3, 2)
    assert torch.equal(PI05PrefixRLTAdapter.flatten_action_chunk(unbatched_chunk), flat_chunk.unsqueeze(0))
