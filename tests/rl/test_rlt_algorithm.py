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

import itertools

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.rlt.configuration_rlt import RLTActorConfig, RLTConfig, RLTokenConfig
from lerobot.policies.rlt.modeling_rlt import RLTPolicy
from lerobot.policies.rlt.vla_adapter import OBS_RLT_STATE
from lerobot.rl.algorithms.rlt.configuration_rlt import RLTAlgorithmConfig
from lerobot.utils.constants import ACTION, OBS_STATE


def _make_rlt_policy() -> RLTPolicy:
    config = RLTConfig(
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
    return RLTPolicy(config)


def _make_replay_batch(policy: RLTPolicy, batch_size: int = 2) -> dict:
    action_chunk_dim = policy._action_chunk_dim
    return {
        "state": {
            OBS_RLT_STATE: torch.randn(batch_size, policy.config.rl_token.rl_token_dim),
            OBS_STATE: torch.randn(batch_size, 3),
        },
        "next_state": {
            OBS_RLT_STATE: torch.randn(batch_size, policy.config.rl_token.rl_token_dim),
            OBS_STATE: torch.randn(batch_size, 3),
        },
        ACTION: torch.randn(batch_size, action_chunk_dim),
        "reward": torch.randn(batch_size),
        "done": torch.zeros(batch_size),
        "truncated": torch.zeros(batch_size),
        "complementary_info": {
            "reference_action": torch.zeros(batch_size, action_chunk_dim),
            "next_reference_action": torch.ones(batch_size, action_chunk_dim),
        },
    }


def test_rlt_algorithm_online_update_accepts_compact_rlt_state():
    policy = _make_rlt_policy()
    config = RLTAlgorithmConfig.from_policy_config(policy.config)
    config.utd_ratio = 1
    algorithm = config.build_algorithm(policy)
    algorithm.transition_to_online()

    stats = algorithm.update(iter(itertools.repeat(_make_replay_batch(policy))))

    assert "loss_critic" in stats.losses
    assert algorithm.optimization_step == 1


def test_rlt_algorithm_keeps_next_reference_action_for_target_policy():
    policy = _make_rlt_policy()
    algorithm = RLTAlgorithmConfig.from_policy_config(policy.config).build_algorithm(policy)
    batch = _make_replay_batch(policy)

    prepared = algorithm._prepare_forward_batch(batch)

    assert torch.equal(prepared["next_reference_action"], batch["complementary_info"]["next_reference_action"])
