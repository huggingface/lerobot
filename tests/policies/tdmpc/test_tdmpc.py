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

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_STATE

BATCH_SIZE = 2
STATE_DIM = 4
ENV_STATE_DIM = 3
ACTION_DIM = 2
IMAGE_SHAPE = (3, 64, 64)


def _make_policy(extra_inputs: dict[str, PolicyFeature]) -> TDMPCPolicy:
    config = TDMPCConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            **extra_inputs,
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        device="cpu",
    )
    return TDMPCPolicy(config)


def _make_batch(policy: TDMPCPolicy) -> dict[str, torch.Tensor]:
    horizon = policy.config.horizon
    batch = {
        OBS_STATE: torch.rand(BATCH_SIZE, horizon + 1, STATE_DIM),
        ACTION: torch.rand(BATCH_SIZE, horizon, ACTION_DIM) * 2 - 1,
        "next.reward": torch.rand(BATCH_SIZE, horizon),
        "index": torch.arange(BATCH_SIZE),
        f"{OBS_STATE}_is_pad": torch.zeros(BATCH_SIZE, horizon + 1, dtype=torch.bool),
        "action_is_pad": torch.zeros(BATCH_SIZE, horizon, dtype=torch.bool),
        "next.reward_is_pad": torch.zeros(BATCH_SIZE, horizon, dtype=torch.bool),
    }
    for key in policy.config.input_features:
        if key == OBS_ENV_STATE:
            batch[key] = torch.rand(BATCH_SIZE, horizon + 1, ENV_STATE_DIM)
        elif key.startswith(OBS_IMAGE):
            batch[key] = torch.rand(BATCH_SIZE, horizon + 1, *IMAGE_SHAPE)
    return batch


@pytest.mark.parametrize(
    "extra_inputs",
    [
        {},
        {OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(ENV_STATE_DIM,))},
        {OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=IMAGE_SHAPE)},
    ],
    ids=["state_only", "state_and_env_state", "state_and_image"],
)
def test_forward_trains_the_state_encoder(extra_inputs):
    # The training horizon used to be read from the image or environment state key,
    # so a policy with only `observation.state` raised a KeyError in `forward`.
    policy = _make_policy(extra_inputs)
    loss, _ = policy.forward(_make_batch(policy))
    loss.backward()

    assert torch.isfinite(loss)
    state_encoder_grads = [p.grad for p in policy.model._encoder.state_enc_layers.parameters()]
    assert all(g is not None for g in state_encoder_grads)
    assert any(g.abs().sum() > 0 for g in state_encoder_grads)
