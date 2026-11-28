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
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def make_policy(use_vae=True):
    return ACTPolicy(
        ACTConfig(
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
                OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
            },
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
            chunk_size=4,
            n_action_steps=4,
            use_vae=use_vae,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            n_vae_encoder_layers=1,
            latent_dim=4,
            dropout=0,
            device="cpu",
        )
    )


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("use_vae", [False, True])
def test_single_state_timestep_preserves_training_loss_and_gradients(batch_size, use_vae):
    policy = make_policy(use_vae)
    stats = {
        key: {"mean": torch.zeros(feature.shape), "std": torch.ones(feature.shape)}
        for key, feature in {**policy.config.input_features, **policy.config.output_features}.items()
    }
    preprocessor, _ = make_act_pre_post_processors(policy.config, stats)
    batch = preprocessor(
        {
            OBS_STATE: torch.randn(batch_size, 3),
            OBS_ENV_STATE: torch.randn(batch_size, 4),
            ACTION: torch.randn(batch_size, 4, 2),
            "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
        }
    )

    # Reuse the same latent sample when the VAE is enabled.
    with torch.random.fork_rng():
        loss, _ = policy(batch)
        loss.backward()
    expected_grads = {
        name: parameter.grad.clone()
        for name, parameter in policy.named_parameters()
        if parameter.grad is not None
    }
    policy.zero_grad(set_to_none=True)

    state_with_time = batch[OBS_STATE].unsqueeze(1)
    temporal_batch = {**batch, OBS_STATE: state_with_time}
    temporal_loss, _ = policy(temporal_batch)
    temporal_loss.backward()

    torch.testing.assert_close(temporal_loss, loss)
    for name, parameter in policy.named_parameters():
        if name in expected_grads:
            torch.testing.assert_close(parameter.grad, expected_grads[name])
    assert temporal_batch[OBS_STATE] is state_with_time
    assert state_with_time.shape == (batch_size, 1, 3)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_single_state_timestep_preserves_action_chunk(batch_size):
    policy = make_policy()
    batch = {OBS_STATE: torch.randn(batch_size, 3), OBS_ENV_STATE: torch.randn(batch_size, 4)}
    expected = policy.predict_action_chunk(batch)
    actual = policy.predict_action_chunk({**batch, OBS_STATE: batch[OBS_STATE].unsqueeze(1)})
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (batch_size, 4, 2)


def test_single_state_timestep_trains_with_visual_observations():
    config = make_policy().config
    del config.input_features[OBS_ENV_STATE]
    config.input_features["observation.images.camera"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, 32, 32)
    )
    config.pretrained_backbone_weights = None
    policy = ACTPolicy(config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    batch = {
        OBS_STATE: torch.randn(1, 1, 3),
        "observation.images.camera": torch.randn(1, 3, 32, 32),
        ACTION: torch.randn(1, 4, 2),
        "action_is_pad": torch.zeros(1, 4, dtype=torch.bool),
    }
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert policy.model.vae_encoder_robot_state_input_proj.weight.grad.abs().sum() > 0
    optimizer.step()
