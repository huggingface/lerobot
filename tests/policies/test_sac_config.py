#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.common.policies.sac.configuration_sac import (
    ActorLearnerConfig,
    ActorNetworkConfig,
    ConcurrencyConfig,
    CriticNetworkConfig,
    PolicyConfig,
    SACConfig,
)
from lerobot.configs.types import NormalizationMode


def test_sac_config_default_initialization():
    config = SACConfig()
    assert config.device == "cuda"
    assert config.storage_device == "cpu"
    assert config.discount == 0.99
    assert config.temperature_init == 1.0
    assert config.num_critics == 2


def test_sac_config_custom_initialization():
    custom_config = SACConfig(
        device="cpu",
        discount=0.95,
        temperature_init=0.5,
        num_critics=3,
    )
    assert custom_config.device == "cpu"
    assert custom_config.discount == 0.95
    assert custom_config.temperature_init == 0.5
    assert custom_config.num_critics == 3


def test_normalization_mapping():
    config = SACConfig()
    expected_mapping = {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MIN_MAX,
        "ENV": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }
    assert config.normalization_mapping == expected_mapping


def test_get_optimizer_preset():
    config = SACConfig(
        actor_lr=1e-4,
        critic_lr=2e-4,
        temperature_lr=3e-4,
    )
    optimizer_config = config.get_optimizer_preset()

    assert optimizer_config.weight_decay == 0.0
    assert optimizer_config.optimizer_groups["actor"]["lr"] == 1e-4
    assert optimizer_config.optimizer_groups["critic"]["lr"] == 2e-4
    assert optimizer_config.optimizer_groups["temperature"]["lr"] == 3e-4


def test_validate_features_with_valid_state():
    config = SACConfig()
    config.input_features = {"observation.state": {"shape": (10,), "type": "float32"}}
    config.output_features = {"action": {"shape": (3,), "type": "float32"}}
    config.validate_features()  # Should not raise exception


def test_validate_features_with_valid_image():
    config = SACConfig()
    config.input_features = {"observation.image_0": {"shape": (3, 224, 224), "type": "float32"}}
    config.output_features = {"action": {"shape": (3,), "type": "float32"}}
    config.validate_features()  # Should not raise exception


def test_validate_features_with_missing_observation():
    config = SACConfig()
    config.input_features = {"wrong_key": {"shape": (10,), "type": "float32"}}
    config.output_features = {"action": {"shape": (3,), "type": "float32"}}

    with pytest.raises(
        ValueError, match="You must provide either 'observation.state' or an image observation"
    ):
        config.validate_features()


def test_validate_features_with_missing_action():
    config = SACConfig()
    config.input_features = {"observation.state": {"shape": (10,), "type": "float32"}}
    config.output_features = {"wrong_key": {"shape": (3,), "type": "float32"}}

    with pytest.raises(ValueError, match="You must provide 'action' in the output features"):
        config.validate_features()


def test_image_features_property():
    config = SACConfig()
    config.input_features = {
        "observation.image_0": {"shape": (3, 224, 224), "type": "float32"},
        "observation.image_1": {"shape": (3, 224, 224), "type": "float32"},
        "observation.state": {"shape": (10,), "type": "float32"},
    }

    expected_image_features = ["observation.image_0", "observation.image_1"]
    assert config.image_features == expected_image_features


def test_delta_indices_properties():
    config = SACConfig()
    assert config.observation_delta_indices is None
    assert config.action_delta_indices is None
    assert config.reward_delta_indices is None


def test_network_configs():
    config = SACConfig()

    assert isinstance(config.actor_network_kwargs, ActorNetworkConfig)
    assert isinstance(config.critic_network_kwargs, CriticNetworkConfig)
    assert isinstance(config.policy_kwargs, PolicyConfig)
    assert isinstance(config.actor_learner_config, ActorLearnerConfig)
    assert isinstance(config.concurrency, ConcurrencyConfig)

    # Test default network dimensions
    assert config.actor_network_kwargs.hidden_dims == [256, 256]
    assert config.critic_network_kwargs.hidden_dims == [256, 256]


def test_scheduler_preset():
    config = SACConfig()
    assert config.get_scheduler_preset() is None
