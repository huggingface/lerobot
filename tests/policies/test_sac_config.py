#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.sac.configuration_sac import (
    ActorLearnerConfig,
    ActorNetworkConfig,
    ConcurrencyConfig,
    CriticNetworkConfig,
    PolicyConfig,
    SACConfig,
)
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def test_sac_config_default_initialization():
    config = SACConfig()

    assert config.normalization_mapping == {
        "VISUAL": NormalizationMode.MEAN_STD,
        "STATE": NormalizationMode.MIN_MAX,
        "ENV": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
    }
    assert config.dataset_stats == {
        OBS_IMAGE: {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        OBS_STATE: {
            "min": [0.0, 0.0],
            "max": [1.0, 1.0],
        },
        ACTION: {
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0],
        },
    }

    # Basic parameters
    assert config.device == "cpu"
    assert config.storage_device == "cpu"
    assert config.discount == 0.99
    assert config.temperature_init == 1.0
    assert config.num_critics == 2

    # Architecture specifics
    assert config.vision_encoder_name is None
    assert config.freeze_vision_encoder is True
    assert config.image_encoder_hidden_dim == 32
    assert config.shared_encoder is True
    assert config.num_discrete_actions is None
    assert config.image_embedding_pooling_dim == 8

    # Training parameters
    assert config.online_steps == 1000000
    assert config.online_buffer_capacity == 100000
    assert config.offline_buffer_capacity == 100000
    assert config.async_prefetch is False
    assert config.online_step_before_learning == 100
    assert config.policy_update_freq == 1

    # SAC algorithm parameters
    assert config.num_subsample_critics is None
    assert config.critic_lr == 3e-4
    assert config.actor_lr == 3e-4
    assert config.temperature_lr == 3e-4
    assert config.critic_target_update_weight == 0.005
    assert config.utd_ratio == 1
    assert config.state_encoder_hidden_dim == 256
    assert config.latent_dim == 256
    assert config.target_entropy is None
    assert config.use_backup_entropy is True
    assert config.grad_clip_norm == 40.0

    # Dataset stats defaults
    expected_dataset_stats = {
        OBS_IMAGE: {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        OBS_STATE: {
            "min": [0.0, 0.0],
            "max": [1.0, 1.0],
        },
        ACTION: {
            "min": [0.0, 0.0, 0.0],
            "max": [1.0, 1.0, 1.0],
        },
    }
    assert config.dataset_stats == expected_dataset_stats

    # Critic network configuration
    assert config.critic_network_kwargs.hidden_dims == [256, 256]
    assert config.critic_network_kwargs.activate_final is True
    assert config.critic_network_kwargs.final_activation is None

    # Actor network configuration
    assert config.actor_network_kwargs.hidden_dims == [256, 256]
    assert config.actor_network_kwargs.activate_final is True

    # Policy configuration
    assert config.policy_kwargs.use_tanh_squash is True
    assert config.policy_kwargs.std_min == 1e-5
    assert config.policy_kwargs.std_max == 10.0
    assert config.policy_kwargs.init_final == 0.05

    # Discrete critic network configuration
    assert config.discrete_critic_network_kwargs.hidden_dims == [256, 256]
    assert config.discrete_critic_network_kwargs.activate_final is True
    assert config.discrete_critic_network_kwargs.final_activation is None

    # Actor learner configuration
    assert config.actor_learner_config.learner_host == "127.0.0.1"
    assert config.actor_learner_config.learner_port == 50051
    assert config.actor_learner_config.policy_parameters_push_frequency == 4

    # Concurrency configuration
    assert config.concurrency.actor == "threads"
    assert config.concurrency.learner == "threads"

    assert isinstance(config.actor_network_kwargs, ActorNetworkConfig)
    assert isinstance(config.critic_network_kwargs, CriticNetworkConfig)
    assert isinstance(config.policy_kwargs, PolicyConfig)
    assert isinstance(config.actor_learner_config, ActorLearnerConfig)
    assert isinstance(config.concurrency, ConcurrencyConfig)


def test_critic_network_kwargs():
    config = CriticNetworkConfig()
    assert config.hidden_dims == [256, 256]
    assert config.activate_final is True
    assert config.final_activation is None


def test_actor_network_kwargs():
    config = ActorNetworkConfig()
    assert config.hidden_dims == [256, 256]
    assert config.activate_final is True


def test_policy_kwargs():
    config = PolicyConfig()
    assert config.use_tanh_squash is True
    assert config.std_min == 1e-5
    assert config.std_max == 10.0
    assert config.init_final == 0.05


def test_actor_learner_config():
    config = ActorLearnerConfig()
    assert config.learner_host == "127.0.0.1"
    assert config.learner_port == 50051
    assert config.policy_parameters_push_frequency == 4


def test_concurrency_config():
    config = ConcurrencyConfig()
    assert config.actor == "threads"
    assert config.learner == "threads"


def test_sac_config_custom_initialization():
    config = SACConfig(
        device="cpu",
        discount=0.95,
        temperature_init=0.5,
        num_critics=3,
    )

    assert config.device == "cpu"
    assert config.discount == 0.95
    assert config.temperature_init == 0.5
    assert config.num_critics == 3


def test_validate_features():
    config = SACConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    config.validate_features()


def test_validate_features_missing_observation():
    config = SACConfig(
        input_features={"wrong_key": PolicyFeature(type=FeatureType.STATE, shape=(10,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    with pytest.raises(
        ValueError, match="You must provide either 'observation.state' or an image observation"
    ):
        config.validate_features()


def test_validate_features_missing_action():
    config = SACConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,))},
        output_features={"wrong_key": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    with pytest.raises(ValueError, match="You must provide 'action' in the output features"):
        config.validate_features()
