# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.optim.optimizers import MultiAdamConfig

from lerobot.policies.sac.configuration_sac import (
    ActorLearnerConfig,
    ConcurrencyConfig,
    CriticNetworkConfig,
)


def is_image_feature(key: str) -> bool:
    """Check if a feature key represents an image feature.

    Args:
        key: The feature key to check

    Returns:
        True if the key represents an image feature, False otherwise
    """
    return key.startswith(OBS_IMAGE)


@PreTrainedConfig.register_subclass("conrft")
@dataclass
class ConRFTConfig(PreTrainedConfig):
    # TODO(lilkm): add docstring

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Statistics for normalizing different types of inputs
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
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
    )

    # Feature definitions (same as Octo)
    input_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "observation.image_primary": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            "observation.image_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        }
    )
    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    )

    # Octo backbone
    model_name: str = "octo-base"
    token_embedding_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    chunk_size: int = 10  # max horizon

    # IO structure
    n_obs_steps: int = 1
    n_action_steps: int = 4

    # Consistency Policy (CP)
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    num_scales: int = 40
    time_dim: int = 16
    hidden_dim: int = 1024
    num_blocks: int = 3
    use_layer_norm: bool = True
    clip_denoised: bool = True

    consistency_hidden_dim: int = 256
    offline_steps: int = 2000  # Number of offline pretraining steps

    # Loss weights
    bc_weight: float = 1.0
    q_weight: float = 1.0
    recon_weight: float = 1.0  # CP reconstruction
    snr_clip: float = 5.0

    # Critic (ensemble + CQL/CalQL)
    critic_hidden_dim: int = 256
    critic_ensemble_size: int = 2
    critic_subsample_size: Optional[int] = None
    discount: float = 0.99

    cql_alpha: float = 1.0
    cql_n_actions: int = 10
    cql_action_sample_method: str = "uniform"
    cql_clip_diff_min: float = -np.inf
    cql_clip_diff_max: float = np.inf

    # stage weights
    bc_weight_offline: float = 1.0
    q_weight_offline: float = 1.0
    bc_weight_online: float = 0.5
    q_weight_online: float = 1.0

    # target update
    soft_target_update_rate: float = 0.005

    # Cal-QL temperature
    cql_temp: float = 1.0

    # Required VLA model (ConRFT always uses Octo)
    base_vla_model_path: str = "lerobot/octo_base"
    freeze_base_vla: bool = True

    # Proprioception settings
    use_proprio: bool = True
    proprio_latent_dim: int = 64

    vision_encoder_name: str | None = "helper2424/resnet10"
    freeze_vision_encoder: bool = True
    image_encoder_hidden_dim: int = 32
    shared_encoder: bool = True
    image_embedding_pooling_dim: int = 8
    latent_dim: int = 64

    # Storage device
    storage_device: str = "cpu"
    use_amp: bool = False

    online_steps: int = 1000000
    online_env_seed: int = 10000
    online_buffer_capacity: int = 100000
    offline_buffer_capacity: int = 100000
    async_prefetch: bool = False
    online_step_before_learning: int = 100
    policy_update_freq: int = 1
    utd_ratio: int = 2
    state_encoder_hidden_dim: int = 256
    grad_clip_norm: float = 10.0

    num_discrete_actions: int | None = None

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Network configurations
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    # TODO(lilkm): add config for consistency policy

    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)

    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Optimizations
    use_torch_compile: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Any validation specific to ConRFT configuration

    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": 3e-4},
                "critic": {"lr": 3e-4},
            },
        )

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        has_image = any(is_image_feature(key) for key in self.input_features)
        has_state = OBS_STATE in self.input_features

        if not (has_state or has_image):
            raise ValueError(
                "You must provide either 'observation.state' or an image observation (key starting with 'observation.image') in the input features"
            )

        if "action" not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self):
        """ConRFT doesn't use observation deltas"""
        return None

    @property
    def action_delta_indices(self):
        """ConRFT doesn't use action deltas"""
        return None

    @property
    def reward_delta_indices(self):
        """ConRFT doesn't use reward deltas"""
        return None
