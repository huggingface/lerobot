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
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
)


@PreTrainedConfig.register_subclass("conrft")
@dataclass
class ConRFTConfig(PreTrainedConfig):
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

    # Action space
    action_dim: int = 8  # override at runtime from dataset meta
    max_action: float = 1.0
    fix_gripper: bool = False

    # Consistency Policy (CP)
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    num_scales: int = 40
    time_dim: int = 256
    hidden_dim: int = 1024
    num_blocks: int = 3
    use_layer_norm: bool = True

    # Loss weights
    bc_weight: float = 1.0
    q_weight: float = 1.0
    recon_weight: float = 1.0  # CP reconstruction
    snr_clip: float = 5.0

    # Critic (ensemble + CQL/CalQL)
    critic_hidden_dim: int = 1024
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
    soft_target_update_rate: float = 0.005  # OR delete and use target_tau consistently

    # Cal-QL temperature
    cql_temp: float = 1.0

    # optional VLA init
    base_vla_model_path: str | None = None
    freeze_base_vla: bool = True

    # Vision encoder
    vision_encoder_name: str | None = "resnetv1-10-frozen"
    freeze_vision_encoder: bool = True
    shared_encoder: bool = True
    image_embedding_pooling_dim: int = 8
    latent_dim: int = 64

    # Optimization
    optim: AdamWConfig = field(
        default_factory=lambda: AdamWConfig(lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0)
    )
    scheduler: CosineDecayWithWarmupSchedulerConfig | None = None

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
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
