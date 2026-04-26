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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.policies.sac.configuration_sac import (
    ActorLearnerConfig,
    ActorNetworkConfig,
    ConcurrencyConfig,
    CriticNetworkConfig,
    PolicyConfig,
    is_image_feature,
)
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


@PreTrainedConfig.register_subclass("twinrl")
@dataclass
class TwinRLConfig(PreTrainedConfig):
    """TwinRL-VLA configuration.

    TwinRL is a digital twin–real-world collaborative RL framework for VLA fine-tuning.
    It trains with a joint IL+RL objective (ConRFT-style) and a Cal-QL critic that
    uses Monte Carlo return lower bounds to prevent Q-value underestimation.

    Paper: https://arxiv.org/abs/2602.09023
    Official repo: https://github.com/zhourui9813/TwinRL
    """

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

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

    # Architecture
    device: str = "cuda"
    storage_device: str = "cpu"
    vision_encoder_name: str | None = None
    freeze_vision_encoder: bool = True
    image_encoder_hidden_dim: int = 32
    shared_encoder: bool = True
    num_discrete_actions: int | None = None
    image_embedding_pooling_dim: int = 8

    # Training
    online_steps: int = 1000000
    online_buffer_capacity: int = 100000
    # Capacity for the offline/demo buffer (used for BC loss and Cal-QL)
    offline_buffer_capacity: int = 100000
    # Capacity for the twin rollout buffer used to seed real-world training (Step 3)
    twin_buffer_capacity: int = 10000
    async_prefetch: bool = False
    online_step_before_learning: int = 100
    policy_update_freq: int = 1

    # Discount factor (γ). Used for TD targets and MC return computation.
    discount: float = 0.95

    # Number of critics in the ensemble
    num_critics: int = 2
    num_subsample_critics: int | None = None

    # Learning rates
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4

    # Soft target update weight (τ)
    critic_target_update_weight: float = 0.005

    # Update-to-data ratio
    utd_ratio: int = 1

    state_encoder_hidden_dim: int = 256
    latent_dim: int = 256
    grad_clip_norm: float = 40.0
    use_torch_compile: bool = False

    # --- TwinRL-specific: Joint IL+RL actor loss weights ---
    # Weight for the BC/IL term (β). Default from official TwinRL repo.
    bc_weight: float = 1.0
    # Weight for the RL Q-gradient term (η). Default from official TwinRL repo.
    q_weight: float = 0.1

    # --- Cal-QL critic hyperparameters ---
    # Weight for the CQL penalty (α)
    cql_alpha: float = 0.1
    # Number of random/policy actions to sample for CQL OOD penalty
    cql_n_actions: int = 10
    # Temperature for CQL logsumexp
    cql_temp: float = 1.0
    # Whether to apply MC return lower bound clipping (Cal-QL)
    use_calql: bool = True
    # Per-element clamp bounds on cql_diff before taking the mean (official default: no clamp)
    cql_clip_diff_min: float = -float("inf")
    cql_clip_diff_max: float = float("inf")

    # --- ConRFT / Consistency policy ---
    # "sac" = SACObservationEncoder for actor; "octo" = frozen OctoTransformer
    actor_encoder_type: str = "octo"
    # HuggingFace repo for the octo-pytorch model (used when actor_encoder_type="octo")
    octo_model_name: str = "lilkm/octo-small-test"
    # When True, use Karras consistency model instead of Gaussian policy
    use_consistency_policy: bool = True
    # Karras noise schedule hyperparameters (match official TwinRL defaults)
    num_scales: int = 40
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    rho: float = 7.0
    # Timestep embedding dimension for the consistency t_network
    consistency_t_dim: int = 16

    # Network architecture configs (reuse SAC sub-configs)
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    optimizer_lr: float = 3e-4
    optimizer_weight_decay: float = 0.0
    optimizer_lr_scheduler: str = "cosine"

    def __post_init__(self):
        super().__post_init__()

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.actor_lr,
            weight_decay=0.0,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        has_image = any(is_image_feature(key) for key in self.input_features)
        has_state = OBS_STATE in self.input_features

        if not (has_state or has_image):
            raise ValueError(
                "You must provide either 'observation.state' or an image observation "
                "(key starting with 'observation.image') in the input features"
            )

        if ACTION not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self) -> list | None:
        return [0, 1]

    @property
    def action_delta_indices(self) -> list | None:
        return [0]

    @property
    def reward_delta_indices(self) -> list | None:
        return [0]
