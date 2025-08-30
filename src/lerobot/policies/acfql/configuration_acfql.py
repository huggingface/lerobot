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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.optim.optimizers import MultiAdamConfig


def is_image_feature(key: str) -> bool:
    """Check if a feature key represents an image feature.

    Args:
        key: The feature key to check

    Returns:
        True if the key represents an image feature, False otherwise
    """
    return key.startswith(OBS_IMAGE)


@dataclass
class ConcurrencyConfig:
    """Configuration for the concurrency of the actor and learner.
    Possible values are:
    - "threads": Use threads for the actor and learner.
    - "processes": Use processes for the actor and learner.
    """

    actor: str = "threads"
    learner: str = "threads"


@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2


@dataclass
class CriticNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    final_activation: str | None = None
    layer_norm: bool = True
    default_init: float | None = None
    init_final: float | None = None


@dataclass
class ActorNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = False
    layer_norm: bool = False
    default_init: float | None = None


@dataclass
class PolicyConfig:
    # use_tanh_squash: bool = True
    # std_min: float = 1e-5
    # std_max: float = 10.0
    # init_final: float = 0.05
    init_final: float | None = None


@dataclass
class DiscreteActorNetworkConfig:
    """Configuration for the discrete actor network."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    layer_norm: bool = False


@dataclass
class DiscretePolicyConfig:
    """Configuration for the discrete policy."""

    init_final: float | None = None


@PreTrainedConfig.register_subclass("acfql")
@dataclass
class ACFQLConfig(PreTrainedConfig):
    """Soft Actor-Critic (SAC) configuration.

    SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
    reinforcement learning framework. It learns a policy and a Q-function simultaneously
    using experience collected from the environment.

    This configuration class contains all the parameters needed to define a SAC agent,
    including network architectures, optimization settings, and algorithm-specific
    hyperparameters.
    """

    chunk_size: int = 10

    # Mapping of feature types to normalization modes
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
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

    # Architecture specifics
    # Device to run the model on (e.g., "cuda", "cpu")
    device: str = "cpu"
    # Device to store the model on
    storage_device: str = "cpu"
    # Name of the vision encoder model (Set to "helper2424/resnet10" for hil serl resnet10)
    vision_encoder_name: str | None = None
    # Whether to freeze the vision encoder during training
    freeze_vision_encoder: bool = True
    # Hidden dimension size for the image encoder
    image_encoder_hidden_dim: int = 32
    # Whether to use a shared encoder for actor and critic
    shared_encoder: bool = True
    # Number of discrete actions, eg for gripper actions
    num_discrete_actions: int | None = None
    # Dimension of the image embedding pooling
    image_embedding_pooling_dim: int = 8

    # Training parameter
    # Number of steps for online training
    online_steps: int = 1000000
    # Number of steps for pretraining (if applicable)
    pretrain_steps: int = 0
    # Whether to reset critics after pretraining
    reset_critics_after_pretraining: bool = False
    # Seed for the online environment
    online_env_seed: int = 10000
    # Capacity of the online replay buffer
    online_buffer_capacity: int = 100000
    # Capacity of the offline replay buffer
    offline_buffer_capacity: int = 100000
    # Whether to use asynchronous prefetching for the buffers
    async_prefetch: bool = False
    # Number of steps before learning starts
    online_step_before_learning: int = 100
    # Frequency of policy updates
    policy_update_freq: int = 1

    force_full_n_steps: bool = True
    use_terminal_for_next_state: bool = True
    mask_truncated_td_loss: bool = False

    # SAC algorithm parameters
    # Discount factor for the SAC algorithm
    discount: float = 0.99
    # Initial temperature value
    temperature_init: float = 1.0
    # Number of critics in the ensemble
    num_critics: int = 2
    # Number of subsampled critics for training
    num_subsample_critics: int | None = None
    # Learning rate for the critic network
    critic_lr: float = 3e-4
    # Learning rate for the actor network
    actor_lr: float = 3e-4
    # Learning rate for the temperature parameter
    temperature_lr: float = 3e-4
    # Weight for the critic target update
    critic_target_update_weight: float = 0.005
    # Aggregation method for Q-values, can be "mean" or "max"
    q_agg: str = "mean"
    # Weight for the alpha parameter in the SAC algorithm
    alpha: float = 10.0
    # Number of steps for the flow in the SAC algorithm
    flow_steps: int = 10
    # Whether to normalize the Q-loss
    normalize_q_loss: bool = False
    # Update-to-data ratio for the UTD algorithm (If you want enable utd_ratio, you need to set it to >1)
    utd_ratio: int = 1
    # Hidden dimension size for the state encoder
    state_encoder_hidden_dim: int = 256
    # Dimension of the latent space
    latent_dim: int = 256
    # Target entropy for the SAC algorithm
    target_entropy: float | None = None
    # Whether to use backup entropy for the SAC algorithm
    use_backup_entropy: bool = True
    # Gradient clipping norm for the SAC algorithm
    grad_clip_norm: float = 40.0

    # Cal-QL parameters
    # Whether to use Cal-QL (Calibrated Q-Learning)
    use_calql: bool = False
    # Whether to apply Cal-QL bounds to random actions (if False, only applies to policy actions)
    calql_bound_random_actions: bool = False
    # Number of actions to sample for CQL/Cal-QL
    cql_n_actions: int = 10
    # CQL temperature parameter
    cql_temp: float = 1.0
    # CQL alpha weight for the CQL loss
    cql_alpha: float = 5.0
    # Whether to use CQL loss
    use_cql_loss: bool = False
    # Whether to use TD loss (can be disabled for pure CQL)
    use_td_loss: bool = True
    # CQL action sampling method: "uniform" or "normal"
    cql_action_sample_method: str = "uniform"
    # Whether to use importance sampling in CQL
    cql_importance_sample: bool = False
    # Whether to use max target backup in CQL
    cql_max_target_backup: bool = False
    # CQL clipping bounds for the Q-value difference
    cql_clip_diff_min: float = -float("inf")
    cql_clip_diff_max: float = float("inf")
    # CQL target action gap for auto-tuning
    cql_target_action_gap: float = 1.0
    # Whether to auto-tune CQL alpha
    cql_autotune_alpha: bool = False
    # Initial value for CQL alpha Lagrange multiplier
    cql_alpha_lagrange_init: float = 1.0
    cql_use_bc_for_sampling: bool = False

    # Network configuration
    # Configuration for the critic network architecture
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for the actor network architecture
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    # Configuration for the discrete actor network architecture
    discrete_actor_network_kwargs: DiscreteActorNetworkConfig = field(
        default_factory=DiscreteActorNetworkConfig
    )
    # Configuration for the policy parameters
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    # Configuration for the discrete policy
    discrete_policy_kwargs: DiscretePolicyConfig = field(default_factory=DiscretePolicyConfig)
    # Configuration for the discrete critic network
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for actor-learner architecture
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    # Configuration for concurrency settings (you can use threads or processes for the actor and learner)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Optimizations
    use_torch_compile: bool = True

    def __post_init__(self):
        super().__post_init__()
        # Any validation specific to SAC configuration

    def get_optimizer_preset(self) -> MultiAdamConfig:
        optimizer_groups = {
            "actor_bc_flow": {"lr": self.actor_lr},
            "actor_onestep_flow": {"lr": self.actor_lr},
            "critic": {"lr": self.critic_lr},
            "temperature": {"lr": self.temperature_lr},
        }

        # Add CQL alpha optimizer if auto-tuning is enabled
        if self.cql_autotune_alpha:
            optimizer_groups["cql_alpha_lagrange"] = {"lr": self.critic_lr}  # Use same lr as critic

        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups=optimizer_groups,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
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
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return None  # SAC typically predicts one action at a time

    @property
    def reward_delta_indices(self) -> None:
        return None
