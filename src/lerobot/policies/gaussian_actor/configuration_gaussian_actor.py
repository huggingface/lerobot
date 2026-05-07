#!/usr/bin/env python

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

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import MultiAdamConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


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


@dataclass
class ActorNetworkConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True


@dataclass
class PolicyConfig:
    use_tanh_squash: bool = True
    std_min: float = 1e-5
    std_max: float = 10.0
    init_final: float = 0.05


@PreTrainedConfig.register_subclass("gaussian_actor")
@dataclass
class GaussianActorConfig(PreTrainedConfig):
    """Gaussian actor configuration.

    This configures the policy-side (actor + observation encoder) of a Gaussian
    policy, as used by SAC and related maximum-entropy continuous-control algorithms.
    By default the actor output is a tanh-squashed diagonal Gaussian
    (``TanhMultivariateNormalDiag``); the tanh squashing can be disabled via
    ``policy_kwargs.use_tanh_squash``. The critics, temperature, and Bellman-update
    logic live on the algorithm side (see ``lerobot.rl.algorithms.sac``).

    CLI: ``--policy.type=gaussian_actor``.
    """

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
    # Name of the vision encoder model (Set to "lerobot/resnet10" for hil serl resnet10)
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

    # Encoder architecture
    # Hidden dimension size for the state encoder
    state_encoder_hidden_dim: int = 256
    # Dimension of the latent space
    latent_dim: int = 256

    # Online training (TODO(Khalil): relocate to TrainRLServerPipelineConfig)
    # Number of steps for online training
    online_steps: int = 1000000
    # Capacity of the online replay buffer
    online_buffer_capacity: int = 100000
    # Capacity of the offline replay buffer
    offline_buffer_capacity: int = 100000
    # Whether to use asynchronous prefetching for the buffers
    async_prefetch: bool = False
    # Number of steps before learning starts
    online_step_before_learning: int = 100

    # Actor-learner transport (TODO(Khalil): relocate to TrainRLServerPipelineConfig).
    # Configuration for actor-learner architecture
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    # Configuration for concurrency settings (you can use threads or processes for the actor and learner)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Network architecture
    # Configuration for the actor network architecture
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    # Configuration for the policy parameters (Gaussian head)
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    # Configuration for the discrete critic network
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    def __post_init__(self):
        super().__post_init__()
        # Any validation specific to GaussianActor configuration

    def get_optimizer_preset(self) -> MultiAdamConfig:
        # Default learning rate used to satisfy the abstract ``get_optimizer_preset()``
        # contract from ``PreTrainedConfig``. The actual optimizers used during RL
        # training are built by ``SACAlgorithm.make_optimizers_and_scheduler()`` from
        # ``SACAlgorithmConfig.{actor_lr,critic_lr,temperature_lr}`` and fully bypass
        # this preset.
        default_lr = 3e-4
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": default_lr},
                "critic": {"lr": default_lr},
                "temperature": {"lr": default_lr},
            },
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

        if ACTION not in self.output_features:
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
