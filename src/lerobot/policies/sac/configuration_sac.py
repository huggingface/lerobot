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
from lerobot.optim.optimizers import MultiAdamConfig
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
    # Either a scalar (applied to all action dims) or a list of length
    # action_dim for per-dim bounds (e.g. tighter std on the gripper to
    # suppress random open/close jitter while keeping wide exploration on xyz/yaw).
    std_min: float | list[float] = 1e-5
    std_max: float | list[float] = 10.0
    init_final: float = 0.05


@PreTrainedConfig.register_subclass("sac")
@dataclass
class SACConfig(PreTrainedConfig):
    """Soft Actor-Critic (SAC) configuration.

    SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
    reinforcement learning framework. It learns a policy and a Q-function simultaneously
    using experience collected from the environment.

    This configuration class contains all the parameters needed to define a SAC agent,
    including network architectures, optimization settings, and algorithm-specific
    hyperparameters.
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

    # Network configuration
    # Configuration for the critic network architecture
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for the actor network architecture
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    # Configuration for the policy parameters
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    # Configuration for the discrete critic network
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for actor-learner architecture
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    # Configuration for concurrency settings (you can use threads or processes for the actor and learner)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Optimizations
    use_torch_compile: bool = True

    # ---- RABC-weighted BC auxiliary loss (paper SARM eq8-9) ----
    # When > 0, an auxiliary `-w_i * log_prob(a_demo|o_demo)` term is added to
    # the actor loss. Demo (obs, actions) sourced from the offline replay buffer.
    bc_loss_weight: float = 0.0
    # When True, use RABC weights from precomputed SARM-progress parquet.
    # When False but bc_loss_weight>0, BC is uniform-weighted.
    bc_use_rabc: bool = False
    bc_rabc_progress_path: str | None = None
    bc_rabc_chunk_size: int = 50
    bc_rabc_head_mode: str = "sparse"
    bc_rabc_kappa: float = 0.01
    # Linearly anneal bc_loss_weight to `bc_loss_weight_final` over `bc_anneal_steps`.
    bc_loss_weight_final: float = 0.0
    bc_anneal_steps: int = 0  # 0 = no annealing

    # ---- Residual SAC (ResFiT-style) ----
    # When True, freeze a base policy and have the SAC actor produce a small
    # residual added to its output. See docs/port/2026-04-29-residual-hilserl-design.md.
    residual_mode: bool = False
    base_policy_path: str | None = None  # path to pretrained_model dir for ACT/Diffusion
    base_policy_type: str = "act"  # "act" | "diffusion"
    # Residual magnitude in normed action space. Either a scalar (applied to all
    # dims) or a list of length action_dim (per-dim scale). Per-dim is critical
    # for chunked bases with discrete-ish gripper: the gripper dim needs scale
    # >= 1.0 to be able to flip the base policy's gripper command across the
    # post-residual deadband, while xyz/yaw should stay small (~0.1) for
    # stability.
    residual_action_scale: float | list[float] = 0.1
    freeze_base_policy: bool = True
    # Critic-only warmup: for the first N opt steps the TD target uses the
    # frozen base policy's action (δ=0) instead of resampling from the SAC
    # actor. Critic learns Q^{π_base} grounded in base-policy actions; actor
    # and temperature updates are skipped during warmup. Set 0 to disable.
    critic_warmup_steps: int = 0
    # Optional: clip target Q to reward range to mitigate overestimation w/ dense reward.
    clip_q_target_to_reward_range: bool = False
    q_target_clip_min: float = 0.0
    q_target_clip_max: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        # Any validation specific to SAC configuration

    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": self.actor_lr},
                "critic": {"lr": self.critic_lr},
                "temperature": {"lr": self.temperature_lr},
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

        # Residual mode: register observation.base_action as a state-like input
        # feature of size action_dim and provide identity (min=-1,max=1) stats so
        # the input normalizer is a no-op on the (already normed) base action.
        if self.residual_mode:
            from lerobot.configs.types import FeatureType, PolicyFeature

            action_shape = self.output_features[ACTION].shape
            self.input_features.setdefault(
                "observation.base_action",
                PolicyFeature(type=FeatureType.STATE, shape=action_shape),
            )
            if self.dataset_stats is None:
                self.dataset_stats = {}
            self.dataset_stats.setdefault(
                "observation.base_action",
                {
                    "min": [-1.0] * action_shape[0],
                    "max": [1.0] * action_shape[0],
                },
            )

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
