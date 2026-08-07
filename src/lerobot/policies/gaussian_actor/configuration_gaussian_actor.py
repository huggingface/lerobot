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
        key (`str`): The feature key to check.

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

    ``multiprocessing_context`` selects the process-wide start method when
    processes are used. Set it to ``None`` to preserve Python's default or a
    method already selected by the embedding application.
    """

    actor: str = "threads"
    learner: str = "threads"
    multiprocessing_context: str | None = "spawn"


@dataclass
class ActorLearnerConfig:
    """Actor-learner distributed architecture settings (network address, weight-push frequency)."""

    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2


@dataclass
class CriticNetworkConfig:
    """MLP architecture settings for the critic network(s)."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    final_activation: str | None = None


@dataclass
class ActorNetworkConfig:
    """MLP architecture settings for the actor network."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True


@dataclass
class PolicyConfig:
    """Gaussian-policy output-head settings (tanh squashing, std clamping)."""

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

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1):
            Number of environment steps of observation to pass to the policy (the current step and
            additional steps going back). This policy predicts a single action from a single step, so
            this is not expected to be changed from 1.
        input_features (`dict[str, PolicyFeature] | None`, *optional*):
            Mapping from input feature name to its `PolicyFeature` (type and shape). Populated
            automatically from the dataset when not explicitly provided.
        output_features (`dict[str, PolicyFeature] | None`, *optional*):
            Mapping from output feature name to its `PolicyFeature` (type and shape). Populated
            automatically from the dataset when not explicitly provided.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device the policy runs on, e.g. `"cuda"`, `"cuda:0"`, `"cpu"`, or `"mps"`.
        use_amp (`bool`, *optional*, defaults to `False`):
            Whether to use Automatic Mixed Precision for training and evaluation.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether this policy is trained with PEFT (parameter-efficient fine-tuning) adapters.
        push_to_hub (`bool`, *optional*, defaults to `True`):
            Whether to push the trained policy to the Hugging Face Hub after training.
        repo_id (`str | None`, *optional*):
            Hugging Face Hub repository id to push the policy to, when `push_to_hub` is enabled.
        private (`bool | None`, *optional*):
            Whether to create/push the Hub repository as private.
        tags (`list[str] | None`, *optional*):
            Tags to attach to the policy's Hub model card.
        license (`str | None`, *optional*):
            License identifier to add to the policy's Hub model card.
        pretrained_path (`Path | None`, *optional*):
            Path or Hub repo id of pretrained weights to initialize the policy from. If `None`, the
            policy is initialized from scratch.
        pretrained_revision (`str | None`, *optional*):
            Hub revision (branch, tag, or commit hash) pinning the pretrained model version.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps a feature type name (e.g. `"STATE"`, `"VISUAL"`) to the `NormalizationMode` to apply to
            it. Defaults to mean/std normalization for visual features and min/max normalization for
            state, environment, and action features.
        dataset_stats (`dict[str, dict[str, list[float]]] | None`, *optional*):
            Statistics used to normalize image, state, and action features. Defaults to placeholder
            values; normally overridden with statistics computed from the actual training dataset.
        storage_device (`str`, *optional*, defaults to `"cpu"`):
            Device on which a copy of the model's parameters is kept for transport between the actor and
            learner processes in the actor-learner architecture.
        vision_encoder_name (`str | None`, *optional*):
            Name of a pretrained vision encoder to use for image observations, e.g.
            `"lerobot/resnet10"` for the HIL-SERL ResNet10 encoder. `None` (the default) uses a
            lightweight from-scratch CNN encoder instead.
        freeze_vision_encoder (`bool`, *optional*, defaults to `True`):
            Whether to freeze the vision encoder's parameters during training.
        image_encoder_hidden_dim (`int`, *optional*, defaults to 32):
            Hidden dimension size for the from-scratch image encoder (unused when `vision_encoder_name`
            is set).
        shared_encoder (`bool`, *optional*, defaults to `True`):
            Whether the actor and critic(s) share the same observation encoder instance.
        num_discrete_actions (`int | None`, *optional*):
            Number of discrete actions appended to the continuous action output, e.g. for a gripper
            open/close action. `None` disables the discrete critic and action head.
        image_embedding_pooling_dim (`int`, *optional*, defaults to 8):
            Number of learned spatial pooling features per image, used by the image encoder's spatial
            embedding layer.
        state_encoder_hidden_dim (`int`, *optional*, defaults to 256):
            Hidden dimension size for the state encoder.
        latent_dim (`int`, *optional*, defaults to 256):
            Dimension of the observation encoder's output latent space.
        online_steps (`int`, *optional*, defaults to 1000000):
            Number of steps to run during online training.
        online_buffer_capacity (`int`, *optional*, defaults to 100000):
            Capacity of the online replay buffer.
        offline_buffer_capacity (`int`, *optional*, defaults to 100000):
            Capacity of the offline replay buffer.
        async_prefetch (`bool`, *optional*, defaults to `False`):
            Whether to use asynchronous prefetching for the replay buffers.
        online_step_before_learning (`int`, *optional*, defaults to 100):
            Number of steps to collect before online learning starts.
        actor_learner_config (`ActorLearnerConfig`, *optional*):
            Transport configuration (host, port, push frequency, queue timeout) for the actor-learner
            architecture.
        concurrency (`ConcurrencyConfig`, *optional*):
            Concurrency configuration (threads or processes) for the actor and learner.
        actor_network_kwargs (`ActorNetworkConfig`, *optional*):
            Architecture configuration (hidden dimensions, final activation) for the actor network.
        policy_kwargs (`PolicyConfig`, *optional*):
            Configuration for the Gaussian policy head (tanh squashing, std bounds, final-layer init
            scale).
        discrete_critic_network_kwargs (`CriticNetworkConfig`, *optional*):
            Architecture configuration (hidden dimensions, final activation) for the discrete critic
            network.
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

    # Architecture specifics
    device: str = "cpu"
    storage_device: str = "cpu"
    vision_encoder_name: str | None = None
    freeze_vision_encoder: bool = True
    image_encoder_hidden_dim: int = 32
    shared_encoder: bool = True
    num_discrete_actions: int | None = None
    image_embedding_pooling_dim: int = 8

    # Encoder architecture
    state_encoder_hidden_dim: int = 256
    latent_dim: int = 256

    # Online training (TODO(Khalil): relocate to TrainRLServerPipelineConfig)
    online_steps: int = 1000000
    online_buffer_capacity: int = 100000
    offline_buffer_capacity: int = 100000
    async_prefetch: bool = False
    online_step_before_learning: int = 100

    # Actor-learner transport (TODO(Khalil): relocate to TrainRLServerPipelineConfig).
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

    # Network architecture
    actor_network_kwargs: ActorNetworkConfig = field(default_factory=ActorNetworkConfig)
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    def __post_init__(self):
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates actor/critic network and learner configuration."""
        super().__post_init__()
        # Any validation specific to GaussianActor configuration

    def get_optimizer_preset(self) -> MultiAdamConfig:
        """See [`~configs.PreTrainedConfig.get_optimizer_preset`]."""
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
        """See [`~configs.PreTrainedConfig.get_scheduler_preset`]."""
        return None

    def validate_features(self) -> None:
        """See [`~configs.PreTrainedConfig.validate_features`]."""
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
        """The names of the input features that are images."""
        return [key for key in self.input_features if is_image_feature(key)]

    @property
    def observation_delta_indices(self) -> list:
        """See [`~configs.PreTrainedConfig.observation_delta_indices`]."""
        return None

    @property
    def action_delta_indices(self) -> list:
        """See [`~configs.PreTrainedConfig.action_delta_indices`]."""
        return None  # SAC typically predicts one action at a time

    @property
    def reward_delta_indices(self) -> None:
        """See [`~configs.PreTrainedConfig.reward_delta_indices`]."""
        return None
