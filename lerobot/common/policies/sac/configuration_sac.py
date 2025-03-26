#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
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
from typing import Any

from lerobot.common.optim.optimizers import MultiAdamConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("sac")
@dataclass
class SACConfig(PreTrainedConfig):
    """Configuration class for Soft Actor-Critic (SAC) policy.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        normalization_mapping: Mapping from feature types to normalization modes.
        dataset_stats: Statistics for normalizing different data types.
        camera_number: Number of cameras to use.
        device: Device to use for training.
        storage_device: Device to use for storage.
        vision_encoder_name: Name of the vision encoder to use.
        freeze_vision_encoder: Whether to freeze the vision encoder.
        image_encoder_hidden_dim: Hidden dimension for the image encoder.
        shared_encoder: Whether to use a shared encoder.
        online_steps: Total number of online training steps.
        online_env_seed: Seed for the online environment.
        online_buffer_capacity: Capacity of the online replay buffer.
        online_step_before_learning: Number of steps to collect before starting learning.
        policy_update_freq: Frequency of policy updates.
        discount: Discount factor for the RL algorithm.
        temperature_init: Initial temperature for entropy regularization.
        num_critics: Number of critic networks.
        num_subsample_critics: Number of critics to subsample.
        critic_lr: Learning rate for critic networks.
        actor_lr: Learning rate for actor network.
        temperature_lr: Learning rate for temperature parameter.
        critic_target_update_weight: Weight for soft target updates.
        utd_ratio: Update-to-data ratio (>1 to enable).
        state_encoder_hidden_dim: Hidden dimension for state encoder.
        latent_dim: Dimension of latent representation.
        target_entropy: Target entropy for automatic temperature tuning.
        use_backup_entropy: Whether to use backup entropy.
        grad_clip_norm: Gradient clipping norm.
        critic_network_kwargs: Additional arguments for critic networks.
        actor_network_kwargs: Additional arguments for actor network.
        policy_kwargs: Additional arguments for policy.
        actor_learner_config: Configuration for actor-learner communication.
        concurrency: Configuration for concurrency model.
    """

    # Input / output structure
    n_obs_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    dataset_stats: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            "observation.image": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "observation.state": {
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
            "action": {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 1.0],
            },
        }
    )

    # Architecture specifics
    camera_number: int = 1
    device: str = "cuda"
    storage_device: str = "cpu"
    # Set to "helper2424/resnet10" for hil serl 
    vision_encoder_name: str | None = None 
    freeze_vision_encoder: bool = True
    image_encoder_hidden_dim: int = 32
    shared_encoder: bool = True

    # Training parameter
    online_steps: int = 1000000
    online_env_seed: int = 10000
    online_buffer_capacity: int = 10000
    online_step_before_learning: int = 100 
    policy_update_freq: int = 1

    # SAC algorithm parameters
    discount: float = 0.99
    temperature_init: float = 1.0
    num_critics: int = 2
    num_subsample_critics: int | None = None
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    temperature_lr: float = 3e-4
    critic_target_update_weight: float = 0.005
    utd_ratio: int = 1  # If you want enable utd_ratio, you need to set it to >1
    state_encoder_hidden_dim: int = 256
    latent_dim: int = 256
    target_entropy: float | None = None
    use_backup_entropy: bool = True
    grad_clip_norm: float = 40.0
    
    # Network configuration
    critic_network_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "final_activation": None,
        }
    )
    actor_network_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_dims": [256, 256],
            "activate_final": True,
        }
    )
    policy_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "use_tanh_squash": True,
            "log_std_min": -5,
            "log_std_max": 2,
            "init_final": 0.05,
        }
    )
    
    actor_learner_config: dict[str, str | int] = field(
        default_factory=lambda: {
            "learner_host": "127.0.0.1",
            "learner_port": 50051,
            "policy_parameters_push_frequency": 4,
        }
    )
    concurrency: dict[str, str] = field(
        default_factory=lambda: {
            "actor": "threads",
            "learner": "threads"
        }
    )

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
        # TODO: Maybe we should remove this raise?
        if len(self.image_features) == 0:
            raise ValueError("You must provide at least one image among the inputs.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return [0]  # SAC typically predicts one action at a time

    @property
    def reward_delta_indices(self) -> None:
        return None

if __name__ == "__main__":
    import draccus
    config = SACConfig()
    draccus.set_config_type("json")
    draccus.dump(config=config, stream=open(file='run_config.json', mode='w'), )
    
