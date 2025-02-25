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


@dataclass
class SACConfig:
    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.image": [3, 84, 84],
            "observation.state": [4],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [2],
        }
    )
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.image": "mean_std",
            "observation.state": "min_max",
            "observation.environment_state": "min_max",
        }
    )
    input_normalization_params: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            "observation.image": {
                "mean": [[0.485, 0.456, 0.406]],
                "std": [[0.229, 0.224, 0.225]],
            },
            "observation.state": {"min": [-1, -1, -1, -1], "max": [1, 1, 1, 1]},
        }
    )
    output_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {"action": "min_max"}
    )
    output_normalization_params: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            "action": {"min": [-1, -1], "max": [1, 1]},
        }
    )
    # TODO: Move it outside of the config
    actor_learner_config: dict[str, str | int] = field(
        default_factory=lambda: {
            "learner_host": "127.0.0.1",
            "learner_port": 50051,
        }
    )
    camera_number: int = 1
    # Add type annotations for these fields:
    vision_encoder_name: str | None = field(default="helper2424/resnet10")
    freeze_vision_encoder: bool = True
    image_encoder_hidden_dim: int = 32
    shared_encoder: bool = True
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
    critic_network_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_dims": [256, 256],
            "activate_final": True,
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
            "init_final": 0.005,
        }
    )
