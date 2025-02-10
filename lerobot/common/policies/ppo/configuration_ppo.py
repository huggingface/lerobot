#!/usr/bin/env python

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
from typing import Literal

from lerobot.common.optim.optimizers import AdamConfig
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("ppo")
@dataclass
class PPOConfig(PreTrainedConfig):
    """Configuration class for the Proximal Policy Optimization (PPO) policy.

    Args:
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [1], indicating 1-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        hidden_dim: A list of integers representing the hidden dimensions of the policy networks.
        gamma: The discount factor for the rewards.
        gae_lambda: The Generalized Advantage Estimation (GAE) lambda parameter which trades off bias and variance.
        clip_range: The clipping range which limits the change in the policy.
        entropy_coeff: The coefficient for the entropy loss.
        value_loss_coeff: The coefficient for the value loss.
        nonlinearity: The activation function used in the policy network. Either "tanh" or "relu".
        optimizer_lr: The learning rate for the optimizer.
    """

    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.state": [3],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [1],
        }
    )
    hidden_dim: list = field(default_factory=lambda: [64, 64])
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    nonlinearity: Literal["tanh", "relu"] = "tanh"

    optimizer_lr: float = 1e-3

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(lr=self.optimizer_lr)

    def get_scheduler_preset(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def validate_features(self) -> None:
        return None
