# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Base classes for pluggable reward models.

RewardModelConfig and RewardModel provide the abstract interface for all reward
models in LeRobot (classifiers, SARM, zero-shot VLM rewards, etc.).

They extend PreTrainedConfig / PreTrainedPolicy so that reward models get
Hub save/load and normalization for free, while defaulting the policy-specific
methods (select_action, predict_action_chunk) that don't apply.
"""

import abc
from dataclasses import dataclass

from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy


@dataclass
class RewardModelConfig(PreTrainedConfig):
    """Base configuration for reward models.

    Provides default no-op implementations for the delta-index properties
    that only apply to action policies.  Concrete reward model configs
    should subclass this and implement get_optimizer_preset,
    get_scheduler_preset, and validate_features.
    """

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None
    
    @property
    def reward_delta_indices(self) -> list | None:
        return None


class RewardModel(PreTrainedPolicy, abc.ABC):
    """Abstract base class for reward models.

    Extends PreTrainedPolicy for Hub save/load and normalization, but
    replaces the action-oriented interface with a reward-oriented one.

    Subclasses must implement:
        - compute_reward(batch) -> Tensor
        - forward(batch) -> tuple[Tensor, dict | None]
    """

    config_class = RewardModelConfig
    name = "reward_model"

    @abc.abstractmethod
    def compute_reward(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute reward signal from a batch of observations.

        This is the canonical inference-time method. Training uses forward().

        Args:
            batch: Dictionary containing observation tensors.

        Returns:
            Reward tensor.
        """
        ...

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError("Reward models do not select actions")

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        raise NotImplementedError("Reward models do not predict action chunks")

    def reset(self):
        pass

    def get_optim_params(self):
        return self.parameters()
