#!/usr/bin/env python

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
"""A minimal real PreTrainedPolicy for checkpoint/publish unit tests (CPU, tiny)."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamConfig, OptimizerConfig
from lerobot.policies.pretrained import PreTrainedPolicy


@PreTrainedConfig.register_subclass("dummy_checkpoint")
@dataclass
class DummyCheckpointConfig(PreTrainedConfig):
    hidden: int = 4

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamConfig(lr=1e-3)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        pass


class DummyCheckpointPolicy(PreTrainedPolicy):
    config_class = DummyCheckpointConfig
    name = "dummy_checkpoint"

    def __init__(self, config: DummyCheckpointConfig, **kwargs):
        super().__init__(config)
        self.net = nn.Linear(config.hidden, config.hidden)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self) -> None:
        pass

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        out = self.net(batch["observation.state"])
        return out.mean(), None

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.net(batch["observation.state"])

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.net(batch["observation.state"])


def make_dummy_policy(repo_id: str | None = None) -> DummyCheckpointPolicy:
    config = DummyCheckpointConfig(device="cpu")
    if repo_id is not None:
        config.repo_id = repo_id
    policy = DummyCheckpointPolicy(config)
    with torch.no_grad():
        policy.net.weight.fill_(0.5)
    return policy
