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

"""Top-level pipeline config for distributed RL training (actor / learner)."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig

from .algorithms.configs import RLAlgorithmConfig
from .algorithms.factory import make_algorithm_config
from .algorithms.sac import SACAlgorithmConfig  # noqa: F401


@dataclass(kw_only=True)
class TrainRLServerPipelineConfig(TrainPipelineConfig):
    # NOTE: In RL, we don't need an offline dataset
    # TODO: Make `TrainPipelineConfig.dataset` optional
    dataset: DatasetConfig | None = None  # type: ignore[assignment] # because the parent class has made it's type non-optional

    # Algorithm config.
    algorithm: RLAlgorithmConfig | None = None

    # Data mixer strategy name. Currently supports "online_offline".
    mixer: str = "online_offline"
    # Fraction sampled from online replay when using OnlineOfflineMixer.
    online_ratio: float = 0.5

    def validate(self) -> None:
        super().validate()

        if self.algorithm is None:
            self.algorithm = make_algorithm_config("sac")

        if getattr(self.algorithm, "policy_config", None) is None:
            self.algorithm.policy_config = self.policy
