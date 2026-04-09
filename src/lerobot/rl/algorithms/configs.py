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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import draccus
import torch

if TYPE_CHECKING:
    from lerobot.rl.algorithms.base import RLAlgorithm


@dataclass
class TrainingStats:
    """Returned by ``algorithm.update()`` for logging and checkpointing."""

    losses: dict[str, float] = field(default_factory=dict)
    grad_norms: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, float]:
        """Flatten all stats into a single dict for logging."""

        d: dict[str, float] = {}
        for name, val in self.losses.items():
            d[name] = val
        for name, val in self.grad_norms.items():
            d[f"{name}_grad_norm"] = val
        for name, val in self.extra.items():
            d[name] = val
        return d


@dataclass
class RLAlgorithmConfig(draccus.ChoiceRegistry):
    """Registry for algorithm configs."""

    def build_algorithm(self, policy: torch.nn.Module) -> RLAlgorithm:
        """Construct the :class:`RLAlgorithm` for this config.

        Must be overridden by every registered config subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement build_algorithm()")

    @classmethod
    def from_policy_config(cls, policy_cfg: Any) -> RLAlgorithmConfig:
        """Build an algorithm config from a policy config.

        Must be overridden by every registered config subclass.
        """
        raise NotImplementedError(f"{cls.__name__} must implement from_policy_config()")
