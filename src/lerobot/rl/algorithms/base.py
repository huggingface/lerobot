# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Base classes for RL algorithms.

Defines the abstract interface that every algorithm must implement, a registry
for algorithm configs, and a dataclass for training statistics.
"""

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import draccus
import torch
from torch import Tensor
from torch.optim import Optimizer

# Batch type alias used throughout the RL stack.
BatchType = dict[str, Any]

# Callable that returns one batch when invoked (no arguments).
SampleFn = Callable[[], BatchType]


@dataclass
class TrainingStats:
    """Returned by ``algorithm.update()`` for logging and checkpointing."""

    loss_actor: float | None = None
    loss_critic: float | None = None
    loss_temperature: float | None = None
    loss_discrete_critic: float | None = None
    grad_norms: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class RLAlgorithmConfig(draccus.ChoiceRegistry):
    """Registry for algorithm configs.

    Subclasses register via ``@RLAlgorithmConfig.register_subclass("name")``,
    following the same pattern used for ``PreTrainedConfig``, ``CameraConfig``, etc.
    """

    pass


class RLAlgorithm(abc.ABC):
    """Base for all RL algorithms.

    An algorithm **owns the policy** and implements ``select_action`` (used by the
    actor) and ``update`` (used by the learner).  It never touches buffers, queues,
    or transport — that is the orchestrator's job.
    """

    @abc.abstractmethod
    def select_action(
        self, observation: dict[str, Tensor], deterministic: bool = False
    ) -> Tensor:
        """Select action(s) for rollout.

        Single-step policies (e.g. SAC) return shape ``(action_dim,)``;
        chunking policies (e.g. VLA) return ``(chunk_size, action_dim)``.
        """
        ...

    @abc.abstractmethod
    def update(self, sample_fn: SampleFn) -> TrainingStats:
        """One complete training step.

        The algorithm calls ``sample_fn()`` as many times as it needs
        (e.g. ``utd_ratio`` times for SAC) to obtain fresh batches.
        This keeps sampling in the orchestrator while the algorithm
        fully owns its gradient-step loop (including UTD warm-up).
        """
        ...

    def process_batch(self, batch: BatchType) -> BatchType:
        """Optional: transform batch before ``update``.  Default: identity."""
        return batch

    def get_optimizers(self) -> dict[str, Optimizer]:
        """Return optimizers for checkpointing / external scheduling."""
        return {}

    def get_weights(self) -> dict[str, Any]:
        """State-dict(s) to push to actors.

        For SAC this is typically the actor network (and optionally the
        discrete critic), **not** the full policy.
        """
        return {}

    @torch.no_grad()
    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """Pre-compute observation features (e.g. frozen encoder cache).

        Returns ``(None, None)`` when caching is not applicable.
        """
        return None, None
