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
"""Base classes for RL algorithms.

Defines the abstract interface that every algorithm must implement, a registry
for algorithm configs, and a dataclass for training statistics.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import draccus
import torch
from torch import Tensor
from torch.optim import Optimizer

if TYPE_CHECKING:
    from lerobot.rl.data_sources.data_mixer import DataMixer

BatchType = dict[str, Any]


@dataclass
class TrainingStats:
    """Returned by ``algorithm.update()`` for logging and checkpointing."""

    # Generic containers for all algorithms
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


class RLAlgorithm(abc.ABC):
    """Base for all RL algorithms."""

    @abc.abstractmethod
    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """One complete training step.

        The algorithm calls ``next(batch_iterator)`` as many times as it
        needs (e.g. ``utd_ratio`` times for SAC) to obtain fresh batches.
        The iterator is owned by the trainer; the algorithm just consumes
        from it.
        """
        ...

    def supports_offline_phase(self) -> bool:
        """Whether this algorithm has an offline pretraining phase.

        Algorithms like RLT (RL-token training) or ConRFT (Cal-QL pretraining)
        return ``True`` here. The learner checks this before the main online
        loop and routes to :meth:`offline_update` accordingly.
        """
        return False

    def offline_update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """One offline training step (called before any online collection).

        Only called when :meth:`supports_offline_phase` returns ``True``.
        Uses the same iterator protocol as :meth:`update`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement offline_update(). "
            "Either override this method or return False from supports_offline_phase()."
        )

    def transition_to_online(self) -> None:  # noqa: B027
        """Called once when switching from offline to online phase.

        Use this to freeze modules trained offline, rebuild optimizers for the
        online phase, reset step counters, etc.

        Default is a no-op; subclasses override when they have an offline phase.
        """

    def configure_data_iterator(
        self,
        data_mixer: DataMixer,
        batch_size: int,
        *,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ) -> Iterator[BatchType]:
        """Create the data iterator this algorithm needs.

        The default implementation uses the standard ``data_mixer.get_iterator()``.
        Algorithms that need specialised sampling should override this method.
        """
        return data_mixer.get_iterator(
            batch_size=batch_size,
            async_prefetch=async_prefetch,
            queue_size=queue_size,
        )

    def make_optimizers(self) -> dict[str, Optimizer]:
        """Create, store, and return the optimizers needed for training.

        Called on the **learner** side after construction.  Subclasses must
        override this with algorithm-specific optimizer setup.
        """
        return {}

    def get_optimizers(self) -> dict[str, Optimizer]:
        """Return optimizers for checkpointing / external scheduling."""
        return {}

    @property
    def optimization_step(self) -> int:
        """Current learner optimization step.

        Part of the stable contract for checkpoint/resume. Algorithms can
        either use this default storage or override for custom behavior.
        """
        return getattr(self, "_optimization_step", 0)

    @optimization_step.setter
    def optimization_step(self, value: int) -> None:
        self._optimization_step = int(value)

    def get_weights(self) -> dict[str, Any]:
        """Policy state-dict to push to actors."""
        return {}

    @abc.abstractmethod
    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        """Load policy state-dict received from the learner (inverse of ``get_weights``)."""

    @torch.no_grad()
    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """Pre-compute observation features (e.g. frozen encoder cache).

        Returns ``(None, None)`` when caching is not applicable.
        """
        return None, None
