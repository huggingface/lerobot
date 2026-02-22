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
    """Returned by ``algorithm.update()`` for logging and checkpointing.

    The fixed fields (``loss_actor``, ``loss_critic``, …) exist for
    backward-compatibility with SAC.  New algorithms should use the
    generic ``losses`` dict for any number of named loss values and
    ``extra`` for non-loss metrics.
    """

    # SAC backward-compat fields
    loss_actor: float | None = None
    loss_critic: float | None = None
    loss_temperature: float | None = None
    loss_discrete_critic: float | None = None

    # Generic containers — preferred for new algorithms
    losses: dict[str, float] = field(default_factory=dict)
    grad_norms: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class RLAlgorithmConfig(draccus.ChoiceRegistry):
    """Registry for algorithm configs.

    Subclasses register via ``@RLAlgorithmConfig.register_subclass("name")``,
    following the same pattern used for ``PreTrainedConfig``, ``CameraConfig``, etc.

    Each registered config **must** implement :meth:`build_algorithm` so that
    :func:`make_algorithm` works without hardcoded ``if/else`` chains.
    """

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
    """Base for all RL algorithms.

    An algorithm **owns the policy** and implements ``select_action`` (used by the
    actor) and ``update`` (used by the learner).  It never touches buffers, queues,
    or transport — that is the orchestrator's job.

    The ``update`` method receives a batch iterator directly
    and pulls as many batches as it needs via ``next(batch_iterator)``.

    **Extension points for new algorithms:**

    - Override :meth:`update` — fully own the gradient loop (UTD, multi-loss, etc.).
    - Override :meth:`configure_data_iterator` — request specialised sampling
      (n-step, chunked actions, etc.) from the :class:`DataMixer`.
    - Override :meth:`make_optimizers` — create algorithm-specific optimizer groups.
    """

    @abc.abstractmethod
    def select_action(self, observation: dict[str, Tensor], deterministic: bool = False) -> Tensor:
        """Select action(s) for rollout.

        Single-step policies (e.g. SAC) return shape ``(action_dim,)``;
        chunking policies (e.g. VLA) return ``(chunk_size, action_dim)``.
        """
        ...

    @abc.abstractmethod
    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """One complete training step.

        The algorithm calls ``next(batch_iterator)`` as many times as it
        needs (e.g. ``utd_ratio`` times for SAC) to obtain fresh batches.
        The iterator is owned by the trainer; the algorithm just consumes
        from it.
        """
        ...

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
        Algorithms that need specialised sampling (e.g. n-step returns, action
        chunking) should override this to call the appropriate mixer/buffer
        method with extra keyword arguments.

        Called by the :class:`~lerobot.rl.trainer.RLTrainer` during lazy
        initialisation and again after each stage transition that provides
        a new data mixer.
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

    def get_weights(self) -> dict[str, Any]:
        """State-dict(s) to push to actors."""
        return {}

    @abc.abstractmethod
    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        """Load state-dict(s) received from the learner (inverse of ``get_weights``)."""

    @torch.no_grad()
    def get_observation_features(
        self, observations: Tensor, next_observations: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """Pre-compute observation features (e.g. frozen encoder cache).

        Returns ``(None, None)`` when caching is not applicable.
        """
        return None, None
