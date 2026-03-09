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

from collections.abc import Iterator
from typing import Any

import torch

from lerobot.rl.algorithms.base import (
    BatchType,
    RLAlgorithm,
    TrainingStats,
)
from lerobot.rl.data_sources.data_mixer import DataMixer
from lerobot.utils.constants import ACTION


def preprocess_rl_batch(preprocessor: Any, batch: BatchType, *, action_dim: int | None = None) -> BatchType:
    """Apply a policy preprocessor to an RL batch."""
    observations = batch["state"]
    next_observations = batch["next_state"]
    actions = batch[ACTION]

    extra_action = None
    if action_dim is not None and actions.shape[-1] > action_dim:
        extra_action = actions[..., action_dim:]
        actions = actions[..., :action_dim]

    obs_action = {**observations, ACTION: actions}
    obs_action = preprocessor(obs_action)
    batch["state"] = {k: v for k, v in obs_action.items() if k.startswith("observation.")}
    batch[ACTION] = obs_action[ACTION]

    if extra_action is not None:
        batch[ACTION] = torch.cat([batch[ACTION], extra_action], dim=-1)

    next_obs = {**next_observations}
    next_obs = preprocessor(next_obs)
    batch["next_state"] = {k: v for k, v in next_obs.items() if k.startswith("observation.")}

    return batch


class _PreprocessedIterator:
    """Iterator wrapper that preprocesses each sampled RL batch."""

    __slots__ = ("_raw", "_preprocessor", "_action_dim")

    def __init__(
        self, raw_iterator: Iterator[BatchType], preprocessor: Any, action_dim: int | None = None
    ) -> None:
        self._raw = raw_iterator
        self._preprocessor = preprocessor
        self._action_dim = action_dim

    def __iter__(self) -> _PreprocessedIterator:
        return self

    def __next__(self) -> BatchType:
        batch = next(self._raw)
        return preprocess_rl_batch(self._preprocessor, batch, action_dim=self._action_dim)


class RLTrainer:
    """Unified training step orchestrator.

    Holds the algorithm, a DataMixer, and an optional preprocessor.
    """

    def __init__(
        self,
        algorithm: RLAlgorithm,
        data_mixer: DataMixer,
        batch_size: int,
        *,
        preprocessor: Any | None = None,
        action_dim: int | None = None,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        self.algorithm = algorithm
        self.data_mixer = data_mixer
        self.batch_size = batch_size
        self._preprocessor = preprocessor
        self._action_dim = action_dim
        self.async_prefetch = async_prefetch
        self.queue_size = queue_size

        self._iterator: Iterator[BatchType] | None = None

        self.algorithm.make_optimizers()

    def _build_data_iterator(self) -> Iterator[BatchType]:
        """Create a fresh algorithm-configured iterator (optionally preprocessed)."""
        raw = self.algorithm.configure_data_iterator(
            data_mixer=self.data_mixer,
            batch_size=self.batch_size,
            async_prefetch=self.async_prefetch,
            queue_size=self.queue_size,
        )
        if self._preprocessor is not None:
            return _PreprocessedIterator(raw, self._preprocessor, self._action_dim)
        return raw

    def reset_data_iterator(self) -> None:
        """Discard the current iterator so it will be rebuilt lazily next step."""
        self._iterator = None

    def set_data_mixer(self, data_mixer: DataMixer, *, reset: bool = True) -> None:
        """Swap the active data mixer, optionally resetting the iterator."""
        self.data_mixer = data_mixer
        if reset:
            self.reset_data_iterator()

    def training_step(self) -> TrainingStats:
        """Run one training step (algorithm-agnostic)."""
        if self._iterator is None:
            self._iterator = self._build_data_iterator()
        return self.algorithm.update(self._iterator)
