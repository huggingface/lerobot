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

from lerobot.rl.algorithms.base import (
    BatchType,
    RLAlgorithm,
    TrainingStats,
)
from lerobot.rl.data_sources.data_mixer import DataMixer


class RLTrainer:
    """Unified training step orchestrator.

    Holds the algorithm and a ``DataMixer``. The batch iterator is created
    lazily (on first ``training_step()``) via the algorithm's
    ``configure_data_iterator()`` hook — this allows each algorithm to
    request specialised sampling (e.g. n-step returns for ACFQL) without
    the trainer needing to know about it.

    The trainer passes the iterator directly to ``algorithm.update()``,
    the algorithm pulls batches via ``next(iterator)``.

    """

    def __init__(
        self,
        algorithm: RLAlgorithm,
        data_mixer: DataMixer,
        batch_size: int,
        *,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        self.algorithm = algorithm
        self.data_mixer = data_mixer
        self.batch_size = batch_size
        self.async_prefetch = async_prefetch
        self.queue_size = queue_size

        self._iterator: Iterator[BatchType] | None = None

        self.algorithm.make_optimizers()

    def training_step(self) -> TrainingStats:
        """Run one training step (algorithm-agnostic).

        The iterator is passed directly to the algorithm. UTD and other
        update details are handled inside ``algorithm.update()``.
        """
        if self._iterator is None:
            self._iterator = self.algorithm.configure_data_iterator(
                data_mixer=self.data_mixer,
                batch_size=self.batch_size,
                async_prefetch=self.async_prefetch,
                queue_size=self.queue_size,
            )
        batch_iterator = self._iterator
        return self.algorithm.update(batch_iterator)
