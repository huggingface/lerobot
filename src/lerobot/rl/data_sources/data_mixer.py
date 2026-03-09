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

import abc
from typing import Any

from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions

BatchType = dict[str, Any]


class DataMixer(abc.ABC):
    """Abstract interface for all data mixing strategies.

    Subclasses must implement ``sample(batch_size)`` and may override
    ``get_iterator`` for specialised iteration.
    """

    @abc.abstractmethod
    def sample(self, batch_size: int) -> BatchType:
        """Draw one batch of ``batch_size`` transitions."""
        ...

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """Infinite iterator that yields batches.

        The default implementation repeatedly calls ``self.sample()``.
        Subclasses with underlying buffer iterators (async prefetch)
        should override this for better throughput.
        """
        while True:
            yield self.sample(batch_size)


class OnlineOfflineMixer(DataMixer):
    """Mixes transitions from an online and an optional offline replay buffer.

    When both buffers are present, each batch is constructed by sampling
    ``ceil(batch_size * online_ratio)`` from the online buffer and the
    remainder from the offline buffer, then concatenating.

    This mixer assumes both online and offline buffers are present.
    """

    def __init__(
        self,
        online_buffer: ReplayBuffer,
        offline_buffer: ReplayBuffer | None = None,
        online_ratio: float = 1.0,
    ):
        if not 0.0 <= online_ratio <= 1.0:
            raise ValueError(f"online_ratio must be in [0, 1], got {online_ratio}")
        self.online_buffer = online_buffer
        self.offline_buffer = offline_buffer
        self.online_ratio = online_ratio

    def sample(self, batch_size: int) -> BatchType:
        if self.offline_buffer is None:
            return self.online_buffer.sample(batch_size)

        n_online = max(1, int(batch_size * self.online_ratio))
        n_offline = batch_size - n_online

        online_batch = self.online_buffer.sample(n_online)
        offline_batch = self.offline_buffer.sample(n_offline)
        return concatenate_batch_transitions(online_batch, offline_batch)

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """Yield batches from online/offline mixed sampling."""
        while True:
            yield self.sample(batch_size)
