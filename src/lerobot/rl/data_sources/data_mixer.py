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
    ``get_iterator`` for specialised iteration (e.g. n-step returns).

    Extra ``**kwargs`` on ``get_iterator`` are forwarded to the underlying
    buffer so that algorithms can request n-step, chunked, or trajectory
    sampling without the mixer needing dedicated methods for each.
    """

    @abc.abstractmethod
    def sample(self, batch_size: int, **kwargs: Any) -> BatchType:
        """Draw one batch of ``batch_size`` transitions."""
        ...

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
        **kwargs: Any,
    ):
        """Infinite iterator that yields batches.

        The default implementation repeatedly calls ``self.sample()``.
        Subclasses with underlying buffer iterators (async prefetch, n-step)
        should override this for better throughput.

        Extra *kwargs* are forwarded to the underlying buffer's iterator
        method, enabling n-step (``n_steps=``, ``gamma=``), chunked
        (``action_chunk_size=``), or other specialised sampling without
        the mixer needing a method per variant.
        """
        while True:
            yield self.sample(batch_size, **kwargs)


class OnlineOfflineMixer(DataMixer):
    """Mixes transitions from an online and an optional offline replay buffer.

    When both buffers are present, each batch is constructed by sampling
    ``ceil(batch_size * online_ratio)`` from the online buffer and the
    remainder from the offline buffer, then concatenating.

    When only the online buffer is provided, all samples come from it
    regardless of the ``online_ratio`` setting.
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

    def sample(self, batch_size: int, **kwargs: Any) -> BatchType:
        if self.offline_buffer is None or self.online_ratio >= 1.0:
            return dict(self.online_buffer.sample(batch_size))

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
        **kwargs: Any,
    ):
        """Yield batches, delegating to buffer iterators when possible.

        If only one buffer is used (no mixing), we delegate to the buffer's
        own ``get_iterator`` which supports async prefetch.  When mixing,
        we fall back to synchronous ``sample()`` calls because the two
        buffers must be sampled independently and concatenated.

        Extra *kwargs* are forwarded to the buffer's ``get_iterator``
        (e.g. ``n_steps``, ``gamma``, ``action_chunk_size``).
        """
        if self.offline_buffer is None or self.online_ratio >= 1.0:
            yield from self.online_buffer.get_iterator(
                batch_size=batch_size,
                async_prefetch=async_prefetch,
                queue_size=queue_size,
                **kwargs,
            )
        else:
            while True:
                yield self.sample(batch_size)
