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
"""What a training sample holds, declared by each policy (``PreTrainedConfig.training_sampler``).

A sampler only decides what one sample reads and how many samples each episode gives. Datasets
own the rest (shuffling, data-parallel sharding, resume, decoding), so every sampler works with
every dataset reader.
"""

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .types import Deltas, FeatureType


@dataclass(frozen=True, eq=False)
class EpisodeTable:
    """The episodes a dataset samples from (its split), read from the dataset metadata.

    Attributes:
        indices: Dataset episode indices of the split, ascending.
        lengths: Frames of every dataset episode, indexed by episode index.
    """

    indices: np.ndarray
    lengths: np.ndarray

    def length(self, episode: int) -> int:
        """The number of frames of dataset episode ``episode``."""
        return int(self.lengths[episode])


class SampleKey(NamedTuple):
    """One training sample: the ``sample``-th sample of dataset episode ``episode`` in ``epoch``."""

    episode: int
    sample: int
    epoch: int = 0


@dataclass(frozen=True)
class Sample:
    """The frames a sample reads: ``deltas`` around frame ``anchor`` of its episode.

    ``anchor`` is episode-local and can lie outside the episode. As for any window, the positions
    outside the episode repeat its first or last frame and are flagged in ``{key}_is_pad``; an
    undeclared feature is read at the anchor, clamped to the episode.
    """

    anchor: int
    deltas: Deltas


class TrainingSampler(abc.ABC):
    """What one training sample holds and how many samples each episode gives per epoch.

    Implementations read metadata only (no I/O, no global random state) and must be picklable:
    the same key gives the same sample in every process.
    """

    @abc.abstractmethod
    def count(self, episodes: EpisodeTable, episode: int) -> int:
        """How many samples dataset episode ``episode`` gives per epoch; 0 skips it."""

    @abc.abstractmethod
    def sample(self, episodes: EpisodeTable, key: SampleKey) -> Sample:
        """The frames sample ``key`` reads."""


@dataclass(frozen=True)
class FrameSampler(TrainingSampler):
    """One sample per frame: ``deltas`` around every frame of the episode.

    Attributes:
        deltas: The windows around each frame.
        drop_first: Frames at the start of each episode that never anchor a sample.
        drop_last: Frames at the end of each episode that never anchor a sample.
    """

    deltas: Deltas = field(default_factory=Deltas)
    drop_first: int = 0
    drop_last: int = 0

    def __post_init__(self) -> None:
        if self.drop_first < 0 or self.drop_last < 0:
            raise ValueError(f"Dropped frames must be >= 0, got {self.drop_first} and {self.drop_last}.")

    def count(self, episodes: EpisodeTable, episode: int) -> int:
        return max(0, episodes.length(episode) - self.drop_first - self.drop_last)

    def sample(self, episodes: EpisodeTable, key: SampleKey) -> Sample:
        return Sample(self.drop_first + key.sample, self.deltas)


@dataclass(frozen=True)
class EpisodeSampler(TrainingSampler):
    """One sample per start: every ``stride``-th frame from the start to the end of the episode.

    Samples of different episodes have different lengths; the collate pads them to the longest
    of the batch and flags the padding in ``{key}_is_pad``.

    Attributes:
        observation: Stride per observation feature type or key, as in
            :class:`~lerobot.configs.types.Deltas`; undeclared observations are read at the start.
        action: Stride of ``action``, or ``None`` to read it at the start only.
        reward: Stride of the reward, or ``None`` to read it at the start only.
        starts: Episode frames samples start from, one sample each. A start before frame 0 (e.g.
            ``range(-31, 1)``) shifts every position of the sample: the positions before the episode
            repeat its first frame and are flagged as padding.
    """

    observation: Mapping[FeatureType, int] | Mapping[str, int] = field(default_factory=dict)
    action: int | None = None
    reward: int | None = None
    starts: Sequence[int] = (0,)

    def __post_init__(self) -> None:
        strides = [*self.observation.values(), self.action, self.reward]
        if any(s is not None and (type(s) is not int or s <= 0) for s in strides):
            raise ValueError(f"Strides must be positive integers, got {strides}.")
        object.__setattr__(self, "starts", tuple(self.starts))
        if not self.starts or any(type(s) is not int for s in self.starts):
            raise ValueError(f"starts must be a nonempty sequence of integers, got {self.starts}.")

    def count(self, episodes: EpisodeTable, episode: int) -> int:
        return len(self._starts(episodes.length(episode)))

    def sample(self, episodes: EpisodeTable, key: SampleKey) -> Sample:
        length = episodes.length(key.episode)
        start = self._starts(length)[key.sample]
        span = length - start
        observation = {kind: range(0, span, stride) for kind, stride in self.observation.items()}
        action = range(0, span, self.action) if self.action is not None else None
        reward = range(0, span, self.reward) if self.reward is not None else None
        return Sample(start, Deltas(observation=observation, action=action, reward=reward))

    def _starts(self, length: int) -> tuple[int, ...]:
        """The starts that leave at least one frame of an episode of ``length`` frames."""
        return tuple(start for start in self.starts if start < length)
