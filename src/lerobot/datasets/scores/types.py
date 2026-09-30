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

"""Dataset contracts for frame-aligned score sidecars."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

SignalDirection = Literal["higher", "lower", "none"]


@dataclass(frozen=True)
class SignalDescriptor:
    """How a stored signal should be interpreted."""

    description: str
    direction: SignalDirection
    bounds: tuple[float, float] | None = None
    unit: str | None = None
    allow_nan: bool = False


@dataclass(frozen=True)
class FrameSignals:
    """Sparse or dense frame-aligned signals for one episode."""

    frame_indices: np.ndarray
    signals: Mapping[str, np.ndarray]
    descriptors: Mapping[str, SignalDescriptor]


class FrameScorer(Protocol):
    """Produce frame-aligned signals for one dataset episode."""

    name: str

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def score_episode(
        self,
        dataset: LeRobotDataset,
        episode_index: int,
    ) -> FrameSignals: ...
