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

"""Public types for frame-aligned offline reward-model signals."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

SignalDirection = Literal["higher", "lower", "none"]
MissingValues = Literal["forbidden", "nan"]


@dataclass(frozen=True)
class SignalDescriptor:
    """Stable semantics attached to one named signal.

    ``bounds`` describes theoretical or semantic bounds. ``missing_values`` is
    strict by default; ``"nan"`` exists for signals such as legacy SARM output
    where some frames were intentionally not scored.
    """

    description: str
    direction: SignalDirection
    bounds: tuple[float, float] | None = None
    unit: str | None = None
    missing_values: MissingValues = "forbidden"


@dataclass(frozen=True)
class FrameSignals:
    """One episode's sparse or dense frame-aligned signals.

    ``frame_indices`` contains episode-local indices. Every signal is a
    one-dimensional NumPy array aligned one-to-one with those indices.
    """

    frame_indices: np.ndarray
    signals: Mapping[str, np.ndarray]
    descriptors: Mapping[str, SignalDescriptor]


@dataclass(frozen=True)
class ScoringSummary:
    """Small result describing one completed dataset-scoring run."""

    output_path: Path
    episode_count: int
    new_episode_count: int
    resumed_episode_count: int
    frame_count: int
