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

"""Reward-model contracts and orchestration for scoring dataset frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from lerobot.utils.import_utils import _av_available, _datasets_available, require_package

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

if TYPE_CHECKING or (_datasets_available and _av_available):
    from lerobot.datasets.annotations import add_annotations, load_annotations_info


@dataclass(frozen=True)
class SignalDescriptor:
    """Describe a signal's meaning, preferred direction, bounds and unit."""

    description: str
    greater_is_better: bool | None
    bounds: tuple[float, float] | None = None
    unit: str | None = None
    allow_nan: bool = False


@dataclass(frozen=True)
class FrameSignals:
    """Sparse or dense frame-aligned signals for one episode."""

    frame_indices: np.ndarray
    signals: Mapping[str, np.ndarray]


class FrameScorer(Protocol):
    """Produce frame-aligned signals and describe their meaning up front."""

    name: str

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    @property
    def descriptors(self) -> Mapping[str, SignalDescriptor]: ...

    def score_episode(self, dataset: LeRobotDataset, episode_index: int) -> FrameSignals: ...


def score_dataset(
    dataset: LeRobotDataset,
    scorer: FrameScorer,
    *,
    name: str | None = None,
    episodes: Sequence[int] | None = None,
    resume: bool = True,
    overwrite: bool = False,
) -> None:
    """Score selected episodes and store their signals as a named annotation table.

    Args:
        dataset: Dataset to score.
        scorer: Model adapter providing signals, their descriptors and provenance.
        name: Table name. Defaults to ``scorer.name``.
        episodes: Global episode indices to score. Defaults to the dataset's selected episodes.
        resume: If ``True``, skip episodes written by an interrupted matching run.
        overwrite: If ``True``, replace an existing table once scoring succeeds.

    Raises:
        ImportError: If the dataset extra's dependencies are unavailable.
        ValueError: If a descriptor has an invalid direction, signal names do not match
            descriptors, frame indices repeat, or signals have non-numeric dtypes.
    """
    require_package("datasets", extra="dataset")
    require_package("av", extra="dataset")
    descriptors = dict(scorer.descriptors)
    for signal_name, descriptor in descriptors.items():
        if not any(descriptor.greater_is_better is value for value in (True, False, None)):
            raise ValueError(f"Signal {signal_name!r}: greater_is_better must be True, False or None")

    info = {
        "provenance": dict(scorer.provenance),
        "signals": {signal_name: asdict(descriptor) for signal_name, descriptor in descriptors.items()},
    }

    def score_episode(episode_index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        frame_signals = scorer.score_episode(dataset, episode_index)
        if set(frame_signals.signals) != set(descriptors):
            raise ValueError(
                f"Signal names for episode {episode_index} must match the scorer's descriptors: "
                f"signals={sorted(frame_signals.signals)}, descriptors={sorted(descriptors)}"
            )
        if np.unique(frame_signals.frame_indices).size != frame_signals.frame_indices.size:
            raise ValueError(f"Frame indices for episode {episode_index} must not contain repeated values")
        for signal_name, values in frame_signals.signals.items():
            if values.dtype.kind not in "biuf":
                raise ValueError(
                    f"Signal {signal_name!r} for episode {episode_index} must have a numeric dtype "
                    f"(bool, integer or float), got {values.dtype}"
                )
        return frame_signals.frame_indices, dict(frame_signals.signals)

    add_annotations(
        dataset,
        scorer.name if name is None else name,
        score_episode,
        info=info,
        episodes=episodes,
        resume=resume,
        overwrite=overwrite,
    )


def load_signal_descriptors(dataset: LeRobotDataset, name: str) -> dict[str, SignalDescriptor]:
    """Load the descriptors stored with a scored annotation table.

    Args:
        dataset: Dataset the table belongs to.
        name: Table name.

    Returns:
        The table's signal descriptors, with bounds restored as tuples.

    Raises:
        ImportError: If the dataset extra's dependencies are unavailable.
        ValueError: If the table has no signal descriptors from ``score_dataset``.
    """
    require_package("datasets", extra="dataset")
    require_package("av", extra="dataset")
    info = load_annotations_info(dataset, name)
    if "signals" not in info:
        raise ValueError(
            f"Annotations {name!r} have no signal descriptors; use a table written by score_dataset"
        )

    descriptors = {}
    for signal_name, descriptor in info["signals"].items():
        fields = dict(descriptor)
        bounds = fields.pop("bounds", None)
        descriptors[signal_name] = SignalDescriptor(
            **fields, bounds=None if bounds is None else tuple(bounds)
        )
    return descriptors
