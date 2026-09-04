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

"""Episode runner and validation for offline frame-signal scoring."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .schema import RESERVED_COLUMNS, build_episode_table, validate_signal_descriptor
from .types import FrameSignals, ScoringSummary, SignalDescriptor
from .writer import _FrameSignalsWriter

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset


def _episode_count(dataset: LeRobotDataset) -> int:
    total = getattr(dataset.meta, "total_episodes", None)
    if total is not None:
        return int(total)
    return len(dataset.meta.episodes)


def _resolve_episode_indices(
    dataset: LeRobotDataset, episode_indices: Sequence[int] | None
) -> tuple[int, ...]:
    total_episodes = _episode_count(dataset)
    if episode_indices is None:
        selected = getattr(dataset, "episodes", None)
        episode_indices = range(total_episodes) if selected is None else selected

    if any(
        not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_))
        for index in episode_indices
    ):
        raise TypeError("episode_indices must contain only integers")
    normalized = tuple(int(index) for index in episode_indices)
    if not normalized:
        raise ValueError("At least one episode must be selected for scoring")
    if len(set(normalized)) != len(normalized):
        raise ValueError("episode_indices must not contain duplicates")
    if any(index < 0 or index >= total_episodes for index in normalized):
        raise ValueError(f"episode_indices must be in [0, {total_episodes}), got {list(normalized)}")
    return tuple(sorted(normalized))


def _episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    episode = dataset.meta.episodes[episode_index]
    try:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Episode {episode_index} has invalid dataset frame bounds") from exc
    if start < 0 or end <= start:
        raise ValueError(f"Episode {episode_index} has invalid dataset frame bounds [{start}, {end})")
    return start, end


def _validate_frame_signals(
    frame_signals: FrameSignals,
    *,
    episode_index: int,
    episode_length: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, SignalDescriptor]]:
    if not isinstance(frame_signals, FrameSignals):
        raise TypeError(f"Scorer for episode {episode_index} must return FrameSignals")
    if not isinstance(frame_signals.frame_indices, np.ndarray):
        raise TypeError("FrameSignals.frame_indices must be a NumPy array")

    frame_indices = frame_signals.frame_indices
    if frame_indices.ndim != 1 or frame_indices.dtype.kind not in "iu":
        raise ValueError("FrameSignals.frame_indices must be a one-dimensional integer array")
    frame_indices = frame_indices.astype(np.int64, copy=False)
    if frame_indices.size:
        if frame_indices[0] < 0 or frame_indices[-1] >= episode_length:
            raise ValueError(f"Frame indices for episode {episode_index} must be in [0, {episode_length})")
        if np.any(np.diff(frame_indices) <= 0):
            raise ValueError(
                f"Frame indices for episode {episode_index} must be unique and strictly increasing"
            )

    if not isinstance(frame_signals.signals, Mapping) or not isinstance(frame_signals.descriptors, Mapping):
        raise TypeError("FrameSignals signals and descriptors must be mappings")
    if any(not isinstance(name, str) or not name.strip() for name in frame_signals.signals):
        raise ValueError("Signal names must be non-empty strings")
    if any(not isinstance(name, str) or not name.strip() for name in frame_signals.descriptors):
        raise ValueError("Descriptor names must be non-empty strings")
    signal_names = set(frame_signals.signals)
    descriptor_names = set(frame_signals.descriptors)
    if not signal_names:
        raise ValueError("FrameSignals must contain at least one signal")
    if signal_names != descriptor_names:
        raise ValueError(
            "FrameSignals signals and descriptors must have identical names: "
            f"signals={sorted(signal_names)}, descriptors={sorted(descriptor_names)}"
        )
    conflicting = signal_names.intersection(RESERVED_COLUMNS)
    if conflicting:
        raise ValueError(f"Signal names conflict with reserved columns: {sorted(conflicting)}")

    signals: dict[str, np.ndarray] = {}
    descriptors: dict[str, SignalDescriptor] = {}
    for name in sorted(signal_names):
        descriptor = frame_signals.descriptors[name]
        validate_signal_descriptor(name, descriptor)
        descriptors[name] = descriptor

        values = frame_signals.signals[name]
        if not isinstance(values, np.ndarray):
            raise TypeError(f"Signal {name!r} must be a NumPy array")
        if values.ndim != 1 or values.shape[0] != frame_indices.shape[0]:
            raise ValueError(
                f"Signal {name!r} must have shape ({frame_indices.shape[0]},), got {values.shape}"
            )
        if values.dtype.kind not in "biuf":
            raise ValueError(f"Signal {name!r} must have a bool, integer, or floating dtype")

        if values.dtype.kind == "f":
            if np.isinf(values).any():
                raise ValueError(f"Signal {name!r} must not contain infinite values")
            if descriptor.missing_values == "forbidden" and np.isnan(values).any():
                raise ValueError(f"Signal {name!r} contains forbidden NaN values")
            values_for_bounds = values[np.isfinite(values)]
        else:
            values_for_bounds = values

        if descriptor.bounds is not None and values_for_bounds.size:
            lower, upper = descriptor.bounds
            if np.any(values_for_bounds < lower) or np.any(values_for_bounds > upper):
                raise ValueError(
                    f"Signal {name!r} contains values outside its semantic bounds [{lower}, {upper}]"
                )
        signals[name] = values

    return frame_indices, signals, descriptors


def score_dataset(
    dataset: LeRobotDataset,
    scorer: Callable[[LeRobotDataset, int], FrameSignals],
    *,
    output_path: Path,
    provenance: Mapping[str, Any],
    episode_indices: Sequence[int] | None = None,
    resume: bool = True,
) -> ScoringSummary:
    """Score selected episodes and atomically build one frame-signal output.

    A scorer failure leaves only fully committed earlier episode parts. Calling
    the function again with identical provenance and selection resumes from
    those parts.
    """
    selected = _resolve_episode_indices(dataset, episode_indices)
    bounds = {episode_index: _episode_bounds(dataset, episode_index) for episode_index in selected}
    writer = _FrameSignalsWriter(
        Path(output_path),
        provenance=provenance,
        episode_indices=selected,
        resume=resume,
    )
    completed_before = writer.completed_episode_indices
    descriptors = writer.existing_descriptors
    signal_types = writer.existing_signal_types

    new_episode_count = 0
    for episode_index in selected:
        if episode_index in completed_before:
            continue
        episode_start, episode_end = bounds[episode_index]
        frame_signals = scorer(dataset, episode_index)
        frame_indices, signals, current_descriptors = _validate_frame_signals(
            frame_signals,
            episode_index=episode_index,
            episode_length=episode_end - episode_start,
        )
        if descriptors is None:
            descriptors = current_descriptors
        elif current_descriptors != descriptors:
            raise ValueError(f"Signal descriptors for episode {episode_index} differ from earlier episodes")

        table = build_episode_table(
            episode_index=episode_index,
            episode_start=episode_start,
            frame_indices=frame_indices,
            signals=signals,
        )
        current_types = {
            name: table.schema.field(name).type for name in table.column_names if name not in RESERVED_COLUMNS
        }
        if signal_types is None:
            signal_types = current_types
        elif current_types != signal_types:
            raise ValueError(
                f"Signal dtypes for episode {episode_index} differ from earlier episodes: "
                f"expected={signal_types}, got={current_types}"
            )
        writer.write_episode(episode_index, table, descriptors)
        new_episode_count += 1

    if descriptors is None:
        raise RuntimeError("Scoring produced no signal descriptors")
    output_path, frame_count = writer.finalize(descriptors)
    return ScoringSummary(
        output_path=output_path,
        episode_count=len(selected),
        new_episode_count=new_episode_count,
        resumed_episode_count=len(completed_before),
        frame_count=frame_count,
    )
