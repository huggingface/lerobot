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

"""Versioned PyArrow contract for offline frame-signal outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _pyarrow_available, require_package

from .types import SignalDescriptor

if TYPE_CHECKING or _pyarrow_available:
    import pyarrow as pa
else:
    pa = None  # type: ignore[assignment]

SCORING_FORMAT = "lerobot.frame_signals"
SCORING_SCHEMA_VERSION = 1
RESERVED_COLUMNS = ("index", "episode_index", "frame_index")

_FORMAT_KEY = b"lerobot.reward_scoring.format"
_SCHEMA_VERSION_KEY = b"lerobot.reward_scoring.schema_version"
_PROVENANCE_KEY = b"lerobot.reward_scoring.provenance"
_DESCRIPTORS_KEY = b"lerobot.reward_scoring.descriptors"
_EPISODE_INDICES_KEY = b"lerobot.reward_scoring.episode_indices"
_LEGACY_SOURCE_KEY = b"lerobot.reward_scoring.legacy_source"
_DIRECTIONS = {"higher", "lower", "none"}
_MISSING_VALUES = {"forbidden", "nan"}


def require_pyarrow() -> None:
    """Require PyArrow only when the scoring format is read or written."""
    require_package("pyarrow", extra="dataset")
    if pa is None:
        raise ImportError("Reward scoring requires pyarrow")


def canonical_json(value: Any, *, label: str) -> str:
    """Serialize metadata deterministically and reject non-finite values."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable with finite numeric values") from exc


def validate_signal_descriptor(name: str, descriptor: SignalDescriptor) -> None:
    """Validate the stable semantics declared for one signal."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Signal names must be non-empty strings")
    if name in RESERVED_COLUMNS:
        raise ValueError(f"Signal name {name!r} conflicts with a reserved column")
    if not isinstance(descriptor, SignalDescriptor):
        raise TypeError(f"Descriptor for signal {name!r} must be a SignalDescriptor")
    if not isinstance(descriptor.description, str) or not descriptor.description.strip():
        raise ValueError(f"Descriptor for signal {name!r} must have a non-empty description")
    if not isinstance(descriptor.direction, str) or descriptor.direction not in _DIRECTIONS:
        raise ValueError(f"Invalid direction for signal {name!r}: {descriptor.direction!r}")
    if not isinstance(descriptor.missing_values, str) or descriptor.missing_values not in _MISSING_VALUES:
        raise ValueError(f"Invalid missing_values for signal {name!r}: {descriptor.missing_values!r}")
    if descriptor.unit is not None and (not isinstance(descriptor.unit, str) or not descriptor.unit.strip()):
        raise ValueError(f"Descriptor unit for signal {name!r} must be non-empty or None")
    if descriptor.bounds is not None:
        if not isinstance(descriptor.bounds, tuple) or len(descriptor.bounds) != 2:
            raise ValueError(f"Bounds for signal {name!r} must contain exactly two values")
        lower, upper = descriptor.bounds
        if any(isinstance(bound, bool) or not isinstance(bound, (int, float)) for bound in (lower, upper)):
            raise ValueError(f"Bounds for signal {name!r} must contain numeric values")
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError(f"Bounds for signal {name!r} must be finite and ordered")


def _normalize_episode_indices(episode_indices: Sequence[int]) -> tuple[int, ...]:
    if any(
        not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_))
        for index in episode_indices
    ):
        raise ValueError("Frame-signal episode selection must contain only integers")
    normalized = tuple(int(index) for index in episode_indices)
    if any(index < 0 for index in normalized) or normalized != tuple(sorted(set(normalized))):
        raise ValueError(
            "Frame-signal episode selection must be a sorted list of unique non-negative integers"
        )
    return normalized


def build_metadata(
    *,
    descriptors: Mapping[str, SignalDescriptor],
    provenance: Mapping[str, Any],
    episode_indices: Sequence[int],
    legacy_source: str | None = None,
) -> dict[bytes, bytes]:
    """Encode the scoring contract into Parquet schema metadata."""
    for name, descriptor in descriptors.items():
        validate_signal_descriptor(name, descriptor)
    normalized_episode_indices = _normalize_episode_indices(episode_indices)
    metadata = {
        _FORMAT_KEY: SCORING_FORMAT.encode(),
        _SCHEMA_VERSION_KEY: str(SCORING_SCHEMA_VERSION).encode(),
        _PROVENANCE_KEY: canonical_json(dict(provenance), label="provenance").encode(),
        _DESCRIPTORS_KEY: canonical_json(
            {name: asdict(descriptor) for name, descriptor in sorted(descriptors.items())},
            label="signal descriptors",
        ).encode(),
        _EPISODE_INDICES_KEY: canonical_json(
            list(normalized_episode_indices), label="episode indices"
        ).encode(),
    }
    if legacy_source is not None:
        metadata[_LEGACY_SOURCE_KEY] = legacy_source.encode()
    return metadata


def decode_signal_descriptors(table: pa.Table) -> dict[str, SignalDescriptor]:
    """Decode and validate signal descriptors from table metadata."""
    metadata = table.schema.metadata or {}
    serialized = metadata.get(_DESCRIPTORS_KEY)
    if serialized is None:
        raise ValueError("Frame-signal table has no serialized signal descriptors")
    try:
        payload = json.loads(serialized)
        if not isinstance(payload, dict):
            raise TypeError
        descriptors: dict[str, SignalDescriptor] = {}
        for name, value in payload.items():
            if (
                not isinstance(name, str)
                or not isinstance(value, dict)
                or not isinstance(value.get("description"), str)
                or not isinstance(value.get("direction"), str)
                or not (value.get("unit") is None or isinstance(value.get("unit"), str))
                or not isinstance(value.get("missing_values"), str)
                or not (
                    value.get("bounds") is None
                    or (
                        isinstance(value.get("bounds"), list)
                        and len(value["bounds"]) == 2
                        and all(
                            isinstance(bound, (int, float)) and not isinstance(bound, bool)
                            for bound in value["bounds"]
                        )
                    )
                )
            ):
                raise TypeError
            descriptor = SignalDescriptor(
                description=value["description"],
                direction=value["direction"],
                bounds=tuple(value["bounds"]) if value["bounds"] is not None else None,
                unit=value["unit"],
                missing_values=value["missing_values"],
            )
            validate_signal_descriptor(name, descriptor)
            descriptors[name] = descriptor
        return descriptors
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Frame-signal descriptors are malformed") from exc


def decode_provenance(table: pa.Table) -> dict[str, Any]:
    """Decode scoring provenance from table metadata."""
    metadata = table.schema.metadata or {}
    serialized = metadata.get(_PROVENANCE_KEY)
    if serialized is None:
        raise ValueError("Frame-signal table has no serialized provenance")
    try:
        value = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise ValueError("Frame-signal provenance is malformed") from exc
    if not isinstance(value, dict):
        raise ValueError("Frame-signal provenance must be a JSON object")
    return value


def decode_episode_indices(table: pa.Table) -> tuple[int, ...]:
    """Decode and validate the episode selection stored in table metadata."""
    metadata = table.schema.metadata or {}
    try:
        episode_indices = json.loads(metadata[_EPISODE_INDICES_KEY])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Frame-signal episode selection is missing or malformed") from exc
    if not isinstance(episode_indices, list):
        raise ValueError("Frame-signal episode selection must be a JSON list")
    return _normalize_episode_indices(episode_indices)


def build_episode_table(
    *,
    episode_index: int,
    episode_start: int,
    frame_indices: np.ndarray,
    signals: Mapping[str, np.ndarray],
) -> pa.Table:
    """Build one normalized episode table from validated NumPy signals."""
    require_pyarrow()
    return pa.table(
        {
            "index": episode_start + frame_indices,
            "episode_index": np.full(frame_indices.shape, episode_index, dtype=np.int64),
            "frame_index": frame_indices,
            **dict(sorted(signals.items())),
        }
    )


def attach_metadata(
    table: pa.Table,
    *,
    descriptors: Mapping[str, SignalDescriptor],
    provenance: Mapping[str, Any],
    episode_indices: Sequence[int],
    legacy_source: str | None = None,
) -> pa.Table:
    """Attach canonical scoring metadata to a table."""
    return table.replace_schema_metadata(
        build_metadata(
            descriptors=descriptors,
            provenance=provenance,
            episode_indices=episode_indices,
            legacy_source=legacy_source,
        )
    )


def is_frame_signal_table(table: pa.Table) -> bool:
    """Return whether a table declares the current frame-signal format."""
    return (table.schema.metadata or {}).get(_FORMAT_KEY) == SCORING_FORMAT.encode()


def validate_frame_signal_table(table: pa.Table) -> None:
    """Validate a serialized frame-signal table before use or publication."""
    require_pyarrow()
    metadata = table.schema.metadata or {}
    if metadata.get(_FORMAT_KEY) != SCORING_FORMAT.encode():
        raise ValueError(
            f"Unsupported frame-signal format in parquet metadata: {metadata.get(_FORMAT_KEY)!r}"
        )
    if metadata.get(_SCHEMA_VERSION_KEY) != str(SCORING_SCHEMA_VERSION).encode():
        raise ValueError(
            "Unsupported frame-signal schema version: "
            f"{metadata.get(_SCHEMA_VERSION_KEY, b'missing').decode(errors='replace')}"
        )

    descriptors = decode_signal_descriptors(table)
    decode_provenance(table)
    episode_indices = decode_episode_indices(table)

    required_columns = set(RESERVED_COLUMNS)
    missing_columns = required_columns.difference(table.column_names)
    if missing_columns:
        raise ValueError(f"Frame-signal output is missing columns: {sorted(missing_columns)}")
    signal_columns = set(table.column_names).difference(required_columns)
    if signal_columns != set(descriptors):
        raise ValueError(
            "Frame-signal columns and descriptors differ: "
            f"columns={sorted(signal_columns)}, descriptors={sorted(descriptors)}"
        )

    for name in RESERVED_COLUMNS:
        column = table[name]
        if not pa.types.is_int64(column.type) or column.null_count:
            raise ValueError(f"Frame-signal column {name!r} must be non-null int64")

    global_indices = table["index"].to_numpy(zero_copy_only=False)
    row_episode_indices = table["episode_index"].to_numpy(zero_copy_only=False)
    frame_indices = table["frame_index"].to_numpy(zero_copy_only=False)
    observed_episode_order = list(dict.fromkeys(int(index) for index in row_episode_indices))
    if global_indices.size > 1 and np.any(np.diff(global_indices) <= 0):
        raise ValueError("Frame-signal global indices must be unique and strictly increasing")
    if any(index not in episode_indices for index in observed_episode_order):
        raise ValueError("Frame-signal rows contain an episode outside the recorded selection")
    if observed_episode_order != sorted(observed_episode_order):
        raise ValueError("Frame-signal rows must be ordered by episode index")
    for episode_index in observed_episode_order:
        mask = row_episode_indices == episode_index
        episode_frames = frame_indices[mask]
        episode_global_indices = global_indices[mask]
        if np.any(episode_frames < 0) or (episode_frames.size > 1 and np.any(np.diff(episode_frames) <= 0)):
            raise ValueError(
                f"Frame indices for episode {episode_index} must be non-negative and strictly increasing"
            )
        if episode_frames.size and np.unique(episode_global_indices - episode_frames).size != 1:
            raise ValueError(f"Global and local frame indices are inconsistent for episode {episode_index}")

    for name, descriptor in descriptors.items():
        column = table[name]
        if column.null_count or not (
            pa.types.is_boolean(column.type)
            or pa.types.is_integer(column.type)
            or pa.types.is_floating(column.type)
        ):
            raise ValueError(f"Signal column {name!r} must be non-null bool, integer, or floating data")
        values = column.to_numpy(zero_copy_only=False)
        if np.issubdtype(values.dtype, np.floating):
            if np.isinf(values).any():
                raise ValueError(f"Signal column {name!r} must not contain infinite values")
            if descriptor.missing_values == "forbidden" and np.isnan(values).any():
                raise ValueError(f"Signal column {name!r} contains forbidden NaN values")
            values_for_bounds = values[np.isfinite(values)]
        else:
            values_for_bounds = values
        if descriptor.bounds is not None and values_for_bounds.size:
            lower, upper = descriptor.bounds
            if np.any(values_for_bounds < lower) or np.any(values_for_bounds > upper):
                raise ValueError(f"Signal column {name!r} contains values outside semantic bounds")
