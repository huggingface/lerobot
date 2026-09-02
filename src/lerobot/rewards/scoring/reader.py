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

"""Reading and validation for frame-signal scoring outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import _pyarrow_available, require_package

if TYPE_CHECKING or _pyarrow_available:
    import pyarrow as pa
    import pyarrow.parquet as pq
else:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

from .types import SignalDescriptor

SCORING_FORMAT = "lerobot.frame_signals"
SCORING_SCHEMA_VERSION = 1

_FORMAT_KEY = b"lerobot.reward_scoring.format"
_SCHEMA_VERSION_KEY = b"lerobot.reward_scoring.schema_version"
_PROVENANCE_KEY = b"lerobot.reward_scoring.provenance"
_DESCRIPTORS_KEY = b"lerobot.reward_scoring.descriptors"
_EPISODE_INDICES_KEY = b"lerobot.reward_scoring.episode_indices"
_LEGACY_SOURCE_KEY = b"lerobot.reward_scoring.legacy_source"
_RESERVED_COLUMNS = ("index", "episode_index", "frame_index")
_DIRECTIONS = {"higher", "lower", "none"}
_COMPARISON_SCOPES = {"episode", "task", "dataset", "global", "none"}
_MISSING_VALUES = {"forbidden", "nan"}


def _require_pyarrow() -> None:
    """Require the parquet dependency only when scoring IO is used."""
    require_package("pyarrow", extra="dataset")
    assert pa is not None and pq is not None


def _canonical_json(value: Any, *, label: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable with finite numeric values") from exc


def _descriptor_payload(descriptors: Mapping[str, SignalDescriptor]) -> dict[str, dict[str, Any]]:
    return {name: asdict(descriptor) for name, descriptor in sorted(descriptors.items())}


def _metadata(
    *,
    descriptors: Mapping[str, SignalDescriptor],
    provenance: Mapping[str, Any],
    episode_indices: Sequence[int],
    legacy_source: str | None = None,
) -> dict[bytes, bytes]:
    metadata = {
        _FORMAT_KEY: SCORING_FORMAT.encode(),
        _SCHEMA_VERSION_KEY: str(SCORING_SCHEMA_VERSION).encode(),
        _PROVENANCE_KEY: _canonical_json(dict(provenance), label="provenance").encode(),
        _DESCRIPTORS_KEY: _canonical_json(
            _descriptor_payload(descriptors), label="signal descriptors"
        ).encode(),
        _EPISODE_INDICES_KEY: _canonical_json(list(episode_indices), label="episode indices").encode(),
    }
    if legacy_source is not None:
        metadata[_LEGACY_SOURCE_KEY] = legacy_source.encode()
    return metadata


def get_signal_descriptors(table: pa.Table) -> dict[str, SignalDescriptor]:
    """Decode signal descriptors attached to a table returned by the reader."""
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
                or not (value.get("unit") is None or isinstance(value.get("unit"), str))
                or not isinstance(value.get("direction"), str)
                or not isinstance(value.get("comparison_scope"), str)
                or not isinstance(value.get("missing_values"), str)
                or not (
                    value.get("bounds") is None
                    or (
                        isinstance(value.get("bounds"), list)
                        and len(value["bounds"]) == 2
                        and all(isinstance(bound, (int, float)) for bound in value["bounds"])
                    )
                )
            ):
                raise TypeError
            descriptors[name] = SignalDescriptor(
                description=value["description"],
                unit=value["unit"],
                direction=value["direction"],
                comparison_scope=value["comparison_scope"],
                missing_values=value["missing_values"],
                bounds=tuple(value["bounds"]) if value["bounds"] is not None else None,
            )
        return descriptors
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Frame-signal descriptors are malformed") from exc


def get_scoring_provenance(table: pa.Table) -> dict[str, Any]:
    """Decode provenance attached to a table returned by the reader."""
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


def _validate_new_artifact(table: pa.Table) -> None:
    _require_pyarrow()
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
    descriptors = get_signal_descriptors(table)
    get_scoring_provenance(table)
    required_columns = set(_RESERVED_COLUMNS)
    missing_columns = required_columns.difference(table.column_names)
    if missing_columns:
        raise ValueError(f"Frame-signal artifact is missing columns: {sorted(missing_columns)}")
    signal_columns = set(table.column_names).difference(required_columns)
    if signal_columns != set(descriptors):
        raise ValueError(
            "Frame-signal columns and descriptors differ: "
            f"columns={sorted(signal_columns)}, descriptors={sorted(descriptors)}"
        )

    try:
        episode_indices = json.loads(metadata[_EPISODE_INDICES_KEY])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Frame-signal episode selection is missing or malformed") from exc
    if (
        not isinstance(episode_indices, list)
        or any(not isinstance(index, int) or isinstance(index, bool) for index in episode_indices)
        or episode_indices != sorted(set(episode_indices))
    ):
        raise ValueError("Frame-signal episode selection must be a sorted list of unique integers")

    for name in _RESERVED_COLUMNS:
        column = table[name]
        if not pa.types.is_int64(column.type) or column.null_count:
            raise ValueError(f"Frame-signal column {name!r} must be non-null int64")

    global_indices = table["index"].to_numpy(zero_copy_only=False)
    row_episode_indices = table["episode_index"].to_numpy(zero_copy_only=False)
    frame_indices = table["frame_index"].to_numpy(zero_copy_only=False)
    observed_episode_order = list(dict.fromkeys(int(index) for index in row_episode_indices))
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
        if not descriptor.description.strip():
            raise ValueError(f"Descriptor for signal {name!r} must have a non-empty description")
        if descriptor.direction not in _DIRECTIONS:
            raise ValueError(f"Descriptor for signal {name!r} has an invalid direction")
        if descriptor.comparison_scope not in _COMPARISON_SCOPES:
            raise ValueError(f"Descriptor for signal {name!r} has an invalid comparison scope")
        if descriptor.missing_values not in _MISSING_VALUES:
            raise ValueError(f"Descriptor for signal {name!r} has an invalid missing-value policy")
        if descriptor.bounds is not None:
            lower, upper = descriptor.bounds
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError(f"Descriptor for signal {name!r} has invalid semantic bounds")

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
                raise ValueError(f"Signal column {name!r} contains infinite values")
            if descriptor.missing_values == "forbidden" and np.isnan(values).any():
                raise ValueError(f"Signal column {name!r} contains forbidden NaN values")
            finite_values = values[np.isfinite(values)]
        else:
            finite_values = values
        if descriptor.bounds is not None and finite_values.size:
            lower, upper = descriptor.bounds
            if np.any(finite_values < lower) or np.any(finite_values > upper):
                raise ValueError(f"Signal column {name!r} contains values outside semantic bounds")


def _read_legacy_progress(table: pa.Table, path: Path) -> pa.Table:
    progress_columns = [name for name in ("progress_sparse", "progress_dense") if name in table.column_names]
    if not progress_columns or any(name not in table.column_names for name in _RESERVED_COLUMNS):
        raise ValueError(f"Not a recognized LeRobot frame-signal artifact: {path}")

    descriptors: dict[str, SignalDescriptor] = {}
    for name in progress_columns:
        values = table[name].to_numpy(zero_copy_only=False)
        descriptors[name] = SignalDescriptor(
            description="Progress imported from a legacy LeRobot reward-model sidecar.",
            unit=None,
            direction="higher",
            comparison_scope="none",
            missing_values="nan"
            if np.issubdtype(values.dtype, np.floating) and np.isnan(values).any()
            else "forbidden",
            bounds=(0.0, 1.0),
        )

    old_metadata = table.schema.metadata or {}
    provenance: dict[str, Any] = {"legacy_artifact": True, "source_path": str(path)}
    reward_model_path = old_metadata.get(b"reward_model_path")
    if reward_model_path is not None:
        provenance["reward_model_path"] = reward_model_path.decode(errors="replace")
    episode_indices = sorted(set(table["episode_index"].to_pylist()))
    normalized = table.replace_schema_metadata(
        _metadata(
            descriptors=descriptors,
            provenance=provenance,
            episode_indices=episode_indices,
            legacy_source="progress_sparse/progress_dense parquet",
        )
    )
    _validate_new_artifact(normalized)
    return normalized


def read_frame_signals(path: str | Path) -> pa.Table:
    """Read a strict scoring artifact or a recognized legacy progress sidecar.

    New artifacts are validated before being returned. Legacy
    ``progress_sparse``/``progress_dense`` parquets keep their columns but gain
    normalized in-memory descriptors and provenance metadata.
    """
    _require_pyarrow()
    artifact_path = Path(path)
    table = pq.read_table(artifact_path)
    metadata = table.schema.metadata or {}
    if metadata.get(_FORMAT_KEY) == SCORING_FORMAT.encode():
        _validate_new_artifact(table)
        return table
    return _read_legacy_progress(table, artifact_path)
