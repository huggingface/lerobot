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

"""Atomic episode-part writing, resume, and merge for frame signals."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .reader import (
    _RESERVED_COLUMNS,
    SCORING_FORMAT,
    SCORING_SCHEMA_VERSION,
    _canonical_json,
    _metadata,
    _require_pyarrow,
    _validate_new_artifact,
    get_signal_descriptors,
    pa,
    pq,
)
from .types import SignalDescriptor

_MANIFEST_FILENAME = "manifest.json"


def _parts_dir(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.parts")


def _part_path(output_path: Path, episode_index: int) -> Path:
    return _parts_dir(output_path) / f"episode-{episode_index:06d}.parquet"


def _manifest_path(output_path: Path) -> Path:
    return _parts_dir(output_path) / _MANIFEST_FILENAME


def _manifest_payload(provenance: Mapping[str, Any], episode_indices: Sequence[int]) -> dict[str, Any]:
    return {
        "format": SCORING_FORMAT,
        "schema_version": SCORING_SCHEMA_VERSION,
        "provenance": dict(provenance),
        "episode_indices": list(episode_indices),
    }


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    serialized = _canonical_json(dict(payload), label="scoring manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_table(table: pa.Table, path: Path) -> None:
    _require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        pq.write_table(table, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class _FrameSignalsWriter:
    """Internal episode-part writer used by the shared scoring runner."""

    def __init__(
        self,
        output_path: Path,
        *,
        provenance: Mapping[str, Any],
        episode_indices: Sequence[int],
        resume: bool,
    ) -> None:
        _require_pyarrow()
        self.output_path = Path(output_path)
        self.provenance = dict(provenance)
        self.episode_indices = tuple(int(index) for index in episode_indices)
        manifest = _manifest_payload(self.provenance, self.episode_indices)
        # Validate and normalize tuples or other JSON-compatible containers
        # before comparing a resumed run with the on-disk JSON manifest.
        self._expected_manifest = json.loads(_canonical_json(manifest, label="scoring manifest"))
        self.provenance = self._expected_manifest["provenance"]
        self._prepare(resume=resume)

    def _prepare(self, *, resume: bool) -> None:
        parts_dir = _parts_dir(self.output_path)
        manifest_path = _manifest_path(self.output_path)
        has_existing_state = self.output_path.exists() or parts_dir.exists()

        if has_existing_state and not resume:
            raise FileExistsError(
                f"Scoring output already exists at {self.output_path} or {parts_dir}; "
                "choose another output path or enable resume"
            )

        if parts_dir.exists():
            if not manifest_path.is_file():
                raise ValueError(f"Scoring parts directory is missing its manifest: {manifest_path}")
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read scoring manifest: {manifest_path}") from exc
            if existing_manifest != self._expected_manifest:
                raise ValueError(
                    "Cannot resume scoring because schema, provenance, or episode selection changed"
                )
            return

        if self.output_path.exists():
            raise ValueError(f"Cannot resume {self.output_path}: its atomic episode parts are unavailable")

        parts_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(self._expected_manifest, manifest_path)

    @property
    def completed_episode_indices(self) -> set[int]:
        completed: set[int] = set()
        requested = set(self.episode_indices)
        for path in _parts_dir(self.output_path).glob("episode-*.parquet"):
            stem_index = path.stem.removeprefix("episode-")
            if not stem_index.isdigit():
                raise ValueError(f"Unexpected scoring part filename: {path.name}")
            episode_index = int(stem_index)
            if episode_index not in requested:
                raise ValueError(f"Scoring part contains unrequested episode {episode_index}: {path}")
            completed.add(episode_index)
        return completed

    @property
    def existing_descriptors(self) -> dict[str, SignalDescriptor] | None:
        """Return and cross-check descriptors from already committed parts."""
        descriptors: dict[str, SignalDescriptor] | None = None
        for episode_index in sorted(self.completed_episode_indices):
            table = pq.read_table(_part_path(self.output_path, episode_index))
            current = get_signal_descriptors(table)
            if descriptors is None:
                descriptors = current
            elif current != descriptors:
                raise ValueError("Committed scoring parts contain inconsistent signal descriptors")
        return descriptors

    @property
    def existing_signal_types(self) -> dict[str, pa.DataType] | None:
        """Return signal column types from committed parts."""
        signal_types: dict[str, pa.DataType] | None = None
        for episode_index in sorted(self.completed_episode_indices):
            table = pq.read_table(_part_path(self.output_path, episode_index))
            current = {
                name: table.schema.field(name).type
                for name in table.column_names
                if name not in _RESERVED_COLUMNS
            }
            if signal_types is None:
                signal_types = current
            elif current != signal_types:
                raise ValueError("Committed scoring parts contain inconsistent signal dtypes")
        return signal_types

    def write_episode(
        self,
        episode_index: int,
        table: pa.Table,
        descriptors: Mapping[str, SignalDescriptor],
    ) -> None:
        if episode_index not in self.episode_indices:
            raise ValueError(f"Episode {episode_index} was not requested by this scoring run")
        path = _part_path(self.output_path, episode_index)
        if path.exists():
            raise FileExistsError(f"Scoring part already exists: {path}")
        table = table.replace_schema_metadata(
            _metadata(
                descriptors=descriptors,
                provenance=self.provenance,
                episode_indices=(episode_index,),
            )
        )
        _validate_new_artifact(table)
        _atomic_write_table(table, path)

    def finalize(self, descriptors: Mapping[str, SignalDescriptor]) -> Path:
        completed = self.completed_episode_indices
        missing = [index for index in self.episode_indices if index not in completed]
        if missing:
            raise RuntimeError(f"Cannot finalize scoring output; missing episode parts: {missing}")

        tables: list[pa.Table] = []
        for episode_index in self.episode_indices:
            part = pq.read_table(_part_path(self.output_path, episode_index))
            tables.append(part.replace_schema_metadata(None))

        if tables:
            table = pa.concat_tables(tables)
        else:
            table = pa.table(
                {
                    "index": np.asarray([], dtype=np.int64),
                    "episode_index": np.asarray([], dtype=np.int64),
                    "frame_index": np.asarray([], dtype=np.int64),
                    **{name: np.asarray([], dtype=np.float32) for name in sorted(descriptors)},
                }
            )
        table = table.replace_schema_metadata(
            _metadata(
                descriptors=descriptors,
                provenance=self.provenance,
                episode_indices=self.episode_indices,
            )
        )
        _validate_new_artifact(table)
        _atomic_write_table(table, self.output_path)
        return self.output_path
