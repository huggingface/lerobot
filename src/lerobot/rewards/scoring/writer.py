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
import logging
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _pyarrow_available

from .schema import (
    RESERVED_COLUMNS,
    SCORING_FORMAT,
    SCORING_SCHEMA_VERSION,
    attach_metadata,
    build_metadata,
    canonical_json,
    decode_episode_indices,
    decode_provenance,
    decode_signal_descriptors,
    require_pyarrow,
    validate_frame_signal_table,
)
from .types import SignalDescriptor

if TYPE_CHECKING or _pyarrow_available:
    import pyarrow as pa
    import pyarrow.parquet as pq
else:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

_MANIFEST_FILENAME = "manifest.json"
logger = logging.getLogger(__name__)


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
    serialized = canonical_json(dict(payload), label="scoring manifest")
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
    require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        pq.write_table(table, temporary_path, compression="snappy", use_dictionary=True)
        with temporary_path.open("rb") as temporary:
            os.fsync(temporary.fileno())
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
        require_pyarrow()
        self.output_path = Path(output_path)
        self.provenance = dict(provenance)
        self.episode_indices = tuple(int(index) for index in episode_indices)
        manifest = _manifest_payload(self.provenance, self.episode_indices)
        # Validate and normalize tuples or other JSON-compatible containers
        # before comparing a resumed run with the on-disk JSON manifest.
        self._expected_manifest = json.loads(canonical_json(manifest, label="scoring manifest"))
        self.provenance = self._expected_manifest["provenance"]
        self._completed_output_frame_count: int | None = None
        self._completed_output_descriptors: dict[str, SignalDescriptor] | None = None
        self._completed_output_signal_types: dict[str, pa.DataType] | None = None
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
            self._load_completed_output()
            return

        parts_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(self._expected_manifest, manifest_path)

    def _load_completed_output(self) -> None:
        """Validate a published output so an identical rerun is a no-op."""
        try:
            table = pq.read_table(self.output_path)
            validate_frame_signal_table(table)
            if decode_provenance(table) != self.provenance:
                raise ValueError("provenance changed")
            if decode_episode_indices(table) != self.episode_indices:
                raise ValueError("episode selection changed")
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Cannot resume scoring because the existing output is invalid or was produced "
                "with different schema, provenance, or episode selection"
            ) from exc

        self._completed_output_frame_count = table.num_rows
        self._completed_output_descriptors = decode_signal_descriptors(table)
        self._completed_output_signal_types = {
            name: table.schema.field(name).type for name in table.column_names if name not in RESERVED_COLUMNS
        }

    @property
    def completed_episode_indices(self) -> set[int]:
        if self._completed_output_frame_count is not None:
            return set(self.episode_indices)

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
        if self._completed_output_descriptors is not None:
            return self._completed_output_descriptors

        descriptors: dict[str, SignalDescriptor] | None = None
        for episode_index in sorted(self.completed_episode_indices):
            table = pq.read_table(_part_path(self.output_path, episode_index))
            current = decode_signal_descriptors(table)
            if descriptors is None:
                descriptors = current
            elif current != descriptors:
                raise ValueError("Committed scoring parts contain inconsistent signal descriptors")
        return descriptors

    @property
    def existing_signal_types(self) -> dict[str, pa.DataType] | None:
        """Return signal column types from committed parts."""
        if self._completed_output_signal_types is not None:
            return self._completed_output_signal_types

        signal_types: dict[str, pa.DataType] | None = None
        for episode_index in sorted(self.completed_episode_indices):
            table = pq.read_table(_part_path(self.output_path, episode_index))
            current = {
                name: table.schema.field(name).type
                for name in table.column_names
                if name not in RESERVED_COLUMNS
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
        table = attach_metadata(
            table,
            descriptors=descriptors,
            provenance=self.provenance,
            episode_indices=(episode_index,),
        )
        validate_frame_signal_table(table)
        _atomic_write_table(table, path)

    def _read_part(
        self,
        episode_index: int,
        descriptors: Mapping[str, SignalDescriptor],
    ) -> pa.Table:
        path = _part_path(self.output_path, episode_index)
        table = pq.read_table(path)
        validate_frame_signal_table(table)
        if decode_episode_indices(table) != (episode_index,):
            raise ValueError(f"Scoring part has the wrong episode selection: {path}")
        if decode_provenance(table) != self.provenance:
            raise ValueError(f"Scoring part has different provenance: {path}")
        if decode_signal_descriptors(table) != descriptors:
            raise ValueError(f"Scoring part has different signal descriptors: {path}")
        return table

    def _stream_parts_to_output(
        self,
        descriptors: Mapping[str, SignalDescriptor],
    ) -> int:
        """Write parts in episode order without materializing the full output."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.output_path.parent,
                prefix=f".{self.output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)

            first_episode = self.episode_indices[0]
            first_table = self._read_part(first_episode, descriptors)
            data_schema = first_table.schema.remove_metadata()
            final_schema = data_schema.with_metadata(
                build_metadata(
                    descriptors=descriptors,
                    provenance=self.provenance,
                    episode_indices=self.episode_indices,
                )
            )

            frame_count = 0
            last_global_index: int | None = None
            with pq.ParquetWriter(
                temporary_path,
                final_schema,
                compression="snappy",
                use_dictionary=True,
            ) as parquet_writer:

                def append_episode(table: pa.Table, episode_index: int) -> None:
                    nonlocal frame_count, last_global_index
                    if not table.schema.remove_metadata().equals(data_schema):
                        raise ValueError(
                            f"Signal schema for episode {episode_index} differs from earlier episodes"
                        )
                    if table.num_rows:
                        first_global_index = int(table["index"][0].as_py())
                        if last_global_index is not None and first_global_index <= last_global_index:
                            raise ValueError("Scoring parts are not strictly ordered by global frame index")
                        last_global_index = int(table["index"][-1].as_py())
                    parquet_writer.write_table(table.replace_schema_metadata(final_schema.metadata))
                    frame_count += table.num_rows

                append_episode(first_table, first_episode)
                del first_table
                for episode_index in self.episode_indices[1:]:
                    table = self._read_part(episode_index, descriptors)
                    append_episode(table, episode_index)
                    del table

            parquet_file = pq.ParquetFile(temporary_path)
            try:
                if parquet_file.metadata.num_rows != frame_count or not parquet_file.schema_arrow.equals(
                    final_schema, check_metadata=True
                ):
                    raise RuntimeError("Final scoring output failed Parquet metadata validation")
            finally:
                parquet_file.close()

            with temporary_path.open("rb") as temporary:
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.output_path)
            return frame_count
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def finalize(self, descriptors: Mapping[str, SignalDescriptor]) -> tuple[Path, int]:
        if self._completed_output_frame_count is not None:
            return self.output_path, self._completed_output_frame_count

        completed = self.completed_episode_indices
        missing = [index for index in self.episode_indices if index not in completed]
        if missing:
            raise RuntimeError(f"Cannot finalize scoring output; missing episode parts: {missing}")

        frame_count = self._stream_parts_to_output(descriptors)
        try:
            shutil.rmtree(_parts_dir(self.output_path))
        except OSError as exc:
            logger.warning("Could not remove completed scoring parts: %s", exc)
        return self.output_path, frame_count
