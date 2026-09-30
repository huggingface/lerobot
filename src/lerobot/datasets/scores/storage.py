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

"""Storage for frame-aligned dataset score sidecars."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from lerobot.__version__ import __version__
from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
from lerobot.datasets.storage import DEFAULT_STORAGE_FORMAT
from lerobot.datasets.utils import SCORES_DIR, resolve_episode_indices
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE
from lerobot.utils.io_utils import load_json, write_json

from .types import FrameScorer, FrameSignals, SignalDescriptor

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

SCORING_FORMAT = "lerobot.frame_signals"
SCORING_SCHEMA_VERSION = 1
RESERVED_COLUMNS = ("index", "episode_index", "frame_index")

_FORMAT_KEY = b"lerobot.scores.format"
_SCHEMA_VERSION_KEY = b"lerobot.scores.schema_version"
_PROVENANCE_KEY = b"lerobot.scores.provenance"
_DESCRIPTORS_KEY = b"lerobot.scores.descriptors"
_LEGACY_SOURCE_KEY = b"lerobot.scores.legacy_source"
_MANIFEST_FILENAME = "manifest.json"
_LEGACY_PROGRESS_COLUMNS = ("progress_sparse", "progress_dense")
_SIGNAL_DIRECTIONS = {"higher", "lower", "none"}

logger = logging.getLogger(__name__)


def _canonical_json(value: Any, *, label: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable with finite numeric values") from exc


def _score_path(dataset: LeRobotDataset, name: str) -> Path:
    if not name.strip() or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Score name must be a non-empty filename component, got {name!r}")
    return Path(dataset.root) / SCORES_DIR / f"{name}.parquet"


def _validate_writable_storage(dataset: LeRobotDataset) -> None:
    """Reject datasets whose root cannot hold score files."""
    if dataset.meta.storage_format != DEFAULT_STORAGE_FORMAT:
        raise NotImplementedError(
            f"Dataset scores are not supported for storage_format={dataset.meta.storage_format!r}; "
            "materialize the dataset in the default local LeRobot format first"
        )
    if dataset._storage_root is not None:
        raise NotImplementedError(
            "Dataset scores cannot be written through a localized metadata-only root; "
            "materialize the dataset locally first"
        )
    if Path(dataset.root).resolve().is_relative_to(Path(HF_LEROBOT_HUB_CACHE).resolve()):
        raise NotImplementedError(
            "Dataset scores cannot be written into the revision-safe Hub snapshot cache; "
            "materialize the dataset into a writable local root first"
        )


def _resolve_episode_indices(dataset: LeRobotDataset, episodes: list[int] | None) -> tuple[int, ...]:
    """Resolve global episode IDs within the current dataset view."""
    view = resolve_episode_indices(dataset.episodes, dataset.meta.total_episodes)
    visible = set(range(dataset.meta.total_episodes)) if view is None else set(view)
    if episodes is None:
        selected = visible
    else:
        if len(set(episodes)) != len(episodes):
            raise ValueError("episodes must not contain duplicates")
        selected = set(episodes)
        outside_view = sorted(selected - visible)
        if outside_view:
            raise ValueError(
                "Explicit episodes must be global IDs visible in the current dataset view; "
                f"not visible: {outside_view}"
            )
    if not selected:
        raise ValueError("At least one episode must be selected for scoring")
    return tuple(sorted(int(index) for index in selected))


def _build_provenance(dataset: LeRobotDataset, scorer: FrameScorer) -> dict[str, Any]:
    provenance = {
        "lerobot_version": __version__,
        "dataset": {
            "repo_id": dataset.repo_id,
            "revision": dataset.revision,
            "total_episodes": int(dataset.meta.total_episodes),
            "total_frames": int(dataset.meta.total_frames),
        },
        "scoring": dict(scorer.provenance),
    }
    # A JSON round trip makes it compare equal to the manifest stored for resume.
    return json.loads(_canonical_json(provenance, label="score provenance"))


def _validate_frame_signals(frame_signals: FrameSignals, *, episode_index: int, episode_length: int) -> None:
    """Check that one episode's signals align with its frames."""
    frame_indices = frame_signals.frame_indices
    if frame_indices.ndim != 1 or frame_indices.dtype.kind not in "iu":
        raise ValueError("FrameSignals.frame_indices must be a one-dimensional integer array")
    if frame_indices.size and (
        frame_indices[0] < 0 or frame_indices[-1] >= episode_length or np.any(np.diff(frame_indices) <= 0)
    ):
        raise ValueError(
            f"Frame indices for episode {episode_index} must be unique, increasing, and in [0, {episode_length})"
        )

    names = set(frame_signals.signals)
    if not names:
        raise ValueError("FrameSignals must contain at least one signal")
    if names != set(frame_signals.descriptors):
        raise ValueError(
            "FrameSignals signals and descriptors must have identical names: "
            f"signals={sorted(names)}, descriptors={sorted(frame_signals.descriptors)}"
        )
    conflicting = names.intersection(RESERVED_COLUMNS)
    if conflicting:
        raise ValueError(f"Signal names conflict with reserved columns: {sorted(conflicting)}")

    for name in sorted(names):
        values = frame_signals.signals[name]
        if values.shape != frame_indices.shape:
            raise ValueError(f"Signal {name!r} must have shape {frame_indices.shape}, got {values.shape}")
        if values.dtype.kind not in "biuf":
            raise ValueError(f"Signal {name!r} must have a bool, integer, or floating dtype")
        direction = frame_signals.descriptors[name].direction
        if direction not in _SIGNAL_DIRECTIONS:
            raise ValueError(f"Invalid direction for signal {name!r}: {direction!r}")


def _build_episode_table(episode_index: int, episode_start: int, frame_signals: FrameSignals) -> pa.Table:
    frame_indices = frame_signals.frame_indices.astype(np.int64, copy=False)
    return pa.table(
        {
            "index": episode_start + frame_indices,
            "episode_index": np.full(frame_indices.shape, episode_index, dtype=np.int64),
            "frame_index": frame_indices,
            **{name: frame_signals.signals[name] for name in sorted(frame_signals.signals)},
        }
    )


def _signal_types(schema: pa.Schema) -> dict[str, pa.DataType]:
    return {field.name: field.type for field in schema if field.name not in RESERVED_COLUMNS}


def _build_metadata(
    *,
    descriptors: Mapping[str, SignalDescriptor],
    provenance: Mapping[str, Any],
    legacy_source: str | None = None,
) -> dict[bytes, bytes]:
    descriptor_payload = {name: asdict(descriptor) for name, descriptor in sorted(descriptors.items())}
    metadata = {
        _FORMAT_KEY: SCORING_FORMAT.encode(),
        _SCHEMA_VERSION_KEY: str(SCORING_SCHEMA_VERSION).encode(),
        _PROVENANCE_KEY: _canonical_json(dict(provenance), label="score provenance").encode(),
        _DESCRIPTORS_KEY: _canonical_json(descriptor_payload, label="signal descriptors").encode(),
    }
    if legacy_source is not None:
        metadata[_LEGACY_SOURCE_KEY] = legacy_source.encode()
    return metadata


def _decode_descriptors(schema: pa.Schema) -> dict[str, SignalDescriptor]:
    payload = json.loads(schema.metadata[_DESCRIPTORS_KEY])
    descriptors = {}
    for name, fields in payload.items():
        bounds = fields.pop("bounds")
        descriptors[name] = SignalDescriptor(**fields, bounds=None if bounds is None else tuple(bounds))
    return descriptors


def _decode_provenance(schema: pa.Schema) -> dict[str, Any]:
    return json.loads(schema.metadata[_PROVENANCE_KEY])


@contextmanager
def _atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary path that replaces ``path`` only if the block succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        yield temporary_path
        # Flush to disk so a finished part survives a machine crash, not just a process crash.
        with temporary_path.open("rb") as written:
            os.fsync(written.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class _ScoreWriter:
    """Write resumable episode parts, then publish them as one sidecar."""

    def __init__(
        self,
        output_path: Path,
        *,
        name: str,
        provenance: dict[str, Any],
        episode_indices: tuple[int, ...],
        resume: bool,
        overwrite: bool,
    ) -> None:
        self.output_path = output_path
        self.parts_dir = output_path.with_name(f".{output_path.name}.parts")
        self.provenance = provenance
        self.episode_indices = episode_indices
        manifest = {
            "format": SCORING_FORMAT,
            "schema_version": SCORING_SCHEMA_VERSION,
            "score_name": name,
            "provenance": provenance,
            "episodes": list(episode_indices),
        }
        self._prepare(manifest, resume=resume, overwrite=overwrite)
        self.completed_episode_indices = {
            int(path.stem.removeprefix("episode-")) for path in self.parts_dir.glob("episode-*.parquet")
        }

    def _prepare(self, manifest: dict[str, Any], *, resume: bool, overwrite: bool) -> None:
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Published dataset score already exists at {self.output_path}; "
                "choose another name or set overwrite=True"
            )

        manifest_path = self.parts_dir / _MANIFEST_FILENAME
        if overwrite:
            shutil.rmtree(self.parts_dir, ignore_errors=True)
        elif self.parts_dir.exists():
            if not resume:
                raise FileExistsError(
                    f"Unpublished score staging state exists at {self.parts_dir}; "
                    "set resume=True or overwrite=True"
                )
            if not manifest_path.is_file() or load_json(manifest_path) != manifest:
                raise ValueError("Cannot resume dataset scoring because the selection or provenance changed")
            return

        self.parts_dir.mkdir(parents=True)
        with _atomic_path(manifest_path) as temporary_path:
            write_json(manifest, temporary_path)

    def _part_path(self, episode_index: int) -> Path:
        return self.parts_dir / f"episode-{episode_index:06d}.parquet"

    def reference_signals(
        self,
    ) -> tuple[dict[str, SignalDescriptor] | None, dict[str, pa.DataType] | None]:
        """Return descriptors and signal dtypes of the first completed part, if any."""
        if not self.completed_episode_indices:
            return None, None
        schema = pq.read_schema(self._part_path(min(self.completed_episode_indices)))
        return _decode_descriptors(schema), _signal_types(schema)

    def write_episode(
        self, episode_index: int, table: pa.Table, descriptors: Mapping[str, SignalDescriptor]
    ) -> None:
        table = table.replace_schema_metadata(
            _build_metadata(descriptors=descriptors, provenance=self.provenance)
        )
        with _atomic_path(self._part_path(episode_index)) as temporary_path:
            pq.write_table(table, temporary_path, compression="snappy", use_dictionary=True)
        self.completed_episode_indices.add(episode_index)

    def finalize(self, descriptors: Mapping[str, SignalDescriptor]) -> None:
        """Combine the parts in episode order into the published sidecar."""
        missing = [index for index in self.episode_indices if index not in self.completed_episode_indices]
        if missing:
            raise RuntimeError(f"Cannot publish dataset score; missing episode parts: {missing}")

        parts = [pq.read_table(self._part_path(index)) for index in self.episode_indices]
        table = pa.concat_tables(parts).replace_schema_metadata(
            _build_metadata(descriptors=descriptors, provenance=self.provenance)
        )
        with _atomic_path(self.output_path) as temporary_path:
            write_table_one_row_group_per_episode(table, temporary_path)
        shutil.rmtree(self.parts_dir, ignore_errors=True)


def _legacy_metadata(schema: pa.Schema, path: Path) -> dict[bytes, bytes]:
    """Describe an older progress_sparse / progress_dense sidecar, which has no score metadata."""
    progress_columns = [name for name in _LEGACY_PROGRESS_COLUMNS if name in schema.names]
    if not progress_columns or set(schema.names) != {*RESERVED_COLUMNS, *progress_columns}:
        raise ValueError(f"Not a recognized LeRobot dataset score sidecar: {path}")

    # The SARM and TOPReward scripts write NaN for frames they skip.
    descriptor = SignalDescriptor(
        description="Progress imported from a legacy LeRobot reward-model sidecar.",
        direction="higher",
        allow_nan=True,
    )
    scoring: dict[str, Any] = {"legacy_output": True, "source_path": str(path)}
    reward_model_path = (schema.metadata or {}).get(b"reward_model_path")
    if reward_model_path is not None:
        scoring["reward_model_path"] = reward_model_path.decode(errors="replace")
    return _build_metadata(
        descriptors=dict.fromkeys(progress_columns, descriptor),
        provenance={"scoring": scoring},
        legacy_source="progress_sparse/progress_dense parquet",
    )


def _read_score_schema(path: Path) -> pa.Schema:
    """Read and check a score file's schema without loading its rows."""
    schema = pq.read_schema(path)
    metadata = schema.metadata or {}
    if metadata.get(_FORMAT_KEY) != SCORING_FORMAT.encode():
        return schema.with_metadata(_legacy_metadata(schema, path))

    version = metadata.get(_SCHEMA_VERSION_KEY)
    if version != str(SCORING_SCHEMA_VERSION).encode():
        raise ValueError(f"Unsupported dataset score schema version: {version!r}")
    signal_columns = set(schema.names).difference(RESERVED_COLUMNS)
    descriptor_names = set(_decode_descriptors(schema))
    if signal_columns != descriptor_names:
        raise ValueError(
            "Dataset score columns and descriptors differ: "
            f"columns={sorted(signal_columns)}, descriptors={sorted(descriptor_names)}"
        )
    return schema


def _read_score_path(path: str | Path) -> pa.Table:
    """Read a current or recognized legacy score sidecar by path."""
    path = Path(path)
    schema = _read_score_schema(path)
    return pq.read_table(path).replace_schema_metadata(schema.metadata)


class DatasetScoreStorage:
    """Store derived frame scores owned by a LeRobot dataset."""

    def __init__(self, dataset: LeRobotDataset) -> None:
        self._dataset = dataset

    def add(
        self,
        scorer: FrameScorer,
        *,
        name: str | None = None,
        episodes: list[int] | None = None,
        resume: bool = True,
        overwrite: bool = False,
    ) -> None:
        score_name = scorer.name if name is None else name
        output_path = _score_path(self._dataset, score_name)
        _validate_writable_storage(self._dataset)
        selected = _resolve_episode_indices(self._dataset, episodes)
        writer = _ScoreWriter(
            output_path,
            name=score_name,
            provenance=_build_provenance(self._dataset, scorer),
            episode_indices=selected,
            resume=resume,
            overwrite=overwrite,
        )

        # Checked per episode so a mismatch fails early instead of after hours of scoring.
        descriptors, signal_types = writer.reference_signals()
        for episode_index in selected:
            if episode_index in writer.completed_episode_indices:
                continue
            episode = self._dataset.meta.episodes[episode_index]
            episode_start = int(episode["dataset_from_index"])
            episode_length = int(episode["dataset_to_index"]) - episode_start

            frame_signals = scorer.score_episode(self._dataset, episode_index)
            _validate_frame_signals(frame_signals, episode_index=episode_index, episode_length=episode_length)
            table = _build_episode_table(episode_index, episode_start, frame_signals)
            if descriptors is None:
                descriptors, signal_types = dict(frame_signals.descriptors), _signal_types(table.schema)
            elif (
                dict(frame_signals.descriptors) != descriptors or _signal_types(table.schema) != signal_types
            ):
                raise ValueError(
                    f"Signal descriptors or dtypes for episode {episode_index} differ from earlier episodes"
                )
            writer.write_episode(episode_index, table, descriptors)

        if descriptors is None:
            raise RuntimeError("Dataset scoring produced no signal descriptors")
        writer.finalize(descriptors)
        logger.info("Published dataset score %r to %s", score_name, output_path)

    def read(self, name: str) -> datasets.Dataset:
        return datasets.Dataset(_read_score_path(_score_path(self._dataset, name)))

    def list(self) -> list[str]:
        score_dir = Path(self._dataset.root) / SCORES_DIR
        return sorted(path.stem for path in score_dir.glob("*.parquet") if path.is_file())

    def descriptors(self, name: str) -> dict[str, SignalDescriptor]:
        return _decode_descriptors(_read_score_schema(_score_path(self._dataset, name)))

    def provenance(self, name: str) -> dict[str, Any]:
        return _decode_provenance(_read_score_schema(_score_path(self._dataset, name)))

    def push_to_hub(self, name: str) -> None:
        local_path = _score_path(self._dataset, name)
        if not local_path.is_file():
            raise FileNotFoundError(f"Dataset score {name!r} does not exist")

        path_in_repo = f"{SCORES_DIR}/{local_path.name}"
        HfApi().upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=self._dataset.repo_id,
            repo_type="dataset",
            commit_message=f"Upload dataset score {name}",
        )
        logger.info(
            "Uploaded dataset score %s to hf://datasets/%s/%s", name, self._dataset.repo_id, path_in_repo
        )
