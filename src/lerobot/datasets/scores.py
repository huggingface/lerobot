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

"""Named frame-aligned tables stored beside a dataset.

A score table adds columns to some frames of a dataset without modifying its data
files. It is written to ``<root>/scores/<name>.parquet`` and joins the dataset on
``(episode_index, frame_index)``. Each table carries a caller-defined ``info`` dict that the
store saves and compares but never interprets.

Episodes are written one at a time to a staging folder, so an interrupted run can resume, and
the table is published only once every selected episode is written.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from lerobot.__version__ import __version__
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE
from lerobot.utils.io_utils import load_json, write_json

from .storage import DEFAULT_STORAGE_FORMAT
from .utils import SCORES_DIR, resolve_episode_indices

if TYPE_CHECKING:
    from .lerobot_dataset import LeRobotDataset

SCORES_FORMAT = "lerobot.scores"
SCORES_SCHEMA_VERSION = 1
RESERVED_COLUMNS = ("index", "episode_index", "frame_index")

_FORMAT_KEY = b"lerobot.scores.format"
_SCHEMA_VERSION_KEY = b"lerobot.scores.schema_version"
_DATASET_KEY = b"lerobot.scores.dataset"
_INFO_KEY = b"lerobot.scores.info"
_MANIFEST_FILENAME = "manifest.json"

logger = logging.getLogger(__name__)


def _canonical_json(value: Any, *, label: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable with finite numeric values") from exc


def _scores_path(dataset: LeRobotDataset, name: str) -> Path:
    if not name.strip() or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Score name must be a non-empty filename component, got {name!r}")
    return Path(dataset.root) / SCORES_DIR / f"{name}.parquet"


def _validate_writable_storage(dataset: LeRobotDataset) -> None:
    """Reject datasets whose root cannot hold score files."""
    if dataset.meta.storage_format != DEFAULT_STORAGE_FORMAT:
        raise NotImplementedError(
            f"Scores are not supported for storage_format={dataset.meta.storage_format!r}; "
            "materialize the dataset in the default local LeRobot format first"
        )
    if dataset._storage_root is not None:
        raise NotImplementedError(
            "Scores cannot be written through a localized metadata-only root; "
            "materialize the dataset locally first"
        )
    if Path(dataset.root).resolve().is_relative_to(Path(HF_LEROBOT_HUB_CACHE).resolve()):
        raise NotImplementedError(
            "Scores cannot be written into the revision-safe Hub snapshot cache; "
            "materialize the dataset into a writable local root first"
        )


def _resolve_episode_indices(dataset: LeRobotDataset, episodes: Sequence[int] | None) -> tuple[int, ...]:
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
        raise ValueError("At least one episode must be selected")
    return tuple(sorted(int(index) for index in selected))


def _build_metadata(dataset_block: Mapping[str, Any], info: Mapping[str, Any]) -> dict[bytes, bytes]:
    return {
        _FORMAT_KEY: SCORES_FORMAT.encode(),
        _SCHEMA_VERSION_KEY: str(SCORES_SCHEMA_VERSION).encode(),
        _DATASET_KEY: _canonical_json(dict(dataset_block), label="dataset identity").encode(),
        _INFO_KEY: _canonical_json(dict(info), label="score info").encode(),
    }


def _column_types(schema: pa.Schema) -> dict[str, pa.DataType]:
    return {field.name: field.type for field in schema if field.name not in RESERVED_COLUMNS}


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


def _write_episode_tables(path: Path, schema: pa.Schema, tables: Iterable[pa.Table]) -> None:
    """Write one row group per episode table, holding a single episode in memory at a time."""
    with _atomic_path(path) as temporary_path:
        writer = pq.ParquetWriter(str(temporary_path), schema, compression="snappy", use_dictionary=True)
        try:
            for table in tables:
                if table.num_rows:
                    # row_group_size >= num_rows guarantees exactly one row group per episode.
                    writer.write_table(table, row_group_size=table.num_rows)
        finally:
            writer.close()


class ScoreWriter:
    """Write a named score table episode by episode, then publish it.

    Each written episode is stored as its own part file in a hidden staging folder next to the
    published table. A new writer with the same name, dataset, ``info`` and episode selection
    resumes from the parts already written.

    Args:
        dataset: Dataset to score. It must use the default storage format in a writable
            local root.
        name: Table name. The table is written to ``<root>/scores/<name>.parquet``.
        info: JSON-serializable description of the table, stored with it and compared on
            resume. The store does not interpret it.
        episodes: Global episode indices to score. Defaults to every episode in the dataset
            view.
        resume: If ``True``, keep the episodes written by an interrupted run with the same
            dataset, ``info`` and episode selection, including replacement runs. Combine
            ``resume=False`` with ``overwrite=True`` to discard staged parts and restart.
        overwrite: If ``True``, allow replacing an existing table once the new one is published.

    Raises:
        RuntimeError: If the dataset is being recorded and not finalized.
        NotImplementedError: If the dataset root cannot hold score files.
        FileExistsError: If the table exists and ``overwrite`` is ``False``, or staged parts
            exist and both ``resume`` and ``overwrite`` are ``False``.
        ValueError: If staged parts were written for a different dataset, ``info`` or
            episode selection.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        name: str,
        *,
        info: Mapping[str, Any],
        episodes: Sequence[int] | None = None,
        resume: bool = True,
        overwrite: bool = False,
    ) -> None:
        dataset._ensure_reader()
        self.output_path = _scores_path(dataset, name)
        _validate_writable_storage(dataset)
        self.episode_indices = _resolve_episode_indices(dataset, episodes)
        self._selected = set(self.episode_indices)
        self.parts_dir = self.output_path.with_name(f".{self.output_path.name}.parts")
        self._dataset = dataset

        dataset_block = {
            "lerobot_version": __version__,
            "repo_id": dataset.repo_id,
            "revision": dataset.revision,
            "total_episodes": int(dataset.meta.total_episodes),
            "total_frames": int(dataset.meta.total_frames),
        }
        # A JSON round trip makes both blocks compare equal to the manifest stored for resume.
        self._dataset_block = json.loads(_canonical_json(dataset_block, label="dataset identity"))
        self._info = json.loads(_canonical_json(dict(info), label="score info"))
        manifest = {
            "format": SCORES_FORMAT,
            "schema_version": SCORES_SCHEMA_VERSION,
            "name": name,
            "dataset": self._dataset_block,
            "info": self._info,
            "episodes": list(self.episode_indices),
        }
        self._prepare(manifest, resume=resume, overwrite=overwrite)

        self._completed = {
            int(path.stem.removeprefix("episode-")) for path in self.parts_dir.glob("episode-*.parquet")
        }
        # Checked per episode so a mismatch fails early instead of after hours of work.
        self._column_types: dict[str, pa.DataType] | None = None
        if self._completed:
            schema = pq.read_schema(self._part_path(min(self._completed)))
            self._column_types = _column_types(schema)

    def _prepare(self, manifest: dict[str, Any], *, resume: bool, overwrite: bool) -> None:
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Published scores already exist at {self.output_path}; "
                "choose another name or set overwrite=True"
            )

        manifest_path = self.parts_dir / _MANIFEST_FILENAME
        if self.parts_dir.exists():
            if resume:
                if not manifest_path.is_file() or load_json(manifest_path) != manifest:
                    raise ValueError(
                        "Cannot resume scoring because the dataset, info or episode selection changed; "
                        "set overwrite=True and resume=False to restart"
                    )
                return
            if not overwrite:
                raise FileExistsError(
                    f"Unpublished score parts exist at {self.parts_dir}; set resume=True or overwrite=True"
                )
            shutil.rmtree(self.parts_dir)

        self.parts_dir.mkdir(parents=True)
        with _atomic_path(manifest_path) as temporary_path:
            write_json(manifest, temporary_path)

    def _part_path(self, episode_index: int) -> Path:
        return self.parts_dir / f"episode-{episode_index:06d}.parquet"

    @property
    def pending_episodes(self) -> tuple[int, ...]:
        """Selected episodes, in ascending order, that have not been written yet."""
        return tuple(index for index in self.episode_indices if index not in self._completed)

    def write_episode(
        self,
        episode_index: int,
        frame_indices: Sequence[int] | np.ndarray,
        columns: Mapping[str, Sequence[Any] | np.ndarray],
    ) -> None:
        """Write the score rows of one episode.

        Args:
            episode_index: Global index of a selected episode.
            frame_indices: One-dimensional integer array of frame positions inside the episode,
                in non-decreasing order. A frame may appear several times.
            columns: One value per row for each column. Every episode must have the same
                column names and types.

        Raises:
            ValueError: If the episode is not selected or the rows do not fit the episode or
                the columns of earlier episodes.
        """
        if episode_index not in self._selected:
            raise ValueError(f"Episode {episode_index} is not in the selected episodes")
        episode = self._dataset.meta.episodes[episode_index]
        episode_start = int(episode["dataset_from_index"])
        episode_length = int(episode["dataset_to_index"]) - episode_start

        indices = np.asarray(frame_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("frame_indices must be a one-dimensional integer array")
        if indices.size and (
            indices[0] < 0 or indices[-1] >= episode_length or np.any(indices[1:] < indices[:-1])
        ):
            raise ValueError(
                f"Frame indices for episode {episode_index} must be non-decreasing and in [0, {episode_length})"
            )
        if not columns:
            raise ValueError("At least one column must be given")
        conflicting = set(columns).intersection(RESERVED_COLUMNS)
        if conflicting:
            raise ValueError(f"Column names conflict with reserved columns: {sorted(conflicting)}")
        for name, values in columns.items():
            if len(values) != len(indices):
                raise ValueError(f"Column {name!r} must have {len(indices)} values, got {len(values)}")

        indices = indices.astype(np.int64, copy=False)
        table = pa.table(
            {
                "index": episode_start + indices,
                "episode_index": np.full(indices.shape, episode_index, dtype=np.int64),
                "frame_index": indices,
                **{name: columns[name] for name in sorted(columns)},
            }
        )
        column_types = _column_types(table.schema)
        if self._column_types is None:
            self._column_types = column_types
        elif column_types != self._column_types:
            raise ValueError(
                f"Column names or types for episode {episode_index} differ from earlier episodes"
            )

        table = table.replace_schema_metadata(_build_metadata(self._dataset_block, self._info))
        with _atomic_path(self._part_path(episode_index)) as temporary_path:
            pq.write_table(table, temporary_path, compression="snappy", use_dictionary=True)
        self._completed.add(episode_index)

    def finalize(self) -> None:
        """Publish the written episodes, in episode order, as one table and remove the parts.

        Raises:
            RuntimeError: If a selected episode has not been written.
        """
        missing = list(self.pending_episodes)
        if missing:
            raise RuntimeError(f"Cannot publish scores; missing episodes: {missing}")

        schema = pq.read_schema(self._part_path(self.episode_indices[0]))
        tables = (pq.read_table(self._part_path(index)) for index in self.episode_indices)
        _write_episode_tables(self.output_path, schema, tables)
        shutil.rmtree(self.parts_dir, ignore_errors=True)
        logger.info("Published scores to %s", self.output_path)


def add_scores(
    dataset: LeRobotDataset,
    name: str,
    function: Callable[[int], tuple[Sequence[int] | np.ndarray, Mapping[str, Sequence[Any] | np.ndarray]]],
    *,
    info: Mapping[str, Any],
    episodes: Sequence[int] | None = None,
    resume: bool = True,
    overwrite: bool = False,
) -> None:
    """Score episodes with ``function`` and publish the result as a named table.

    ``function`` is called once per episode that still needs work, in ascending episode order.

    Args:
        dataset: Dataset to score.
        name: Table name.
        function: Called as ``function(episode_index)``; returns ``(frame_indices, columns)``
            as accepted by :meth:`ScoreWriter.write_episode`.
        info: JSON-serializable description of the table, stored with it.
        episodes: Global episode indices to score. Defaults to every episode in the dataset
            view.
        resume: If ``True``, skip episodes written by an interrupted run with the same
            dataset, ``info`` and episode selection, including replacement runs. Combine
            ``resume=False`` with ``overwrite=True`` to discard staged parts and restart.
        overwrite: If ``True``, replace an existing table once the new one is published.
    """
    writer = ScoreWriter(dataset, name, info=info, episodes=episodes, resume=resume, overwrite=overwrite)
    for episode_index in writer.pending_episodes:
        frame_indices, columns = function(episode_index)
        writer.write_episode(episode_index, frame_indices, columns)
    writer.finalize()


def _read_schema(path: Path) -> pa.Schema:
    """Read and check a score table's schema without loading its rows."""
    schema = pq.read_schema(path)
    metadata = schema.metadata or {}
    if metadata.get(_FORMAT_KEY) != SCORES_FORMAT.encode():
        raise ValueError(f"Not a LeRobot score table: {path}")
    version = metadata.get(_SCHEMA_VERSION_KEY)
    if version != str(SCORES_SCHEMA_VERSION).encode():
        raise ValueError(f"Unsupported score schema version: {version!r}")
    return schema


def load_scores(dataset: LeRobotDataset, name: str, columns: Sequence[str] | None = None) -> datasets.Dataset:
    """Load a named score table.

    Args:
        dataset: Dataset the table belongs to.
        name: Table name.
        columns: Columns to load. Defaults to every column, including ``index``,
            ``episode_index`` and ``frame_index``.

    Returns:
        The table as a Hugging Face dataset.
    """
    path = _scores_path(dataset, name)
    schema = _read_schema(path)
    table = pq.read_table(path, columns=None if columns is None else list(columns))
    return datasets.Dataset(table.replace_schema_metadata(schema.metadata))


def load_scores_info(dataset: LeRobotDataset, name: str) -> dict[str, Any]:
    """Return the ``info`` dict a named score table was written with."""
    return json.loads(_read_schema(_scores_path(dataset, name)).metadata[_INFO_KEY])


def list_scores(dataset: LeRobotDataset) -> list[str]:
    """List the names of the score tables published for a dataset."""
    directory = Path(dataset.root) / SCORES_DIR
    return sorted(path.stem for path in directory.glob("*.parquet") if path.is_file())


def merge_scores(
    dataset: LeRobotDataset, names: Sequence[str], *, name: str, overwrite: bool = False
) -> None:
    """Merge score tables of disjoint episodes into one named table.

    Useful when episodes were scored in parallel, each worker with its own table name. The
    merged tables are kept.

    Args:
        dataset: Dataset the tables belong to.
        names: Tables to merge. They must have the same columns, dataset identity and ``info``,
            and share no episode.
        name: Name of the merged table.
        overwrite: If ``True``, replace an existing table with that name.

    Raises:
        FileExistsError: If the table exists and ``overwrite`` is ``False``.
        ValueError: If the tables differ in columns, dataset identity or ``info``, or share an
            episode.
    """
    if not names:
        raise ValueError("At least one score table must be given to merge")
    output_path = _scores_path(dataset, name)
    _validate_writable_storage(dataset)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Published scores already exist at {output_path}; choose another name or set overwrite=True"
        )

    paths = [_scores_path(dataset, part) for part in names]
    schemas = [_read_schema(path) for path in paths]
    reference = schemas[0]
    episode_paths: dict[int, Path] = {}
    for part, path, schema in zip(names, paths, schemas, strict=True):
        # Equal schemas, metadata included, mean equal columns, dtypes, dataset identity and info.
        if not schema.equals(reference, check_metadata=True):
            raise ValueError(f"Scores {part!r} have different columns or info than {names[0]!r}")
        episodes = np.unique(
            pq.read_table(path, columns=["episode_index"]).column("episode_index").to_numpy()
        )
        shared = sorted(int(index) for index in episodes if int(index) in episode_paths)
        if shared:
            raise ValueError(f"Scores {part!r} repeat episodes of earlier tables: {shared}")
        episode_paths.update((int(index), path) for index in episodes)

    tables = (
        pq.read_table(episode_paths[index], filters=[("episode_index", "==", index)], schema=reference)
        for index in sorted(episode_paths)
    )
    _write_episode_tables(output_path, reference, tables)
    logger.info("Published merged scores %r to %s", name, output_path)


def push_scores_to_hub(dataset: LeRobotDataset, name: str) -> None:
    """Upload one named score table to the dataset repository on the Hub."""
    local_path = _scores_path(dataset, name)
    if not local_path.is_file():
        raise FileNotFoundError(f"Scores {name!r} do not exist")

    path_in_repo = f"{SCORES_DIR}/{local_path.name}"
    HfApi().upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=dataset.repo_id,
        repo_type="dataset",
        commit_message=f"Upload scores {name}",
    )
    logger.info("Uploaded scores %s to hf://datasets/%s/%s", name, dataset.repo_id, path_in_repo)
