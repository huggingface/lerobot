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

"""Named sets of per-frame scores stored beside a dataset.

Scores add columns to some frames of a dataset without modifying its data files. Each named
set of scores is one Parquet file, ``<root>/scores/<name>.parquet``, that joins the dataset on
``(episode_index, frame_index)``. Each file carries a caller-defined ``info`` dict that the
store saves and compares but never interprets.

Episodes are written one at a time to a staging folder, so an interrupted run can resume, and
the file is published only once every selected episode is written.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.__version__ import __version__
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE
from lerobot.utils.io_utils import load_json, write_json

from .dataset_metadata import CODEBASE_VERSION
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
    if not name.strip() or name.startswith(".") or "/" in name or "\\" in name:
        raise ValueError(f"Score name must be a non-empty filename without a leading dot, got {name!r}")
    return Path(dataset.root) / SCORES_DIR / f"{name}.parquet"


def _validate_writable_storage(dataset: LeRobotDataset) -> None:
    """Reject datasets whose root cannot hold score files."""
    if dataset.meta.storage_format != DEFAULT_STORAGE_FORMAT:
        raise NotImplementedError(
            f"Scores are not supported for storage_format={dataset.meta.storage_format!r}; "
            "materialize the dataset in the default local LeRobot format first"
        )
    if Path(dataset.root).resolve().is_relative_to(Path(HF_LEROBOT_HUB_CACHE).resolve()):
        raise NotImplementedError(
            "Scores cannot be written into the revision-safe Hub snapshot cache; "
            "materialize the dataset into a writable local root first"
        )


def _resolve_episode_indices(dataset: LeRobotDataset, episodes: Sequence[int] | None) -> set[int]:
    """Resolve global episode IDs within the current dataset view."""
    view = resolve_episode_indices(dataset.episodes, dataset.meta.total_episodes)
    visible = set(range(dataset.meta.total_episodes)) if view is None else set(view)
    if episodes is None:
        selected = visible
    else:
        selected = set(episodes)
        if len(selected) != len(episodes):
            raise ValueError("episodes must not contain duplicates")
        outside_view = sorted(selected - visible)
        if outside_view:
            raise ValueError(
                "Explicit episodes must be global IDs visible in the current dataset view; "
                f"not visible: {outside_view}"
            )
    if not selected:
        raise ValueError("At least one episode must be selected")
    return {int(index) for index in selected}


def _build_metadata(dataset_metadata: Mapping[str, Any], info: Mapping[str, Any]) -> dict[bytes, bytes]:
    return {
        _FORMAT_KEY: SCORES_FORMAT.encode(),
        _SCHEMA_VERSION_KEY: str(SCORES_SCHEMA_VERSION).encode(),
        _DATASET_KEY: _canonical_json(dict(dataset_metadata), label="dataset identity").encode(),
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
        # Flush the completed file before publishing it.
        with temporary_path.open("rb") as written:
            os.fsync(written.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
    """Write named scores episode by episode, then publish them as one Parquet file.

    Each written episode is stored as its own part file in a hidden staging folder next to the
    published file. A new writer with the same name, dataset, ``info`` and episode selection
    resumes from the parts already written.

    Args:
        dataset: Dataset to score. It must use the default storage format in a writable
            local root.
        name: Score name. The scores are written to ``<root>/scores/<name>.parquet``.
        info: JSON-serializable description of the scores, stored with them and compared on
            resume. The store does not interpret it.
        episodes: Global episode indices to score. Defaults to every episode in the dataset
            view.
        resume: If ``True``, keep the episodes written by an interrupted run with the same
            dataset, ``info`` and episode selection, including replacement runs. Combine
            ``resume=False`` with ``overwrite=True`` to discard staged parts and restart.
        overwrite: If ``True``, allow replacing existing scores once the new ones are published.

    Raises:
        RuntimeError: If the dataset is being recorded and not finalized.
        NotImplementedError: If the dataset root cannot hold score files.
        FileExistsError: If the scores exist and ``overwrite`` is ``False``, or staged parts
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
        self._selected = _resolve_episode_indices(dataset, episodes)
        self.episode_indices = tuple(sorted(self._selected))
        self.parts_dir = self.output_path.with_name(f".{self.output_path.name}.parts")
        self._dataset = dataset
        self._finalized = False

        self._dataset_metadata = {
            "lerobot_version": __version__,
            "repo_id": dataset.repo_id,
            "revision": dataset.revision,
            "total_episodes": int(dataset.meta.total_episodes),
            "total_frames": int(dataset.meta.total_frames),
        }
        # Normalize info so it compares equal to the JSON manifest stored for resume.
        self._info = json.loads(_canonical_json(dict(info), label="score info"))
        # Selection-specific ranges belong in the manifest, not the metadata shared by merged scores.
        episode_ranges: list[list[int]] = []
        for index in self.episode_indices:
            episode = dataset.meta.episodes[index]
            episode_ranges.append([int(episode["dataset_from_index"]), int(episode["dataset_to_index"])])
        manifest = {
            "format": SCORES_FORMAT,
            "schema_version": SCORES_SCHEMA_VERSION,
            "name": name,
            "dataset": self._dataset_metadata,
            "info": self._info,
            "episodes": list(self.episode_indices),
            "episode_ranges": episode_ranges,
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
            RuntimeError: If the writer has already been finalized.
            ValueError: If the episode is not selected or the rows do not fit the episode or
                the columns of earlier episodes.
        """
        if self._finalized:
            raise RuntimeError("Cannot write episodes after finalization; create a new ScoreWriter")
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

        table = table.replace_schema_metadata(_build_metadata(self._dataset_metadata, self._info))
        with _atomic_path(self._part_path(episode_index)) as temporary_path:
            pq.write_table(table, temporary_path, compression="snappy", use_dictionary=True)
        self._completed.add(episode_index)

    def finalize(self) -> None:
        """Publish the written episodes, in episode order, as one Parquet file and remove the parts.

        Safe to call multiple times after successful publication. Staging cleanup failures are logged.

        Raises:
            RuntimeError: If a selected episode has not been written.
        """
        if self._finalized:
            return
        missing = list(self.pending_episodes)
        if missing:
            raise RuntimeError(f"Cannot publish scores; missing episodes: {missing}")

        schema = pq.read_schema(self._part_path(self.episode_indices[0]))
        tables = (pq.read_table(self._part_path(index)) for index in self.episode_indices)
        _write_episode_tables(self.output_path, schema, tables)
        self._finalized = True
        try:
            shutil.rmtree(self.parts_dir)
        except OSError as error:
            logger.warning("Could not remove score staging directory %s: %s", self.parts_dir, error)
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
    """Score episodes with ``function`` and publish the result under a name.

    ``function`` is called once per episode that still needs work, in ascending episode order.

    Args:
        dataset: Dataset to score.
        name: Score name.
        function: Called as ``function(episode_index)``; returns ``(frame_indices, columns)``
            as accepted by :meth:`ScoreWriter.write_episode`.
        info: JSON-serializable description of the scores, stored with them.
        episodes: Global episode indices to score. Defaults to every episode in the dataset
            view.
        resume: If ``True``, skip episodes written by an interrupted run with the same
            dataset, ``info`` and episode selection, including replacement runs. Combine
            ``resume=False`` with ``overwrite=True`` to discard staged parts and restart.
        overwrite: If ``True``, replace existing scores once the new ones are published.
    """
    writer = ScoreWriter(dataset, name, info=info, episodes=episodes, resume=resume, overwrite=overwrite)
    for episode_index in writer.pending_episodes:
        frame_indices, columns = function(episode_index)
        writer.write_episode(episode_index, frame_indices, columns)
    writer.finalize()


def _read_schema(path: Path) -> pa.Schema:
    """Read and check a score file's schema without loading its rows."""
    schema = pq.read_schema(path)
    metadata = schema.metadata or {}
    if metadata.get(_FORMAT_KEY) != SCORES_FORMAT.encode():
        raise ValueError(f"Not a LeRobot score file: {path}")
    version = metadata.get(_SCHEMA_VERSION_KEY)
    if version != str(SCORES_SCHEMA_VERSION).encode():
        raise ValueError(f"Unsupported score schema version: {version!r}")
    return schema


def _get_scores_path(dataset: LeRobotDataset, name: str, *, download: bool, token: str | bool | None) -> Path:
    path = _scores_path(dataset, name)
    if path.is_file() or not download:
        return path
    root = Path(dataset.root)
    in_snapshot = root.parent.name == "snapshots" and root.is_relative_to(HF_LEROBOT_HUB_CACHE)
    location = {"cache_dir": HF_LEROBOT_HUB_CACHE} if in_snapshot else {"local_dir": root}
    downloaded_root = Path(
        snapshot_download(
            dataset.repo_id,
            repo_type="dataset",
            revision=root.name if in_snapshot else dataset.revision,
            allow_patterns=[f"{SCORES_DIR}/{path.name}"],
            token=token,
            **location,
        )
    )
    return downloaded_root / SCORES_DIR / path.name


def load_scores(
    dataset: LeRobotDataset,
    name: str,
    columns: Sequence[str] | None = None,
    *,
    episodes: Sequence[int] | None = None,
    download: bool = True,
    token: str | bool | None = None,
) -> datasets.Dataset:
    """Load named scores.

    Args:
        dataset: Dataset the scores belong to.
        name: Score name.
        columns: Columns to load. Defaults to every column, including ``index``,
            ``episode_index`` and ``frame_index``.
        episodes: Global episodes to read within the dataset view. Defaults to that view.
        download: Download this named score file if absent locally. Episodes and columns are
            selected when reading the local file. Set to ``False`` for local-only reads.
        token: Optional authentication token for the download.

    Returns:
        The scores as a Hugging Face dataset backed by a memory-mapped Arrow cache.
        A selection with no matching rows returns an empty dataset.
    """
    path = _get_scores_path(dataset, name, download=download, token=token)
    schema = _read_schema(path)
    identity = json.loads(schema.metadata[_DATASET_KEY])
    if (
        identity["repo_id"] != dataset.repo_id
        or identity["total_episodes"] > dataset.meta.total_episodes
        or identity["total_frames"] > dataset.meta.total_frames
    ):
        raise ValueError("Scores no longer align with this dataset; score it again")
    selected = sorted(_resolve_episode_indices(dataset, episodes))
    requested = schema.names if columns is None else list(columns)
    filters = pa_ds.field("episode_index").isin(selected)
    # Return a typed empty dataset when no score rows match the selection.
    if pa_ds.dataset(path, format="parquet").count_rows(filter=filters) == 0:
        scores = datasets.Dataset(schema.empty_table().select(requested))
    else:
        scores = datasets.Dataset.from_parquet(
            str(path),
            columns=requested,
            filters=filters,
        )
    return scores.select_columns(requested)


def load_scores_info(
    dataset: LeRobotDataset,
    name: str,
    *,
    download: bool = True,
    token: str | bool | None = None,
) -> dict[str, Any]:
    """Read the info stored with named scores, such as the scorer's provenance.

    Args:
        dataset: Dataset the scores belong to.
        name: Score name.
        download: Download the named Parquet file if absent locally. Set to ``False`` for
            local-only reads.
        token: Optional authentication token for the download.

    Returns:
        The ``info`` passed when the scores were written.
    """
    path = _get_scores_path(dataset, name, download=download, token=token)
    return json.loads(_read_schema(path).metadata[_INFO_KEY])


def list_scores(dataset: LeRobotDataset) -> list[str]:
    """List the names of the scores stored locally, without downloading files.

    Args:
        dataset: Dataset the scores belong to.

    Returns:
        The sorted score names.
    """
    directory = Path(dataset.root) / SCORES_DIR
    return sorted(path.stem for path in directory.glob("*.parquet") if path.is_file())


def merge_scores(
    dataset: LeRobotDataset, names: Sequence[str], *, name: str, overwrite: bool = False
) -> None:
    """Merge scores of disjoint episodes under one name.

    Useful for parallel scoring and for adding scores for appended episodes without recomputing
    existing scores.
    Source score files are not deleted.

    Args:
        dataset: Dataset the scores belong to.
        names: Score names to merge. They must have the same columns, dataset repository and ``info``,
            and share no episode.
        name: Name of the merged scores.
        overwrite: If ``True``, replace existing scores with that name.

    Raises:
        FileExistsError: If the scores exist and ``overwrite`` is ``False``.
        ValueError: If the scores differ in columns or ``info``, do not fit the dataset, or share
            an episode.
    """
    if not names:
        raise ValueError("At least one score name must be given to merge")
    output_path = _scores_path(dataset, name)
    _validate_writable_storage(dataset)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Published scores already exist at {output_path}; choose another name or set overwrite=True"
        )

    paths = [_scores_path(dataset, part) for part in names]
    schemas = [_read_schema(path) for path in paths]
    reference = schemas[0]
    reference_metadata = dict(reference.metadata)
    reference_metadata.pop(_DATASET_KEY)
    # Score files hold one row group per episode, so each episode is copied without rescanning files.
    episode_row_groups: dict[int, tuple[pq.ParquetFile, int]] = {}
    for part, path, schema in zip(names, paths, schemas, strict=True):
        metadata = dict(schema.metadata)
        identity = json.loads(metadata.pop(_DATASET_KEY))
        if (
            identity["repo_id"] != dataset.repo_id
            or identity["total_episodes"] > dataset.meta.total_episodes
            or identity["total_frames"] > dataset.meta.total_frames
        ):
            raise ValueError(f"Scores {part!r} no longer align with this dataset; score it again")
        if not schema.equals(reference, check_metadata=False) or metadata != reference_metadata:
            raise ValueError(f"Scores {part!r} have different columns or info than {names[0]!r}")
        parquet_file = pq.ParquetFile(path)
        row_groups = {}
        for row_group in range(parquet_file.num_row_groups):
            episodes = np.unique(
                parquet_file.read_row_group(row_group, columns=["episode_index"]).column(0).to_numpy()
            )
            if episodes.size != 1 or int(episodes[0]) in row_groups:
                raise ValueError(f"Scores {part!r} do not hold one row group per episode; score it again")
            row_groups[int(episodes[0])] = (parquet_file, row_group)
        shared = sorted(set(row_groups) & set(episode_row_groups))
        if shared:
            raise ValueError(f"Scores {part!r} repeat episodes of earlier scores: {shared}")
        episode_row_groups.update(row_groups)

    # The merged file describes the current dataset, including any appended episodes.
    identity = json.loads(reference.metadata[_DATASET_KEY])
    identity.update(
        revision=dataset.revision,
        total_episodes=int(dataset.meta.total_episodes),
        total_frames=int(dataset.meta.total_frames),
    )
    reference_metadata[_DATASET_KEY] = _canonical_json(identity, label="dataset identity").encode()
    reference = reference.with_metadata(reference_metadata)

    tables = (
        parquet_file.read_row_group(row_group).replace_schema_metadata(reference.metadata)
        for parquet_file, row_group in (episode_row_groups[index] for index in sorted(episode_row_groups))
    )
    _write_episode_tables(output_path, reference, tables)
    logger.info("Published merged scores %r to %s", name, output_path)


def push_scores_to_hub(dataset: LeRobotDataset, name: str) -> None:
    """Upload named scores to the dataset repository and move its version tag to that commit.

    Args:
        dataset: Dataset the scores belong to.
        name: Score name.

    Raises:
        FileNotFoundError: If the scores do not exist locally.
    """
    local_path = _scores_path(dataset, name)
    if not local_path.is_file():
        raise FileNotFoundError(f"Scores {name!r} do not exist")

    path_in_repo = f"{SCORES_DIR}/{local_path.name}"
    hub_api = HfApi()
    commit = hub_api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=dataset.repo_id,
        repo_type="dataset",
        commit_message=f"Upload scores {name}",
    )
    with suppress(RevisionNotFoundError):
        hub_api.delete_tag(dataset.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    hub_api.create_tag(dataset.repo_id, tag=CODEBASE_VERSION, revision=commit.oid, repo_type="dataset")
    logger.info("Uploaded scores %s to hf://datasets/%s/%s", name, dataset.repo_id, path_in_repo)
