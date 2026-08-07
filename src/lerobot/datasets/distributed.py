#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team.
# All rights reserved.
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

"""Planning primitives for distributed LeRobot dataset writes."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lerobot.utils.utils import flatten_dict

from .compute_stats import aggregate_stats
from .io_utils import cast_stats_to_numpy, write_info, write_stats, write_tasks
from .utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_DEPTH_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_IMAGE_PATH,
    DEFAULT_VIDEO_PATH,
    serialize_dict,
    update_chunk_file_indices,
)

_SESSION_DIRECTORY = ".distributed-write-session"
_PLAN_FILENAME = "plan.json"
_RESULTS_DIRECTORY = "results"


@dataclass(frozen=True)
class DistributedEpisodeSpec:
    """The immutable output allocation for one distributed episode."""

    episode_index: int
    dataset_from_index: int
    dataset_to_index: int
    tasks: tuple[str, ...]
    task_indices: tuple[int, ...]
    data_chunk_index: int
    data_file_index: int

    @property
    def length(self) -> int:
        """Number of frames assigned to this episode."""
        return self.dataset_to_index - self.dataset_from_index


@dataclass(frozen=True)
class DistributedEpisodeResult:
    """Artifacts and statistics produced by one distributed worker."""

    episode_index: int
    dataset_from_index: int
    dataset_to_index: int
    tasks: tuple[str, ...]
    task_indices: tuple[int, ...]
    data_chunk_index: int
    data_file_index: int
    data_path: Path
    video_metadata: dict[str, Any]
    video_info: dict[str, dict[str, Any]]
    stats: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class DistributedWritePlan:
    """A deterministic, driver-created allocation for distributed episode writes."""

    episodes: tuple[DistributedEpisodeSpec, ...]
    task_to_index: dict[str, int]
    chunks_size: int
    video_keys: tuple[str, ...] = ()

    @classmethod
    def from_episode_lengths(
        cls,
        *,
        episode_lengths: list[int],
        episode_tasks: list[list[str]],
        chunks_size: int,
        video_keys: list[str] | None = None,
    ) -> DistributedWritePlan:
        """Allocate contiguous global frame ranges and unique data files per episode."""
        if not episode_lengths:
            raise ValueError("A distributed write plan requires at least one episode.")
        if len(episode_lengths) != len(episode_tasks):
            raise ValueError("episode_lengths and episode_tasks must have the same length.")
        if chunks_size <= 0:
            raise ValueError("chunks_size must be positive.")

        task_to_index: dict[str, int] = {}
        episodes: list[DistributedEpisodeSpec] = []
        dataset_from_index = 0
        chunk_index = 0
        file_index = 0

        for episode_index, (episode_length, tasks) in enumerate(
            zip(episode_lengths, episode_tasks, strict=True)
        ):
            if episode_length <= 0:
                raise ValueError(f"Episode {episode_index} length must be positive.")
            if len(tasks) != episode_length:
                raise ValueError(
                    f"Episode {episode_index} task sequence length {len(tasks)} does not match "
                    f"episode length {episode_length}."
                )
            if not all(isinstance(task, str) and task for task in tasks):
                raise ValueError(f"Episode {episode_index} tasks must be non-empty strings.")

            for task in tasks:
                task_to_index.setdefault(task, len(task_to_index))

            dataset_to_index = dataset_from_index + episode_length
            episodes.append(
                DistributedEpisodeSpec(
                    episode_index=episode_index,
                    dataset_from_index=dataset_from_index,
                    dataset_to_index=dataset_to_index,
                    tasks=tuple(tasks),
                    task_indices=tuple(task_to_index[task] for task in tasks),
                    data_chunk_index=chunk_index,
                    data_file_index=file_index,
                )
            )
            dataset_from_index = dataset_to_index
            chunk_index, file_index = update_chunk_file_indices(chunk_index, file_index, chunks_size)

        return cls(
            episodes=tuple(episodes),
            task_to_index=task_to_index,
            chunks_size=chunks_size,
            video_keys=tuple(video_keys or ()),
        )


class DistributedWriteSession:
    """Persistent progress ledger for an explicitly resumable distributed write."""

    def __init__(
        self,
        root: Path,
        plan: DistributedWritePlan,
        *,
        data_path_template: str,
        video_path_template: str | None,
        camera_keys: tuple[str, ...],
        depth_keys: tuple[str, ...],
    ):
        self.root = root
        self.plan = plan
        self.data_path_template = data_path_template
        self.video_path_template = video_path_template
        self.camera_keys = camera_keys
        self.depth_keys = depth_keys
        self.session_root = self.root / _SESSION_DIRECTORY
        self.results_root = self.session_root / _RESULTS_DIRECTORY

    @classmethod
    def create(
        cls,
        root: str | Path,
        plan: DistributedWritePlan,
        *,
        data_path_template: str = DEFAULT_DATA_PATH,
        video_path_template: str | None = DEFAULT_VIDEO_PATH,
        camera_keys: tuple[str, ...] = (),
        depth_keys: tuple[str, ...] = (),
    ) -> DistributedWriteSession:
        session = cls(
            Path(root),
            plan,
            data_path_template=data_path_template,
            video_path_template=video_path_template,
            camera_keys=camera_keys,
            depth_keys=depth_keys,
        )
        if session.session_root.exists():
            raise FileExistsError(
                f"Distributed write session already exists: {session.session_root}. "
                "Use DistributedWriteSession.resume() to continue it."
            )
        temporary_session_root = session.session_root.with_name(
            f".{session.session_root.name}.{uuid4().hex}.tmp"
        )
        try:
            temporary_session_root.mkdir(parents=True)
            (temporary_session_root / _RESULTS_DIRECTORY).mkdir()
            session._write_json_atomically(temporary_session_root / _PLAN_FILENAME, session._serialize())
            os.replace(temporary_session_root, session.session_root)
        finally:
            shutil.rmtree(temporary_session_root, ignore_errors=True)
        return session

    @classmethod
    def resume(cls, root: str | Path) -> DistributedWriteSession:
        root = Path(root)
        plan_path = root / _SESSION_DIRECTORY / _PLAN_FILENAME
        if not plan_path.is_file():
            raise FileNotFoundError(f"Distributed write session plan is missing: {plan_path}")
        with plan_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        return cls(
            root,
            _deserialize_plan(payload["plan"]),
            data_path_template=payload["data_path_template"],
            video_path_template=payload["video_path_template"],
            camera_keys=tuple(payload["camera_keys"]),
            depth_keys=tuple(payload["depth_keys"]),
        )

    def result_path(self, episode_index: int) -> Path:
        return self.results_root / f"episode-{episode_index:06d}.json"

    def pending_specs(self) -> list[DistributedEpisodeSpec]:
        completed = {result.episode_index for result in self.load_results()}
        return [spec for spec in self.plan.episodes if spec.episode_index not in completed]

    def record_result(self, result: DistributedEpisodeResult) -> None:
        spec = _get_plan_spec(self.plan, result.episode_index)
        _validate_persisted_result(
            self.root,
            self.data_path_template,
            self.video_path_template,
            self.plan,
            spec,
            result,
        )
        self._write_json_atomically(self.result_path(result.episode_index), _serialize_result(result))

    def load_results(self) -> list[DistributedEpisodeResult]:
        results = []
        for spec in self.plan.episodes:
            result_path = self.result_path(spec.episode_index)
            if not result_path.is_file():
                continue
            with result_path.open(encoding="utf-8") as file:
                result = _deserialize_result(json.load(file))
            _validate_persisted_result(
                self.root,
                self.data_path_template,
                self.video_path_template,
                self.plan,
                spec,
                result,
            )
            results.append(result)
        return results

    def cleanup_orphaned_artifacts(self) -> None:
        """Remove planned output paths for specs without a durable success record."""
        for spec in self.plan.episodes:
            result_path = self.result_path(spec.episode_index)
            if self._has_valid_result_record(spec, result_path):
                continue
            result_path.unlink(missing_ok=True)
            self._cleanup_spec_artifacts(spec)

    def _has_valid_result_record(self, spec: DistributedEpisodeSpec, result_path: Path) -> bool:
        if not result_path.is_file():
            return False
        try:
            with result_path.open(encoding="utf-8") as file:
                result = _deserialize_result(json.load(file))
            _validate_persisted_result(
                self.root,
                self.data_path_template,
                self.video_path_template,
                self.plan,
                spec,
                result,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return result.data_path.is_file()

    def _cleanup_spec_artifacts(self, spec: DistributedEpisodeSpec) -> None:
        data_path = self.root / self.data_path_template.format(
            chunk_index=spec.data_chunk_index,
            file_index=spec.data_file_index,
        )
        data_path.unlink(missing_ok=True)
        if self.video_path_template is not None:
            for video_key in self.plan.video_keys:
                video_path = self.root / self.video_path_template.format(
                    video_key=video_key,
                    chunk_index=spec.data_chunk_index,
                    file_index=spec.data_file_index,
                )
                video_path.unlink(missing_ok=True)
        for camera_key in self.camera_keys:
            image_path_template = DEFAULT_DEPTH_PATH if camera_key in self.depth_keys else DEFAULT_IMAGE_PATH
            image_dir = self.root / image_path_template.format(
                image_key=camera_key,
                episode_index=spec.episode_index,
                frame_index=0,
            )
            shutil.rmtree(image_dir.parent, ignore_errors=True)

    def _serialize(self) -> dict[str, Any]:
        return {
            "plan": _serialize_plan(self.plan),
            "data_path_template": self.data_path_template,
            "video_path_template": self.video_path_template,
            "camera_keys": list(self.camera_keys),
            "depth_keys": list(self.depth_keys),
        }

    @staticmethod
    def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary_path.open("x", encoding="utf-8") as file:
                json.dump(payload, file, sort_keys=True)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _get_plan_spec(plan: DistributedWritePlan, episode_index: int) -> DistributedEpisodeSpec:
    for spec in plan.episodes:
        if spec.episode_index == episode_index:
            return spec
    raise ValueError(f"Distributed result references unknown episode {episode_index}.")


def _validate_result_matches_spec(spec: DistributedEpisodeSpec, result: DistributedEpisodeResult) -> None:
    expected_fields = (
        ("dataset_from_index", spec.dataset_from_index),
        ("dataset_to_index", spec.dataset_to_index),
        ("tasks", spec.tasks),
        ("task_indices", spec.task_indices),
        ("data_chunk_index", spec.data_chunk_index),
        ("data_file_index", spec.data_file_index),
    )
    for field, expected in expected_fields:
        if getattr(result, field) != expected:
            raise ValueError(
                f"Distributed result for episode {spec.episode_index} has unexpected {field}: "
                f"{getattr(result, field)!r} != {expected!r}."
            )


def _serialize_plan(plan: DistributedWritePlan) -> dict[str, Any]:
    return {
        "episodes": [asdict(spec) for spec in plan.episodes],
        "task_to_index": plan.task_to_index,
        "chunks_size": plan.chunks_size,
        "video_keys": list(plan.video_keys),
    }


def _deserialize_plan(payload: dict[str, Any]) -> DistributedWritePlan:
    return DistributedWritePlan(
        episodes=tuple(
            DistributedEpisodeSpec(
                **{
                    **episode,
                    "tasks": tuple(episode["tasks"]),
                    "task_indices": tuple(episode["task_indices"]),
                }
            )
            for episode in payload["episodes"]
        ),
        task_to_index=payload["task_to_index"],
        chunks_size=payload["chunks_size"],
        video_keys=tuple(payload["video_keys"]),
    )


def _serialize_result(result: DistributedEpisodeResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["tasks"] = list(result.tasks)
    payload["task_indices"] = list(result.task_indices)
    payload["data_path"] = str(result.data_path)
    payload["stats"] = serialize_dict(result.stats)
    return payload


def _deserialize_result(payload: dict[str, Any]) -> DistributedEpisodeResult:
    return DistributedEpisodeResult(
        **{
            **payload,
            "tasks": tuple(payload["tasks"]),
            "task_indices": tuple(payload["task_indices"]),
            "data_path": Path(payload["data_path"]),
            "stats": cast_stats_to_numpy(payload["stats"]),
        }
    )


def validate_distributed_results(
    root: Path,
    data_path_template: str,
    video_path_template: str | None,
    plan: DistributedWritePlan,
    results: list[DistributedEpisodeResult],
) -> list[DistributedEpisodeResult]:
    """Validate every worker result before any global metadata is written."""
    if len(results) != len(plan.episodes):
        raise ValueError(
            f"Distributed plan has {len(plan.episodes)} episodes but received {len(results)} results."
        )

    result_by_episode: dict[int, DistributedEpisodeResult] = {}
    for result in results:
        if result.episode_index in result_by_episode:
            raise ValueError(f"Duplicate distributed result for episode {result.episode_index}.")
        result_by_episode[result.episode_index] = result

    ordered_results = []
    for spec in plan.episodes:
        result = result_by_episode.pop(spec.episode_index, None)
        if result is None:
            raise ValueError(f"Missing distributed result for episode {spec.episode_index}.")
        _validate_persisted_result(
            root,
            data_path_template,
            video_path_template,
            plan,
            spec,
            result,
        )
        ordered_results.append(result)

    if result_by_episode:
        unknown = sorted(result_by_episode)
        raise ValueError(f"Results contain unknown distributed episode indexes: {unknown}.")
    return ordered_results


def _validate_persisted_result(
    root: Path,
    data_path_template: str,
    video_path_template: str | None,
    plan: DistributedWritePlan,
    spec: DistributedEpisodeSpec,
    result: DistributedEpisodeResult,
) -> None:
    _validate_result_matches_spec(spec, result)
    if not result.stats:
        raise ValueError(f"Distributed result for episode {spec.episode_index} has no statistics.")

    expected_path = root / data_path_template.format(
        chunk_index=spec.data_chunk_index,
        file_index=spec.data_file_index,
    )
    if result.data_path != expected_path:
        raise ValueError(
            f"Distributed result for episode {spec.episode_index} wrote unexpected path "
            f"{result.data_path}; expected {expected_path}."
        )
    if not result.data_path.is_file():
        raise ValueError(f"Distributed artifact is missing: {result.data_path}")

    table = pq.read_table(result.data_path, columns=["index", "episode_index"])
    expected_indices = list(range(spec.dataset_from_index, spec.dataset_to_index))
    if table.num_rows != spec.length:
        raise ValueError(
            f"Distributed artifact {result.data_path} has {table.num_rows} rows; expected {spec.length}."
        )
    if table.column("index").to_pylist() != expected_indices:
        raise ValueError(f"Distributed artifact {result.data_path} has an unexpected frame index range.")
    if set(table.column("episode_index").to_pylist()) != {spec.episode_index}:
        raise ValueError(f"Distributed artifact {result.data_path} has an unexpected episode index.")
    for video_key in plan.video_keys:
        if video_path_template is None:
            raise ValueError("Distributed plan includes videos but the dataset has no video path template.")
        expected_video_path = root / video_path_template.format(
            video_key=video_key,
            chunk_index=spec.data_chunk_index,
            file_index=spec.data_file_index,
        )
        required_metadata = {
            f"videos/{video_key}/chunk_index": spec.data_chunk_index,
            f"videos/{video_key}/file_index": spec.data_file_index,
        }
        for key, expected in required_metadata.items():
            if result.video_metadata.get(key) != expected:
                raise ValueError(
                    f"Distributed result for episode {spec.episode_index} has unexpected {key}: "
                    f"{result.video_metadata.get(key)!r} != {expected!r}."
                )
        if not expected_video_path.is_file():
            raise ValueError(f"Distributed video artifact is missing: {expected_video_path}")


def write_distributed_metadata(
    staging_root: Path,
    info: Any,
    plan: DistributedWritePlan,
    results: list[DistributedEpisodeResult],
) -> None:
    """Write complete global metadata below an unpublished staging root."""
    tasks = pd.DataFrame(
        {"task_index": list(plan.task_to_index.values())},
        index=pd.Index(plan.task_to_index.keys(), name="task"),
    )
    rows: list[dict[str, Any]] = []
    for result in results:
        flattened_stats = {
            key: _normalize_metadata_stat_value(value)
            for key, value in flatten_dict({"stats": result.stats}).items()
        }
        row = {
            "episode_index": result.episode_index,
            "tasks": list(result.tasks),
            "length": result.dataset_to_index - result.dataset_from_index,
            "data/chunk_index": result.data_chunk_index,
            "data/file_index": result.data_file_index,
            "dataset_from_index": result.dataset_from_index,
            "dataset_to_index": result.dataset_to_index,
            **flattened_stats,
            **result.video_metadata,
        }
        rows.append(row)

    staged_info = info
    staged_info.total_episodes = len(results)
    staged_info.total_frames = sum(result.dataset_to_index - result.dataset_from_index for result in results)
    staged_info.total_tasks = len(tasks)
    staged_info.splits = {"train": f"0:{len(results)}"}
    if plan.video_keys:
        if staged_info.video_path is None:
            raise ValueError("Distributed plan includes videos but the dataset has no video path template.")
        first_result = results[0]
        for video_key in plan.video_keys:
            if video_key not in first_result.video_info:
                raise ValueError(
                    f"Distributed result for episode {first_result.episode_index} has no video info "
                    f"for {video_key!r}."
                )
            staged_info.features[video_key]["info"] = first_result.video_info[video_key]

    write_info(staged_info, staging_root)
    write_tasks(tasks, staging_root)
    write_stats(aggregate_stats([result.stats for result in results]), staging_root)
    for result, row in zip(results, rows, strict=True):
        path = staging_root / DEFAULT_EPISODES_PATH.format(
            chunk_index=result.data_chunk_index,
            file_index=result.data_file_index,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        datasets.Dataset.from_list([row]).to_parquet(path)


def _normalize_metadata_stat_value(value: Any) -> Any:
    """Keep one stable Arrow dtype per flattened per-episode statistic column."""
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float64).tolist()
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64).tolist()
    return array.tolist()


def publish_distributed_metadata(root: Path, staging_root: Path) -> None:
    """Atomically publish staged metadata while retaining rollback on failure."""
    source_meta = staging_root / "meta"
    destination_meta = root / "meta"
    backup_meta = root / f".meta-distributed-backup-{uuid4().hex}"
    backup_restored = False
    try:
        os.replace(destination_meta, backup_meta)
        os.replace(source_meta, destination_meta)
    except Exception:
        if backup_meta.exists() and not destination_meta.exists():
            os.replace(backup_meta, destination_meta)
            backup_restored = True
        raise
    finally:
        if destination_meta.exists() or backup_restored:
            shutil.rmtree(backup_meta, ignore_errors=True)
