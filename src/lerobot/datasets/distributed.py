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

import hashlib
import json
import os
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass, field as dataclass_field
from pathlib import Path
from typing import Any
from uuid import uuid4

import datasets
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from filelock import FileLock

from lerobot.configs import DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT, is_depth_map
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
from .video_utils import get_video_duration_in_s, get_video_info

_SESSION_DIRECTORY = ".distributed-write-session"
_PLAN_FILENAME = "plan.json"
_RESULTS_DIRECTORY = "results"


def artifact_digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def plan_digest(plan: DistributedWritePlan) -> str:
    return hashlib.sha256(json.dumps(_serialize_plan(plan), sort_keys=True).encode()).hexdigest()


def commit_receipt(root: Path, plan: DistributedWritePlan) -> Path:
    return root / "meta" / "distributed-commits" / f"{plan_digest(plan)}.json"


def result_digest(results: list[DistributedEpisodeResult]) -> str:
    payload = [
        _serialize_result(result) for result in sorted(results, key=lambda result: result.episode_index)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_commit_receipt(
    root: Path, plan: DistributedWritePlan, results: list[DistributedEpisodeResult]
) -> None:
    path = commit_receipt(root, plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    DistributedWriteSession._write_json_atomically(
        path,
        {
            "plan": _serialize_plan(plan),
            "digest": result_digest(results),
            "results": [_serialize_result(result) for result in results],
        },
    )


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
    depth_units: dict[str, str] = dataclass_field(default_factory=dict)
    artifact_digests: dict[str, str] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class DistributedShardSpec:
    """Contiguous whole episodes offered to one task.

    A task writes only the first episode's reserved file slot. On reaching a
    soft size target it seals a completed prefix and releases the untouched
    suffix as a new task, using that suffix's first reserved slot.
    """

    episodes: tuple[DistributedEpisodeSpec, ...]

    @property
    def shard_index(self) -> int:
        return self.episodes[0].episode_index

    @property
    def num_frames(self) -> int:
        return sum(episode.length for episode in self.episodes)


@dataclass(frozen=True)
class DistributedWritePlan:
    """A deterministic, driver-created allocation for distributed episode writes."""

    episodes: tuple[DistributedEpisodeSpec, ...]
    task_to_index: dict[str, int]
    chunks_size: int
    video_keys: tuple[str, ...] = ()
    shards: tuple[DistributedShardSpec, ...] = ()
    data_files_size_in_mb: float | None = None
    video_files_size_in_mb: float | None = None

    @classmethod
    def from_episode_lengths(
        cls,
        *,
        episode_lengths: list[int],
        episode_tasks: list[list[str]],
        chunks_size: int,
        video_keys: list[str] | None = None,
        start_episode_index: int = 0,
        start_dataset_index: int = 0,
        start_chunk_index: int = 0,
        start_file_index: int = 0,
        task_to_index: dict[str, int] | None = None,
        pack_episodes: bool = True,
        max_frames_per_worker: int = 100_000,
        data_files_size_in_mb: float | None = None,
        video_files_size_in_mb: float | None = None,
    ) -> DistributedWritePlan:
        """Allocate contiguous global ranges and unique files for new or appended episodes."""
        if not episode_lengths:
            raise ValueError("A distributed write plan requires at least one episode.")
        if len(episode_lengths) != len(episode_tasks):
            raise ValueError("episode_lengths and episode_tasks must have the same length.")
        if chunks_size <= 0:
            raise ValueError("chunks_size must be positive.")
        if max_frames_per_worker <= 0:
            raise ValueError("max_frames_per_worker must be positive.")
        if start_episode_index < 0:
            raise ValueError("start_episode_index must be non-negative.")
        if start_dataset_index < 0:
            raise ValueError("start_dataset_index must be non-negative.")
        if start_chunk_index < 0:
            raise ValueError("start_chunk_index must be non-negative.")
        if not 0 <= start_file_index < chunks_size:
            raise ValueError("start_file_index must be in the current chunk range.")

        task_mapping = dict(task_to_index or {})
        if (
            any(not isinstance(task, str) or not task for task in task_mapping)
            or any(not isinstance(index, int) or index < 0 for index in task_mapping.values())
            or sorted(task_mapping.values()) != list(range(len(task_mapping)))
        ):
            raise ValueError("task_to_index must contain non-empty tasks with contiguous unique indices.")

        episodes: list[DistributedEpisodeSpec] = []
        dataset_from_index = start_dataset_index
        chunk_index = start_chunk_index
        file_index = start_file_index

        for episode_offset, (episode_length, tasks) in enumerate(
            zip(episode_lengths, episode_tasks, strict=True)
        ):
            episode_index = start_episode_index + episode_offset
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
                task_mapping.setdefault(task, len(task_mapping))

            dataset_to_index = dataset_from_index + episode_length
            episodes.append(
                DistributedEpisodeSpec(
                    episode_index=episode_index,
                    dataset_from_index=dataset_from_index,
                    dataset_to_index=dataset_to_index,
                    tasks=tuple(tasks),
                    task_indices=tuple(task_mapping[task] for task in tasks),
                    data_chunk_index=chunk_index,
                    data_file_index=file_index,
                )
            )
            dataset_from_index = dataset_to_index
            chunk_index, file_index = update_chunk_file_indices(chunk_index, file_index, chunks_size)

        shards: list[DistributedShardSpec] = []
        if pack_episodes:
            group: list[DistributedEpisodeSpec] = []
            frames = 0
            for episode in episodes:
                if group and frames + episode.length > max_frames_per_worker:
                    shards.append(DistributedShardSpec(tuple(group)))
                    group, frames = [], 0
                group.append(episode)
                frames += episode.length
            if group:
                shards.append(DistributedShardSpec(tuple(group)))
        return cls(
            episodes=tuple(episodes),
            task_to_index=task_mapping,
            chunks_size=chunks_size,
            video_keys=tuple(video_keys or ()),
            shards=tuple(shards),
            data_files_size_in_mb=data_files_size_in_mb,
            video_files_size_in_mb=video_files_size_in_mb,
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
        self.tasks_root = self.session_root / "tasks"

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
            (temporary_session_root / "tasks").mkdir()
            session._write_json_atomically(temporary_session_root / _PLAN_FILENAME, session._serialize())
            os.replace(temporary_session_root, session.session_root)
        finally:
            shutil.rmtree(temporary_session_root, ignore_errors=True)
        return session

    @classmethod
    def resume(cls, root: str | Path) -> DistributedWriteSession:
        root = Path(root)
        recover_distributed_metadata(root)
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

    def episode_lock(self, spec: DistributedEpisodeSpec) -> FileLock:
        if self.plan.shards:
            raise ValueError("Packed plans must use write_distributed_shard(), not episode-level writes.")
        # Keep lock inodes outside the disposable session directory. The filesystem
        # must implement cross-node advisory locks; process death releases ownership.
        lock_root = self.root / ".distributed-write-locks"
        lock_root.mkdir(exist_ok=True)
        return FileLock(lock_root / f"episode-{spec.episode_index:06d}.lock", timeout=0)

    def shard_lock(self, shard: DistributedShardSpec) -> FileLock:
        self.resolve_shard(shard)
        lock_root = self.root / ".distributed-write-locks"
        lock_root.mkdir(exist_ok=True)
        return FileLock(lock_root / f"shard-{shard.shard_index:06d}.lock", timeout=0)

    def shard_boundary_path(self, shard: DistributedShardSpec) -> Path:
        return self.tasks_root / f"shard-{shard.shard_index:06d}.json"

    def task_specs(self) -> list[DistributedShardSpec]:
        """Return durable task partitions plus the unstarted suffix of each plan group."""
        receipt = commit_receipt(self.root, self.plan)
        committed = None
        if receipt.is_file():
            with receipt.open(encoding="utf-8") as file:
                committed = {r["episode_index"]: r for r in json.load(file)["results"]}
        else:
            self.require_active()
        tasks = []
        for group in self.plan.shards:
            offset = 0
            while offset < len(group.episodes):
                remaining = group.episodes[offset:]
                candidate = DistributedShardSpec(remaining)
                count = len(remaining)
                if committed is not None:
                    first = committed[remaining[0].episode_index]
                    count = 1
                    while count < len(remaining):
                        following = committed[remaining[count].episode_index]
                        if following["data_path"] != first["data_path"]:
                            break
                        count += 1
                else:
                    boundary = self.shard_boundary_path(candidate)
                    if not boundary.is_file():
                        dependent_state = self.shard_result_path(candidate).is_file() or any(
                            self.shard_boundary_path(DistributedShardSpec((spec,))).is_file()
                            for spec in remaining[1:]
                        )
                        # Recheck: a live worker may have sealed this boundary
                        # while discovery was scanning its continuation records.
                        if dependent_state and not boundary.is_file():
                            raise ValueError(f"Missing persisted task boundary: {boundary}")
                    if boundary.is_file():
                        with boundary.open(encoding="utf-8") as file:
                            indices = json.load(file)["episode_indices"]
                        count = len(indices)
                        if not count or indices != [s.episode_index for s in remaining[:count]]:
                            raise ValueError(f"Invalid persisted task boundary: {boundary}")
                        if any(
                            self.shard_boundary_path(DistributedShardSpec((spec,))).is_file()
                            or self.shard_result_path(DistributedShardSpec((spec,))).is_file()
                            for spec in remaining[1:count]
                        ):
                            raise ValueError(f"Overlapping persisted task boundary: {boundary}")
                tasks.append(DistributedShardSpec(remaining[:count]))
                offset += count
        return tasks

    def resolve_shard(self, shard: DistributedShardSpec) -> DistributedShardSpec:
        """Resolve a queued spec against boundaries sealed since it was queued."""
        if not shard.episodes:
            raise ValueError("Task specification must not be empty.")
        task = next((s for s in self.task_specs() if s.shard_index == shard.shard_index), None)
        if task is not None:
            group = next(g for g in self.plan.shards if task.episodes[0] in g.episodes)
            offset = group.episodes.index(task.episodes[0])
            if shard.episodes in (task.episodes, group.episodes[offset:]):
                return task
        raise ValueError("Task specification is not a currently allocated task in this session.")

    def record_shard_boundary(self, shard: DistributedShardSpec, episode_count: int) -> DistributedShardSpec:
        """Seal a task's exact episode prefix before publishing any of its files.

        The caller holds the task lock. This decision survives failed writes and
        corrupt results, so retry cleanup cannot touch a subsequent task's files.
        """
        self.require_active()
        resolved = self.resolve_shard(shard)
        if not 0 < episode_count <= len(resolved.episodes):
            raise ValueError("A task boundary must contain a nonempty prefix of its assigned episodes.")
        path = self.shard_boundary_path(resolved)
        if path.is_file():
            if episode_count != len(resolved.episodes):
                raise ValueError("Cannot change a previously sealed task boundary.")
            return resolved
        completed = DistributedShardSpec(resolved.episodes[:episode_count])
        self.tasks_root.mkdir(exist_ok=True)
        self._write_json_atomically(path, {"episode_indices": [s.episode_index for s in completed.episodes]})
        return completed

    def shard_result_path(self, shard: DistributedShardSpec) -> Path:
        return self.results_root / f"shard-{shard.shard_index:06d}.json"

    def shard_staging_root(self, shard: DistributedShardSpec) -> Path:
        return self.root / ".distributed-shards" / f"shard-{shard.shard_index:06d}"

    def reusable_shard_results(self, shard: DistributedShardSpec) -> list[DistributedEpisodeResult] | None:
        """Read a complete, verified shard result; caller must hold the shard lock."""
        shard = self.resolve_shard(shard)
        receipt = commit_receipt(self.root, self.plan)
        if receipt.is_file():
            with receipt.open(encoding="utf-8") as file:
                payload = json.load(file)
            indices = {spec.episode_index for spec in shard.episodes}
            results = [_deserialize_result(r) for r in payload["results"] if r["episode_index"] in indices]
        else:
            self.require_active()
            path = self.shard_result_path(shard)
            if not path.is_file():
                return None
            with path.open(encoding="utf-8") as file:
                results = [_deserialize_result(r) for r in json.load(file)["results"]]
        self._validate_shard_results(shard, results)
        return results

    def _validate_shard_results(
        self, shard: DistributedShardSpec, results: list[DistributedEpisodeResult]
    ) -> None:
        shard = self.resolve_shard(shard)
        if [r.episode_index for r in results] != [s.episode_index for s in shard.episodes]:
            raise ValueError("A shard result must contain every assigned episode exactly once, in order.")
        cache: dict = {}
        _validate_task_file_slots(shard, self.plan.video_keys, results)
        for spec, result in zip(shard.episodes, results, strict=True):
            _validate_persisted_result(
                self.root, self.data_path_template, self.video_path_template, self.plan, spec, result, cache
            )
        _validate_packed_file_coverage(self.root, self.video_path_template, results, cache)

    def record_shard_results(
        self, shard: DistributedShardSpec, results: list[DistributedEpisodeResult]
    ) -> None:
        """Publish one success record only after every shared file has been closed."""
        self.require_active()
        if not self.shard_boundary_path(shard).is_file():
            raise ValueError("Seal the task boundary before recording its result.")
        self._validate_shard_results(shard, results)
        self._write_json_atomically(
            self.shard_result_path(shard), {"results": [_serialize_result(r) for r in results]}
        )

    def cleanup_shard(self, shard: DistributedShardSpec) -> None:
        """Clean only this task's single file slot; retain durable task boundaries."""
        shard = self.resolve_shard(shard)
        if commit_receipt(self.root, self.plan).is_file():
            shutil.rmtree(self.shard_staging_root(shard), ignore_errors=True)
            return
        self.require_active()
        try:
            complete = self.reusable_shard_results(shard) is not None
        except (OSError, KeyError, TypeError, ValueError):
            complete = False
        path = self.shard_result_path(shard)
        for temporary in self.results_root.glob(f".{path.name}.*.tmp"):
            temporary.unlink(missing_ok=True)
        boundary = self.shard_boundary_path(shard)
        for temporary in self.tasks_root.glob(f".{boundary.name}.*.tmp"):
            temporary.unlink(missing_ok=True)
        shutil.rmtree(self.shard_staging_root(shard), ignore_errors=True)
        if not complete:
            path.unlink(missing_ok=True)
            self._cleanup_spec_artifacts(shard.episodes[0])

    def pending_shards(self) -> list[DistributedShardSpec]:
        completed = {result.episode_index for result in self.load_results()}
        return [s for s in self.task_specs() if s.shard_index not in completed]

    def require_active(self) -> None:
        with (self.session_root / _PLAN_FILENAME).open(encoding="utf-8") as file:
            current = _deserialize_plan(json.load(file)["plan"])
        if current != self.plan:
            raise ValueError("Distributed session no longer owns the active write plan.")

    def reusable_result(self, spec: DistributedEpisodeSpec) -> DistributedEpisodeResult | None:
        if _get_plan_spec(self.plan, spec.episode_index) != spec:
            raise ValueError("Worker specification does not match its session plan.")
        receipt = commit_receipt(self.root, self.plan)
        if receipt.is_file():
            with receipt.open(encoding="utf-8") as file:
                payload = json.load(file)
            result = next(
                _deserialize_result(r) for r in payload["results"] if r["episode_index"] == spec.episode_index
            )
        else:
            self.require_active()
            path = self.result_path(spec.episode_index)
            if not self._has_valid_result_record(spec, path):
                return None
            with path.open(encoding="utf-8") as file:
                result = _deserialize_result(json.load(file))
        _validate_persisted_result(
            self.root, self.data_path_template, self.video_path_template, self.plan, spec, result
        )
        return result

    def pending_specs(self) -> list[DistributedEpisodeSpec]:
        if self.plan.shards:
            raise ValueError("Packed plans must be scheduled with pending_shards().")
        completed = {result.episode_index for result in self.load_results()}
        return [spec for spec in self.plan.episodes if spec.episode_index not in completed]

    def record_result(self, result: DistributedEpisodeResult) -> None:
        if self.plan.shards:
            raise ValueError("Packed plans require an atomic record_shard_results() for the whole shard.")
        self.require_active()
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
        if self.plan.shards:
            for shard in self.task_specs():
                completed = self.reusable_shard_results(shard)
                if completed is not None:
                    results.extend(completed)
            return results
        if commit_receipt(self.root, self.plan).is_file():
            for spec in self.plan.episodes:
                reused = self.reusable_result(spec)
                if reused is None:
                    raise ValueError(
                        f"Committed distributed result is missing for episode {spec.episode_index}."
                    )
                results.append(reused)
            return results
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
        if self.plan.shards:
            for shard in self.task_specs():
                with self.shard_lock(shard):
                    self.cleanup_shard(shard)
            return
        for spec in self.plan.episodes:
            with self.episode_lock(spec):
                self.cleanup_episode(spec)

    def cleanup_episode(self, spec: DistributedEpisodeSpec) -> None:
        """Clean one episode while its exclusive lock is held."""
        if commit_receipt(self.root, self.plan).is_file():
            return
        self.require_active()
        result_path = self.result_path(spec.episode_index)
        for path in self.results_root.glob(f".{result_path.name}.*.tmp"):
            path.unlink(missing_ok=True)
        shutil.rmtree(
            self.root / ".distributed-video-tmp" / f"episode-{spec.episode_index:06d}", ignore_errors=True
        )
        if not self._has_valid_result_record(spec, result_path):
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
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _get_plan_spec(plan: DistributedWritePlan, episode_index: int) -> DistributedEpisodeSpec:
    for spec in plan.episodes:
        if spec.episode_index == episode_index:
            return spec
    raise ValueError(f"Distributed result references unknown episode {episode_index}.")


def _validate_result_matches_spec(
    spec: DistributedEpisodeSpec, result: DistributedEpisodeResult, *, packed: bool = False
) -> None:
    expected_fields: tuple[tuple[str, Any], ...] = (
        ("episode_index", spec.episode_index),
        ("dataset_from_index", spec.dataset_from_index),
        ("dataset_to_index", spec.dataset_to_index),
        ("tasks", spec.tasks),
        ("task_indices", spec.task_indices),
    )
    if not packed:
        expected_fields += (
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
    payload = {
        "episodes": [asdict(spec) for spec in plan.episodes],
        "task_to_index": plan.task_to_index,
        "chunks_size": plan.chunks_size,
        "video_keys": list(plan.video_keys),
    }
    if plan.shards:
        payload.update(
            shards=[[spec.episode_index for spec in shard.episodes] for shard in plan.shards],
            data_files_size_in_mb=plan.data_files_size_in_mb,
            video_files_size_in_mb=plan.video_files_size_in_mb,
        )
    return payload


def _deserialize_plan(payload: dict[str, Any]) -> DistributedWritePlan:
    episodes = tuple(
        DistributedEpisodeSpec(
            **{
                **episode,
                "tasks": tuple(episode["tasks"]),
                "task_indices": tuple(episode["task_indices"]),
            }
        )
        for episode in payload["episodes"]
    )
    by_index = {spec.episode_index: spec for spec in episodes}
    return DistributedWritePlan(
        episodes=episodes,
        task_to_index=payload["task_to_index"],
        chunks_size=payload["chunks_size"],
        video_keys=tuple(payload["video_keys"]),
        shards=tuple(
            DistributedShardSpec(tuple(by_index[index] for index in group))
            for group in payload.get("shards", [])
        ),
        data_files_size_in_mb=payload.get("data_files_size_in_mb"),
        video_files_size_in_mb=payload.get("video_files_size_in_mb"),
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
            "depth_units": payload.get("depth_units", {}),
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
    cache: dict = {}
    for spec in plan.episodes:
        matched_result = result_by_episode.pop(spec.episode_index, None)
        if matched_result is None:
            raise ValueError(f"Missing distributed result for episode {spec.episode_index}.")
        _validate_persisted_result(
            root,
            data_path_template,
            video_path_template,
            plan,
            spec,
            matched_result,
            cache,
        )
        ordered_results.append(matched_result)

    if result_by_episode:
        unknown = sorted(result_by_episode)
        raise ValueError(f"Results contain unknown distributed episode indexes: {unknown}.")
    if plan.shards:
        session = DistributedWriteSession(
            root,
            plan,
            data_path_template=data_path_template,
            video_path_template=video_path_template,
            camera_keys=(),
            depth_keys=(),
        )
        by_index = {result.episode_index: result for result in ordered_results}
        committed = commit_receipt(root, plan).is_file()
        for shard in session.task_specs():
            if not committed and not session.shard_boundary_path(shard).is_file():
                raise ValueError("Every task must have a durable boundary before metadata commit.")
            _validate_task_file_slots(
                shard, plan.video_keys, [by_index[s.episode_index] for s in shard.episodes]
            )
        _validate_packed_file_coverage(root, video_path_template, ordered_results, cache)
    return ordered_results


def _validate_task_file_slots(
    shard: DistributedShardSpec, video_keys: tuple[str, ...], results: list[DistributedEpisodeResult]
) -> None:
    first = shard.episodes[0]
    slot = (first.data_chunk_index, first.data_file_index)
    for result in results:
        if (result.data_chunk_index, result.data_file_index) != slot:
            raise ValueError("Each task must write exactly one Parquet in its first reserved slot.")
        for key in video_keys:
            if (
                result.video_metadata.get(f"videos/{key}/chunk_index"),
                result.video_metadata.get(f"videos/{key}/file_index"),
            ) != slot:
                raise ValueError("Each task must write exactly one MP4 per video feature in its own slot.")


def _validate_persisted_result(
    root: Path,
    data_path_template: str,
    video_path_template: str | None,
    plan: DistributedWritePlan,
    spec: DistributedEpisodeSpec,
    result: DistributedEpisodeResult,
    cache: dict | None = None,
) -> None:
    cache = {} if cache is None else cache
    _validate_result_matches_spec(spec, result, packed=bool(plan.shards))
    slots = {(spec.data_chunk_index, spec.data_file_index)}
    if plan.shards:
        shard = next((shard for shard in plan.shards if spec in shard.episodes), None)
        if shard is None:
            raise ValueError(f"Episode {spec.episode_index} has no shard owner.")
        slots = {(s.data_chunk_index, s.data_file_index) for s in shard.episodes}
        if (result.data_chunk_index, result.data_file_index) not in slots:
            raise ValueError("Distributed result uses a data file outside its shard reservation.")
    if not result.stats:
        raise ValueError(f"Distributed result for episode {spec.episode_index} has no statistics.")

    expected_path = root / data_path_template.format(
        chunk_index=result.data_chunk_index,
        file_index=result.data_file_index,
    )
    if result.data_path != expected_path:
        raise ValueError(
            f"Distributed result for episode {spec.episode_index} wrote unexpected path "
            f"{result.data_path}; expected {expected_path}."
        )
    if not result.data_path.is_file():
        raise ValueError(f"Distributed artifact is missing: {result.data_path}")

    table_key = ("table", result.data_path)
    if table_key not in cache:
        cache[table_key] = pq.read_table(result.data_path, columns=["index", "episode_index"])
    table = cache[table_key]
    if plan.shards:
        table = table.filter(pc.equal(table.column("episode_index"), spec.episode_index))
    expected_indices = list(range(spec.dataset_from_index, spec.dataset_to_index))
    if table.num_rows != spec.length:
        raise ValueError(
            f"Distributed artifact {result.data_path} has {table.num_rows} rows; expected {spec.length}."
        )
    if table.column("index").to_pylist() != expected_indices:
        raise ValueError(f"Distributed artifact {result.data_path} has an unexpected frame index range.")
    if set(table.column("episode_index").to_pylist()) != {spec.episode_index}:
        raise ValueError(f"Distributed artifact {result.data_path} has an unexpected episode index.")
    expected_artifacts = {str(result.data_path.relative_to(root))}
    if plan.shards and set(result.video_info) != set(plan.video_keys):
        raise ValueError("Packed result video info must cover every planned video feature.")
    for video_key in plan.video_keys:
        if video_path_template is None:
            raise ValueError("Distributed plan includes videos but the dataset has no video path template.")
        video_slot = (
            result.video_metadata.get(f"videos/{video_key}/chunk_index"),
            result.video_metadata.get(f"videos/{video_key}/file_index"),
        )
        if video_slot not in slots:
            raise ValueError(
                f"Distributed result has unexpected videos/{video_key}/file_index or chunk_index."
            )
        expected_video_path = root / video_path_template.format(
            video_key=video_key, chunk_index=video_slot[0], file_index=video_slot[1]
        )
        expected_artifacts.add(str(expected_video_path.relative_to(root)))
        if not expected_video_path.is_file():
            raise ValueError(f"Distributed video artifact is missing: {expected_video_path}")
        try:
            video_cache_key = ("video", expected_video_path)
            if video_cache_key not in cache:
                cache[video_cache_key] = (
                    get_video_info(expected_video_path),
                    get_video_duration_in_s(expected_video_path),
                )
            video_info, duration = cache[video_cache_key]
        except Exception as error:
            raise ValueError(f"Invalid distributed video artifact: {expected_video_path}") from error
        expected_duration = spec.length / video_info["video.fps"]
        start = result.video_metadata.get(f"videos/{video_key}/from_timestamp", -1)
        end = result.video_metadata.get(f"videos/{video_key}/to_timestamp", -1)
        if start < 0 or end <= start or end > duration + 1 / video_info["video.fps"]:
            raise ValueError(f"Invalid distributed video interval: {expected_video_path}")
        if abs((end - start) - expected_duration) > 1 / video_info["video.fps"]:
            raise ValueError(f"Unexpected distributed video duration: {expected_video_path}")
        if not plan.shards and not np.isclose(start, 0):
            raise ValueError(f"Invalid video start timestamp: {expected_video_path}")
        if not plan.shards and not np.isclose(end, duration):
            raise ValueError(f"Invalid video end timestamp: {expected_video_path}")
    if plan.shards and set(result.artifact_digests) != expected_artifacts:
        raise ValueError("Packed result checksums must cover exactly its referenced artifacts.")
    for relative_path, expected_digest in result.artifact_digests.items():
        path = root / relative_path
        digest_key = ("digest", path)
        if path.is_file() and digest_key not in cache:
            cache[digest_key] = artifact_digest(path)
        if cache.get(digest_key) != expected_digest:
            raise ValueError(f"Distributed artifact checksum mismatch: {path}")


def _validate_packed_file_coverage(
    root: Path, video_template: str | None, results: list[DistributedEpisodeResult], cache: dict
) -> None:
    """Reject extra Parquet rows and overlapping or missing video intervals."""
    files: dict[Path, list[DistributedEpisodeResult]] = defaultdict(list)
    videos: dict[Path, list[tuple[float, float]]] = defaultdict(list)
    for result in results:
        files[result.data_path].append(result)
        if video_template is not None:
            for key in result.video_info:
                path = root / video_template.format(
                    video_key=key,
                    chunk_index=result.video_metadata[f"videos/{key}/chunk_index"],
                    file_index=result.video_metadata[f"videos/{key}/file_index"],
                )
                videos[path].append(
                    (
                        result.video_metadata[f"videos/{key}/from_timestamp"],
                        result.video_metadata[f"videos/{key}/to_timestamp"],
                    )
                )
    for path, episodes in files.items():
        expected = [i for r in episodes for i in range(r.dataset_from_index, r.dataset_to_index)]
        if expected != list(range(expected[0], expected[-1] + 1)):
            raise ValueError(f"Packed Parquet file must contain contiguous episodes: {path}")
        if cache[("table", path)].column("index").to_pylist() != expected:
            raise ValueError(f"Packed Parquet file has unexpected episode coverage: {path}")
    for path, intervals in videos.items():
        end = 0.0
        for start, next_end in sorted(intervals):
            if not np.isclose(start, end):
                raise ValueError(f"Packed video has overlapping or missing episode intervals: {path}")
            end = next_end
        info, duration = cache[("video", path)]
        if abs(end - duration) > 1 / info["video.fps"]:
            raise ValueError(f"Packed video has unexpected duration: {path}")


def write_distributed_metadata(
    staging_root: Path,
    info: Any,
    plan: DistributedWritePlan,
    results: list[DistributedEpisodeResult],
    *,
    existing_metadata_root: Path | None = None,
    existing_stats: dict[str, dict[str, np.ndarray]] | None = None,
) -> None:
    """Write complete global metadata below an unpublished staging root."""
    base_total_episodes = info.total_episodes
    base_total_frames = info.total_frames
    if base_total_episodes:
        if existing_metadata_root is None:
            raise ValueError("Appending distributed metadata requires the existing dataset metadata root.")
        shutil.copytree(existing_metadata_root / "meta", staging_root / "meta")

    tasks = pd.DataFrame(
        {"task_index": list(plan.task_to_index.values())},
        index=pd.Index(plan.task_to_index.keys(), name="task"),
    ).sort_values("task_index")
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
            "meta/episodes/chunk_index": result.data_chunk_index,
            "meta/episodes/file_index": result.data_file_index,
            "dataset_from_index": result.dataset_from_index,
            "dataset_to_index": result.dataset_to_index,
            **flattened_stats,
            **result.video_metadata,
        }
        rows.append(row)

    staged_info = info
    staged_info.total_episodes = base_total_episodes + len(results)
    staged_info.total_frames = base_total_frames + sum(
        result.dataset_to_index - result.dataset_from_index for result in results
    )
    staged_info.total_tasks = len(tasks)
    staged_info.splits = {"train": f"0:{staged_info.total_episodes}"}
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
            if base_total_episodes == 0:
                staged_info.features[video_key]["info"] = first_result.video_info[video_key]

    expected_depth_keys = {key for key, feature in staged_info.features.items() if is_depth_map(feature)}
    declared_depth_units = {
        key: (staged_info.features[key].get("info") or {}).get("depth_unit") for key in expected_depth_keys
    }
    depth_units: dict[str, str] = {}
    for result in results:
        result_depth_keys = set(result.depth_units)
        unknown_depth_keys = result_depth_keys - expected_depth_keys
        if unknown_depth_keys:
            raise ValueError(
                f"Distributed result for episode {result.episode_index} has unknown depth keys "
                f"{sorted(unknown_depth_keys)!r}."
            )
        effective_depth_units = {
            depth_key: result.depth_units.get(depth_key) or declared_depth_units[depth_key]
            for depth_key in expected_depth_keys
        }
        conflicting_declared_units = {
            depth_key
            for depth_key in expected_depth_keys
            if result.depth_units.get(depth_key) is not None
            and declared_depth_units[depth_key] is not None
            and result.depth_units[depth_key] != declared_depth_units[depth_key]
        }
        if conflicting_declared_units:
            raise ValueError(
                f"Distributed result for episode {result.episode_index} disagrees with the existing "
                f"depth unit for {sorted(conflicting_declared_units)!r}."
            )
        missing_depth_keys = {
            depth_key for depth_key, depth_unit in effective_depth_units.items() if depth_unit is None
        }
        if missing_depth_keys:
            raise ValueError(
                f"Distributed result for episode {result.episode_index} is missing depth units for "
                f"{sorted(missing_depth_keys)!r}."
            )
        for depth_key, depth_unit in effective_depth_units.items():
            if depth_unit not in (DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT):
                raise ValueError(
                    f"Distributed result for episode {result.episode_index} has unsupported depth unit "
                    f"{depth_unit!r} for {depth_key!r}."
                )
            existing_unit = depth_units.setdefault(depth_key, depth_unit)
            if existing_unit != depth_unit:
                raise ValueError(
                    f"Distributed results disagree on depth unit for {depth_key!r}: "
                    f"{existing_unit!r} != {depth_unit!r}."
                )
    for depth_key, depth_unit in depth_units.items():
        feature_info = staged_info.features[depth_key].setdefault("info", {})
        feature_info["depth_unit"] = depth_unit

    write_info(staged_info, staging_root)
    write_tasks(tasks, staging_root)
    stats_to_aggregate = [result.stats for result in results]
    if existing_stats is not None:
        stats_to_aggregate.insert(0, existing_stats)
    write_stats(aggregate_stats(stats_to_aggregate), staging_root)
    metadata_files: dict[Path, list[dict]] = defaultdict(list)
    for result, row in zip(results, rows, strict=True):
        path = staging_root / DEFAULT_EPISODES_PATH.format(
            chunk_index=result.data_chunk_index,
            file_index=result.data_file_index,
        )
        metadata_files[path].append(row)
    for path, file_rows in metadata_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        datasets.Dataset.from_list(file_rows).to_parquet(path)


def _normalize_metadata_stat_value(value: Any) -> Any:
    """Keep one stable Arrow dtype per flattened per-episode statistic column."""
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float64).tolist()
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64).tolist()
    return array.tolist()


def publish_distributed_metadata(root: Path, staging_root: Path) -> None:
    """Publish metadata with a persistent recovery journal (single driver)."""
    with FileLock(root / ".distributed-metadata.lock"):
        _publish_distributed_metadata(root, staging_root)


def _publish_distributed_metadata(root: Path, staging_root: Path) -> None:
    journal = root / ".distributed-metadata-publish.json"
    if journal.exists():
        with journal.open(encoding="utf-8") as file:
            previous_source = json.load(file)["source"]
        published = _recover_distributed_metadata(root)
        if published and previous_source == str(staging_root.relative_to(root) / "meta"):
            return
    source_meta = staging_root / "meta"
    destination_meta = root / "meta"
    if not source_meta.is_dir():
        raise FileNotFoundError(f"Staged distributed metadata is missing: {source_meta}")
    if not destination_meta.is_dir():
        raise FileNotFoundError(
            f"Cannot publish distributed metadata because {destination_meta} is missing. "
            "Initialize the dataset with LeRobotDataset.create() before committing distributed results. "
            f"Staged metadata remains at {source_meta}."
        )
    backup_meta = root / f".meta-distributed-backup-{uuid4().hex}"
    DistributedWriteSession._write_json_atomically(
        journal,
        {
            "source": str(source_meta.relative_to(root)),
            "backup": backup_meta.name,
        },
    )
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
            journal.unlink(missing_ok=True)


def recover_distributed_metadata(root: Path) -> bool:
    """Recover a metadata swap; return whether staged metadata was published."""
    with FileLock(root / ".distributed-metadata.lock"):
        return _recover_distributed_metadata(root)


def _recover_distributed_metadata(root: Path) -> bool:
    journal = root / ".distributed-metadata-publish.json"
    if not journal.is_file():
        return False
    with journal.open(encoding="utf-8") as file:
        state = json.load(file)
    source = root / state["source"]
    backup = root / state["backup"]
    destination = root / "meta"
    published = True
    if source.is_dir():
        if destination.exists():
            if backup.exists():
                raise RuntimeError("Ambiguous distributed metadata recovery state; preserving all copies.")
            os.replace(destination, backup)
        os.replace(source, destination)
    elif not destination.exists():
        if not backup.is_dir():
            raise RuntimeError("Distributed metadata recovery has neither staged nor backup metadata.")
        os.replace(backup, destination)
        published = False
    shutil.rmtree(backup, ignore_errors=True)
    journal.unlink()
    return published
