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

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lerobot.utils.utils import flatten_dict

from .compute_stats import aggregate_stats
from .io_utils import write_info, write_stats, write_tasks
from .utils import DEFAULT_EPISODES_PATH, update_chunk_file_indices


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
                raise ValueError(
                    "Distributed plan includes videos but the dataset has no video path template."
                )
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
        ordered_results.append(result)

    if result_by_episode:
        unknown = sorted(result_by_episode)
        raise ValueError(f"Results contain unknown distributed episode indexes: {unknown}.")
    return ordered_results


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
