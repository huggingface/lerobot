#!/usr/bin/env python

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
"""Datatrove-shaped reader.

The reader walks ``data/chunk-*/file-*.parquet`` and yields one record per
episode containing:

- ``episode_index``: int
- ``frame_timestamps``: tuple[float, ...]
- ``frame_indices``: tuple[int, ...]
- ``episode_task``: str (canonical task from ``meta/tasks.parquet``)
- ``data_path``: pathlib.Path of the source parquet shard
- ``frames_df``: pandas.DataFrame slice for the episode (only loaded on demand)

This shape lets each module operate per-episode without loading all parquet
rows into memory at once.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lerobot.datasets.utils import DEFAULT_TASKS_PATH


@dataclass
class EpisodeRecord:
    """Per-episode record yielded by the reader."""

    episode_index: int
    episode_task: str
    frame_timestamps: tuple[float, ...]
    frame_indices: tuple[int, ...]
    data_path: Path
    row_offset: int  # row offset within the parquet file where this episode starts
    row_count: int  # number of rows for this episode

    def frames_df(self):  # type: ignore[no-untyped-def]
        """Lazy-load the pandas slice for this episode."""
        import pandas as pd  # noqa: PLC0415  - deferred for optional dataset extra

        table = pq.read_table(self.data_path)
        df: pd.DataFrame = table.to_pandas()
        slice_ = df.iloc[self.row_offset : self.row_offset + self.row_count].reset_index(drop=True)
        return slice_


def _load_tasks_lookup(root: Path) -> dict[int, str]:
    tasks_path = root / DEFAULT_TASKS_PATH
    if not tasks_path.exists():
        return {}
    table = pq.read_table(tasks_path)
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    if "task_index" in cols and "task" in cols:
        return dict(zip(cols["task_index"], cols["task"], strict=True))
    raise ValueError(f"meta/tasks.parquet at {tasks_path} missing 'task_index' or 'task'")


def iter_episodes(root: Path, *, only_episodes: tuple[int, ...] | None = None) -> Iterator[EpisodeRecord]:
    """Yield :class:`EpisodeRecord` for every episode under ``root/data/``.

    Episodes are yielded in ascending ``episode_index`` order. The reader does
    not assume a specific chunk/file layout: it scans every ``*.parquet``
    under ``data/`` and groups by ``episode_index``.
    """
    tasks = _load_tasks_lookup(root)
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))

    only_set = set(only_episodes) if only_episodes is not None else None

    for path in parquet_files:
        yield from _iter_one_path(path, tasks, only_set)


def _iter_one_path(path: Path, tasks: dict[int, str], only_set: set[int] | None) -> Iterator[EpisodeRecord]:
    table = pq.read_table(path)
    names = table.column_names
    if "episode_index" not in names:
        return
    episode_col = table.column("episode_index").to_pylist()
    timestamp_col = (
        table.column("timestamp").to_pylist() if "timestamp" in names else [0.0] * len(episode_col)
    )
    frame_col = (
        table.column("frame_index").to_pylist() if "frame_index" in names else list(range(len(episode_col)))
    )
    task_col = table.column("task_index").to_pylist() if "task_index" in names else None

    def _build(
        ep: int,
        start: int,
        end: int,
        task_idx: int | None,
        ts_buf: list[float],
        fi_buf: list[int],
    ) -> EpisodeRecord | None:
        if only_set is not None and ep not in only_set:
            return None
        task = tasks.get(task_idx, "") if task_idx is not None else ""
        return EpisodeRecord(
            episode_index=ep,
            episode_task=task,
            frame_timestamps=tuple(ts_buf),
            frame_indices=tuple(fi_buf),
            data_path=path,
            row_offset=start,
            row_count=end - start,
        )

    cur_ep: int | None = None
    start_offset = 0
    ts_buf: list[float] = []
    fi_buf: list[int] = []
    cur_task_idx: int | None = None

    for i, ep in enumerate(episode_col):
        if cur_ep is None:
            cur_ep = ep
            start_offset = i
            ts_buf = [timestamp_col[i]]
            fi_buf = [frame_col[i]]
            cur_task_idx = task_col[i] if task_col is not None else None
            continue
        if ep != cur_ep:
            rec = _build(cur_ep, start_offset, i, cur_task_idx, ts_buf, fi_buf)
            if rec is not None:
                yield rec
            cur_ep = ep
            start_offset = i
            ts_buf = [timestamp_col[i]]
            fi_buf = [frame_col[i]]
            cur_task_idx = task_col[i] if task_col is not None else None
        else:
            ts_buf.append(timestamp_col[i])
            fi_buf.append(frame_col[i])

    if cur_ep is not None:
        rec = _build(cur_ep, start_offset, len(episode_col), cur_task_idx, ts_buf, fi_buf)
        if rec is not None:
            yield rec


def gather_data_paths(root: Path) -> list[Path]:
    """Return every ``data/chunk-*/file-*.parquet`` path under ``root``."""
    return sorted((root / "data").rglob("*.parquet"))


def episode_offsets_per_path(path: Path) -> dict[int, tuple[int, int]]:
    """Return ``{episode_index: (row_offset, row_count)}`` for one parquet."""
    table = pq.read_table(path, columns=["episode_index"])
    episode_col = table.column("episode_index").to_pylist()
    out: dict[int, tuple[int, int]] = {}
    cur_ep: int | None = None
    start = 0
    for i, ep in enumerate(episode_col):
        if cur_ep is None:
            cur_ep = ep
            start = i
            continue
        if ep != cur_ep:
            out[cur_ep] = (start, i - start)
            cur_ep = ep
            start = i
    if cur_ep is not None:
        out[cur_ep] = (start, len(episode_col) - start)
    return out


def keyframe_indices(record: EpisodeRecord, k: int) -> list[int]:
    """Return ``k`` evenly spaced row indices into the episode (relative)."""
    n = record.row_count
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(range(n))
    step = (n - 1) / (k - 1) if k > 1 else 0.0
    return [int(round(i * step)) for i in range(k)] if k > 1 else [n // 2]


def lookup_data_path(root: Path, episode_index: int) -> tuple[Path, int, int] | None:
    """Find the parquet file containing ``episode_index`` and its slice bounds."""
    for path in gather_data_paths(root):
        offsets = episode_offsets_per_path(path)
        if episode_index in offsets:
            start, count = offsets[episode_index]
            return path, start, count
    return None


def episode_frame_timestamps(root: Path, episode_index: int) -> tuple[Any, list[float]]:
    """Return the parquet path and per-frame timestamps for ``episode_index``."""
    found = lookup_data_path(root, episode_index)
    if found is None:
        raise ValueError(f"Episode {episode_index} not found under {root}/data/")
    path, start, count = found
    table = pq.read_table(path, columns=["timestamp"])
    timestamps = table.column("timestamp").to_pylist()[start : start + count]
    return path, [float(t) for t in timestamps]
