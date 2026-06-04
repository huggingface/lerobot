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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from lerobot.datasets.io_utils import load_tasks
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

    # Memoized parquet slice — populated on first ``frames_df()`` call so
    # repeat queries from different modules don't re-read the whole shard.
    _frames_df_cache: Any = field(default=None, init=False, repr=False, compare=False)

    def frames_df(self):  # type: ignore[no-untyped-def]
        """Lazy-load the pandas slice for this episode (memoized)."""
        if self._frames_df_cache is None:
            import pandas as pd  # noqa: PLC0415  - deferred for optional dataset extra

            table = pq.read_table(self.data_path)
            df: pd.DataFrame = table.to_pandas()
            self._frames_df_cache = df.iloc[self.row_offset : self.row_offset + self.row_count].reset_index(
                drop=True
            )
        return self._frames_df_cache


def reconstruct_subtask_spans(
    rows: Sequence[dict[str, Any]],
    *,
    episode_end_t: float | None = None,
) -> list[dict[str, Any]]:
    """Turn ``style="subtask"`` rows into ``{text, start, end}`` spans.

    Each span's ``end`` is the next span's ``start``. The final span's
    ``end`` defaults to its own ``start`` (zero-duration) — pass
    ``episode_end_t`` to extend it to the episode's last frame instead,
    which is what downstream consumers (memory, interjection boundary
    selection) expect.

    Used by the ``plan`` module (plan-update pass) and the
    ``interjections`` module (interjection anchoring), which both need the
    same span shape.
    """
    sorted_rows = sorted(
        (r for r in rows if r.get("style") == "subtask"),
        key=lambda r: float(r["timestamp"]),
    )
    spans: list[dict[str, Any]] = []
    for r in sorted_rows:
        t = float(r["timestamp"])
        if spans:
            spans[-1]["end"] = t
        spans.append({"text": r.get("content") or "", "start": t, "end": t})
    if spans and episode_end_t is not None and float(episode_end_t) > spans[-1]["start"]:
        spans[-1]["end"] = float(episode_end_t)
    return spans


def snap_to_frame(t: float, frame_timestamps: Sequence[float]) -> float:
    """Snap an arbitrary float to the nearest exact source frame timestamp.

    Modules use this when emitting event-style rows so the row's
    timestamp matches a real parquet frame: event rows must land on an
    exact frame, otherwise the per-frame event lookup the writer does
    would never match them.
    """
    if not frame_timestamps:
        return float(t)
    nearest = min(frame_timestamps, key=lambda f: abs(f - t))
    return float(nearest)


def _load_tasks_lookup(root: Path) -> dict[int, str]:
    """Map ``task_index -> task`` from ``meta/tasks.parquet``.

    Returns an empty dict when the file is absent — the task description is
    derived later from the video if needed. Reuses the library-level
    :func:`lerobot.datasets.io_utils.load_tasks`, which returns the tasks
    frame indexed by task string with a ``task_index`` column.
    """
    if not (root / DEFAULT_TASKS_PATH).exists():
        return {}
    tasks = load_tasks(root)
    return {int(idx): str(task) for task, idx in zip(tasks.index, tasks["task_index"], strict=True)}


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
