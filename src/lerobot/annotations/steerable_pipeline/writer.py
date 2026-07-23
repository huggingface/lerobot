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
"""Final parquet rewrite.

For every episode the writer:

1. reads the staged module outputs,
2. partitions them into a persistent slice (PERSISTENT_STYLES) and an event
   slice (EVENT_ONLY_STYLES + style=None tool-call atoms),
3. sorts each slice deterministically,
4. broadcasts the persistent slice across every frame in the episode,
5. for each frame, materializes the sublist of event rows whose timestamp
   exactly equals that frame's timestamp,
6. drops the legacy ``subtask_index`` column,
7. writes the parquet shard back in place.

The writer does NOT add a dataset-level ``tools`` column. Tool *calls* are
emitted per-row via the existing ``tool_calls`` field on the v3.1 row
struct for every speech atom. The tool *schema* (the description
of the ``say`` function and its parameters) is a fixed code constant —
``SAY_TOOL_SCHEMA`` below — and downstream chat-template consumers import
it directly rather than reading a redundant per-row column.

Invariants enforced here (and re-checked by the validator):

- per-episode persistent slice is byte-identical across every frame;
- ``language_events`` rows on a frame all have ``timestamp == frame_ts``
  (timestamps come straight from the source parquet — never recomputed);
- every row passes ``column_for_style(style)``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
from lerobot.datasets.language import (
    EVENT_ONLY_STYLES,
    LANGUAGE_EVENTS,
    LANGUAGE_PERSISTENT,
    PERSISTENT_STYLES,
    column_for_style,
    validate_camera_field,
)

from .reader import EpisodeRecord
from .staging import EpisodeStaging

logger = logging.getLogger(__name__)


# Tool schema constants live in lerobot.datasets.language — single
# source of truth. Re-exported here so existing imports
# (``from lerobot.annotations.steerable_pipeline.writer import SAY_TOOL_SCHEMA``)
# keep working.
from lerobot.datasets.language import DEFAULT_TOOLS, SAY_TOOL_SCHEMA  # noqa: F401, E402


def _row_persistent_sort_key(row: dict[str, Any]) -> tuple:
    return (float(row["timestamp"]), row.get("style") or "", row.get("role") or "")


def _row_event_sort_key(row: dict[str, Any]) -> tuple:
    # events are bucketed per-frame, but within a frame we still want determinism
    return (
        row.get("style") or "",
        row.get("role") or "",
        row.get("camera") or "",
    )


def _normalize_row(row: dict[str, Any], style: str | None, *, with_timestamp: bool) -> dict[str, Any]:
    """Coerce a staged row into the language-column struct shape.

    Key order matches ``PERSISTENT_ROW_FIELDS`` / ``EVENT_ROW_FIELDS`` — the
    writer infers the parquet struct schema from insertion order, so
    ``timestamp`` (persistent rows only) sits between ``style`` and ``camera``.
    """
    camera = row.get("camera")
    validate_camera_field(style, camera)
    out: dict[str, Any] = {
        "role": str(row["role"]),
        "content": None if row.get("content") is None else str(row["content"]),
        "style": style,
    }
    if with_timestamp:
        out["timestamp"] = float(row["timestamp"])
    out["camera"] = None if camera is None else str(camera)
    out["tool_calls"] = _normalize_tool_calls(row.get("tool_calls"))
    return out


def _normalize_persistent_row(row: dict[str, Any]) -> dict[str, Any]:
    """Coerce a staged row into the persistent column's struct shape."""
    style = row.get("style")
    if style not in PERSISTENT_STYLES:
        raise ValueError(
            f"persistent slice contains row with non-persistent style {style!r}; "
            "row would be misrouted under column_for_style()"
        )
    if "timestamp" not in row:
        raise ValueError(f"persistent row missing timestamp: {row!r}")
    if "role" not in row:
        # Friendly error from the writer instead of a raw KeyError below;
        # the validator doesn't check ``role`` yet.
        raise ValueError(f"persistent row missing role: {row!r}")
    return _normalize_row(row, style, with_timestamp=True)


def _normalize_event_row(row: dict[str, Any]) -> dict[str, Any]:
    """Coerce a staged row into the event column's struct shape (no timestamp)."""
    style = row.get("style")
    if style is not None and style not in EVENT_ONLY_STYLES:
        raise ValueError(
            f"event slice contains row with style {style!r}; expected None or one of {EVENT_ONLY_STYLES}"
        )
    if column_for_style(style) != LANGUAGE_EVENTS:
        raise ValueError(f"event row with style {style!r} would not route to language_events")
    if "role" not in row:
        raise ValueError(f"event row missing role: {row!r}")
    return _normalize_row(row, style, with_timestamp=False)


def _normalize_tool_calls(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"tool_calls must be a list or None, got {type(value).__name__}")
    return list(value)


def _validate_atom_invariants(row: dict[str, Any]) -> None:
    """At-least-one of content/tool_calls; style=None implies tool_calls."""
    has_content = row.get("content") is not None
    has_tools = row.get("tool_calls") is not None
    if not (has_content or has_tools):
        raise ValueError(f"row has neither content nor tool_calls: {row!r}")
    if row.get("style") is None and not has_tools:
        raise ValueError(f"style=None requires tool_calls: {row!r}")


def _validate_speech_atom(row: dict[str, Any]) -> None:
    """Speech atoms: role=assistant, style=None, content=None, say tool call."""
    if row.get("style") is not None:
        return  # not a speech atom
    if row.get("role") != "assistant":
        raise ValueError(f"speech atom must have role=assistant: {row!r}")
    if row.get("content") is not None:
        raise ValueError(f"speech atom must have content=null: {row!r}")
    tool_calls = row.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        raise ValueError(f"speech atom must have non-empty tool_calls list: {row!r}")
    first = tool_calls[0]
    if not isinstance(first, dict):
        raise ValueError(f"speech atom tool_calls[0] must be a dict: {row!r}")
    if first.get("type") != "function":
        raise ValueError(f"speech atom tool_calls[0].type must be 'function': {row!r}")
    fn = first.get("function") or {}
    if fn.get("name") != "say":
        raise ValueError(f"speech atom tool_calls[0].function.name must be 'say': {row!r}")
    args = fn.get("arguments") or {}
    if not isinstance(args, dict) or "text" not in args or not isinstance(args["text"], str):
        raise ValueError(f"speech atom must carry 'text' string in arguments: {row!r}")


@dataclass
class LanguageColumnsWriter:
    """Rewrite ``data/chunk-*/file-*.parquet`` with the two language columns."""

    drop_existing_subtask_index: bool = True

    def write_all(
        self,
        records: Sequence[EpisodeRecord],
        staging_dir: Path,
        root: Path,
    ) -> list[Path]:
        episodes_by_path: dict[Path, list[EpisodeRecord]] = defaultdict(list)
        for record in records:
            episodes_by_path[record.data_path].append(record)

        written: list[Path] = []
        for path, eps in episodes_by_path.items():
            self._rewrite_one(path, eps, staging_dir, root)
            written.append(path)
        return written

    def _rewrite_one(
        self,
        path: Path,
        episodes: Sequence[EpisodeRecord],
        staging_dir: Path,
        root: Path,
    ) -> None:
        table = pq.read_table(path)
        n_rows = table.num_rows

        # Ensure we cover every episode in the file. Episodes that don't have
        # staging artifacts are passed through with empty annotation lists —
        # this keeps the writer idempotent and safe for partial reruns.
        staged_per_ep: dict[int, dict[str, list[dict[str, Any]]]] = {}
        for record in episodes:
            staging = EpisodeStaging(staging_dir, record.episode_index)
            staged_per_ep[record.episode_index] = staging.read_all()

        persistent_by_ep: dict[int, list[dict[str, Any]]] = {}
        events_by_ep_ts: dict[int, dict[float, list[dict[str, Any]]]] = {}

        for ep_index, ep_staged in staged_per_ep.items():
            persistent_rows: list[dict[str, Any]] = []
            event_rows: list[dict[str, Any]] = []  # carry timestamp until bucketed
            for _module_name, rows in ep_staged.items():
                for row in rows:
                    style = row.get("style")
                    if column_for_style(style) == LANGUAGE_PERSISTENT:
                        persistent_rows.append(row)
                    else:
                        event_rows.append(row)

            persistent_rows.sort(key=_row_persistent_sort_key)
            normalized_persistent = []
            for r in persistent_rows:
                _validate_atom_invariants(r)
                _validate_speech_atom(r)
                normalized_persistent.append(_normalize_persistent_row(r))
            persistent_by_ep[ep_index] = normalized_persistent

            buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for r in event_rows:
                _validate_atom_invariants(r)
                _validate_speech_atom(r)
                ts = float(r["timestamp"])
                buckets[ts].append(_normalize_event_row(r))
            for ts in list(buckets.keys()):
                buckets[ts].sort(key=_row_event_sort_key)
            events_by_ep_ts[ep_index] = buckets

        episode_col = (
            table.column("episode_index").to_pylist() if "episode_index" in table.column_names else None
        )
        ts_col = table.column("timestamp").to_pylist() if "timestamp" in table.column_names else None
        if episode_col is None or ts_col is None:
            raise ValueError(f"{path} is missing 'episode_index' or 'timestamp' — required by the writer.")

        per_row_persistent: list[list[dict[str, Any]]] = []
        per_row_events: list[list[dict[str, Any]]] = []
        for i in range(n_rows):
            ep = episode_col[i]
            ts = float(ts_col[i])
            per_row_persistent.append(persistent_by_ep.get(ep, []))
            buckets = events_by_ep_ts.get(ep, {})
            per_row_events.append(buckets.get(ts, []))

        new_table = self._materialize_table(
            table, per_row_persistent, per_row_events, drop_old=self.drop_existing_subtask_index
        )
        # Re-emit one row group per episode (a bulk pq.write_table would collapse
        # them into one). Write to a sibling tmp path and atomically rename so a
        # crash mid-write can't leave a half-written shard.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        write_table_one_row_group_per_episode(new_table, tmp_path)
        tmp_path.replace(path)

    def _materialize_table(
        self,
        table: pa.Table,
        persistent: list[list[dict[str, Any]]],
        events: list[list[dict[str, Any]]],
        *,
        drop_old: bool,
    ) -> pa.Table:
        cols = []
        names = []
        for name in table.column_names:
            if drop_old and name == "subtask_index":
                continue
            if name in (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS):
                continue  # we'll re-add canonical versions
            # Strip any legacy ``tools`` column previously emitted by older
            # writers — the schema no longer uses it (constant lives in
            # SAY_TOOL_SCHEMA / DEFAULT_TOOLS).
            if name == "tools":
                continue
            cols.append(table.column(name))
            names.append(name)

        # We let pyarrow infer struct/list schema rather than passing the
        # canonical type from `lerobot.datasets.language` directly: that type
        # uses `pa.json_()` for the `tool_calls` element type, which
        # `pa.array(..., type=...)` cannot materialize from Python lists on
        # current pyarrow versions. The inferred schema round-trips through
        # parquet and `LeRobotDataset` correctly — `tests/datasets/test_language.py`
        # exercises the same flow.
        persistent_arr = pa.array(persistent)
        events_arr = pa.array(events)

        cols.extend([persistent_arr, events_arr])
        names.extend([LANGUAGE_PERSISTENT, LANGUAGE_EVENTS])

        return pa.Table.from_arrays(cols, names=names)


def speech_atom(timestamp: float, text: str) -> dict[str, Any]:
    """Build a canonical speech tool-call atom for the events column."""
    return {
        "role": "assistant",
        "content": None,
        "style": None,
        "timestamp": float(timestamp),
        "camera": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "say",
                    "arguments": {"text": text},
                },
            }
        ],
    }
