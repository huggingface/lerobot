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
"""Read-only loader for LeRobot datasets stored as LanceDB tables.

Layout of a Lance-backed dataset (video layout):

```
<root>/
  meta/         # standard LeRobot v3.0 metadata (info.json, stats.json, tasks, episodes)
  frames.lance  # one row per frame: tabular features, no pixels
  videos.lance  # one row per source video file: encoded bytes in a blob column
```

Feature keys contain dots (``observation.state``) which Lance column names do not
allow, so columns are stored with dots replaced by underscores and mapped back on
read. Metadata is byte-identical to the standard format and is loaded with
:class:`~lerobot.datasets.dataset_metadata.LeRobotDatasetMetadata`, so stats,
tasks and episode boundaries behave exactly as they do for parquet-backed
datasets.

Reading from the Hub streams the tables directly over ``hf://`` (only ``meta/``
is downloaded locally). Set ``HF_TOKEN`` in the environment for private repos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.import_utils import require_package

from .dataset_metadata import LeRobotDatasetMetadata
from .feature_utils import check_delta_timestamps, get_delta_indices

FRAMES_TABLE = "frames"
VIDEOS_TABLE = "videos"


def to_lance_column(key: str) -> str:
    """Map a LeRobot feature key to its Lance column name."""
    return key.replace(".", "_")


class LanceDBDataset(torch.utils.data.Dataset):
    """Map-style dataset over a Lance-backed LeRobot dataset.

    Returns the same item dict as :class:`LeRobotDataset` and satisfies the same
    duck-typed contract (``meta``, ``episodes``, ``absolute_to_relative_idx``,
    ``num_frames``), so it can be consumed by the training pipeline unchanged.

    Implements batched ``__getitems__``: the PyTorch ``DataLoader`` fetcher uses
    it automatically, and all rows a batch needs (including delta-timestamp
    windows) are fetched from the frames table in a single deduplicated read.

    Args:
        repo_id: Hub dataset repository (e.g. ``'lerobot/pusht_lance'``). Tables
            are streamed over ``hf://``; only ``meta/`` is downloaded.
        root: Local directory containing ``meta/`` and the ``.lance`` tables.
            When both ``repo_id`` and ``root`` are given, local tables win if
            present.
        episodes: Optional episode indices to select. ``None`` means all.
        delta_timestamps: Optional mapping of feature key to relative timestamp
            offsets (seconds) for temporal context windows, as in
            :class:`LeRobotDataset`.
        tolerance_s: Timestamp synchronization tolerance in seconds.
        revision: Hub revision for the ``meta/`` download.
        storage_options: Extra options forwarded to ``lancedb.connect`` (e.g.
            object-store credentials).
    """

    def __init__(
        self,
        repo_id: str | None = None,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        storage_options: dict | None = None,
    ):
        super().__init__()
        require_package("lancedb", extra="lance")
        if repo_id is None and root is None:
            raise ValueError("Provide `repo_id`, `root`, or both.")

        self.repo_id = repo_id
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self.tolerance_s = tolerance_s
        self._storage_options = storage_options

        self.meta = LeRobotDatasetMetadata(
            repo_id if repo_id is not None else str(self.root), root=self.root, revision=revision
        )

        if (self.root / f"{FRAMES_TABLE}.lance").exists():
            self._db_uri = str(self.root)
        elif repo_id is not None:
            self._db_uri = f"hf://datasets/{repo_id}"
        else:
            raise FileNotFoundError(f"No '{FRAMES_TABLE}.lance' table under {self.root}.")

        if self.meta.video_keys:
            raise NotImplementedError("Video decoding is not implemented yet (stage 3).")

        self.episodes = sorted(episodes) if episodes is not None else None
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)

        # Episode boundaries (absolute frame index space), used to clamp delta
        # windows and compute padding masks without a per-row lookup.
        self._ep_from = np.asarray(self.meta.episodes["dataset_from_index"], dtype=np.int64)
        self._ep_to = np.asarray(self.meta.episodes["dataset_to_index"], dtype=np.int64)

        # Row position in the frames table == absolute frame index. When a
        # subset of episodes is selected, __getitem__ indices are relative and
        # mapped through _rel_to_abs (mirrors DatasetReader's index mapping).
        if self.episodes is not None:
            self._rel_to_abs = np.concatenate(
                [np.arange(self._ep_from[ep], self._ep_to[ep]) for ep in self.episodes]
            )
            self._absolute_to_relative_idx = {
                int(abs_idx): rel_idx for rel_idx, abs_idx in enumerate(self._rel_to_abs)
            }
        else:
            self._rel_to_abs = None
            self._absolute_to_relative_idx = None

        # Tabular features live in the frames table; pixels do not.
        self._tabular_keys = [
            key
            for key in self.meta.features
            if key not in self.meta.video_keys and key not in self.meta.image_keys
        ]
        self._fetch_columns = [to_lance_column(key) for key in self._tabular_keys]
        self._feature_shapes = {
            key: tuple(self.meta.features[key].get("shape") or ()) for key in self._tabular_keys
        }

        # Lazily opened per process; see __getstate__.
        self._frames_perm = None

    # ── connection management ──────────────────────────────────────────────

    def _ensure_open(self) -> None:
        """Open the frames table handle for this process if needed.

        Handles are opened lazily and dropped on pickling so that each
        DataLoader worker builds its own. Lance's runtime is not fork-safe:
        use ``multiprocessing_context='spawn'`` with ``num_workers > 0``.
        """
        if self._frames_perm is not None:
            return
        import lancedb
        from lancedb.permutation import Permutation

        connect_kwargs = {"storage_options": self._storage_options} if self._storage_options else {}
        db = lancedb.connect(self._db_uri, **connect_kwargs)
        table = db.open_table(FRAMES_TABLE)
        self._frames_perm = (
            Permutation.identity(table).select_columns(self._fetch_columns).with_format("arrow")
        )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_frames_perm"] = None
        return state

    # ── dataset protocol ───────────────────────────────────────────────────

    @property
    def num_frames(self) -> int:
        return len(self._rel_to_abs) if self._rel_to_abs is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def absolute_to_relative_idx(self) -> dict[int, int] | None:
        return self._absolute_to_relative_idx

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def fps(self) -> int:
        return self.meta.fps

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Batched fetch: one deduplicated frames-table read for the whole batch."""
        self._ensure_open()
        plans = self._plan_batch(indices)
        rows = sorted({row for plan in plans for row in plan["rows"]})
        row_pos = {row: pos for pos, row in enumerate(rows)}
        columns = self._fetch_rows(rows)
        return [self._build_item(plan, columns, row_pos) for plan in plans]

    # ── internals ──────────────────────────────────────────────────────────

    def _plan_batch(self, indices: list[int]) -> list[dict]:
        """Resolve each sample to the absolute rows it needs and its padding masks."""
        plans = []
        for idx in indices:
            abs_idx = int(self._rel_to_abs[idx]) if self._rel_to_abs is not None else int(idx)
            ep_idx = int(np.searchsorted(self._ep_from, abs_idx, side="right") - 1)
            start, end = int(self._ep_from[ep_idx]), int(self._ep_to[ep_idx])
            plan = {"abs_idx": abs_idx, "rows": {abs_idx}, "windows": {}, "padding": {}}
            if self.delta_indices is not None:
                for key, deltas in self.delta_indices.items():
                    window = [min(max(abs_idx + delta, start), end - 1) for delta in deltas]
                    plan["windows"][key] = window
                    plan["rows"].update(window)
                    plan["padding"][f"{key}_is_pad"] = torch.BoolTensor(
                        [not (start <= abs_idx + delta < end) for delta in deltas]
                    )
            plans.append(plan)
        return plans

    def _fetch_rows(self, rows: list[int]) -> dict[str, np.ndarray]:
        """Fetch rows from the frames table and decode columns to numpy.

        Vector features come back as 2-D arrays ``(n_rows, dim)``; scalars as 1-D.
        """
        batch = self._frames_perm.__getitems__(rows)
        columns = {}
        for key, lance_name in zip(self._tabular_keys, self._fetch_columns, strict=True):
            array = batch.column(lance_name)
            if hasattr(array, "combine_chunks"):
                array = array.combine_chunks()
            if hasattr(array, "flatten") and hasattr(array.type, "value_type"):
                values = array.flatten().to_numpy(zero_copy_only=False)
                columns[key] = values.reshape(len(array), -1)
            else:
                columns[key] = array.to_numpy(zero_copy_only=False)
        return columns

    def _build_item(self, plan: dict, columns: dict[str, np.ndarray], row_pos: dict[int, int]) -> dict:
        base = row_pos[plan["abs_idx"]]
        item = {}
        for key in self._tabular_keys:
            data = columns[key]
            shape = self._feature_shapes[key]
            if key in plan["windows"]:
                window = data[[row_pos[row] for row in plan["windows"][key]]]
                if len(shape) > 1:
                    window = window.reshape(len(window), *shape)
                item[key] = torch.from_numpy(np.ascontiguousarray(window))
            elif data.ndim > 1:
                value = data[base]
                if len(shape) > 1:
                    value = value.reshape(shape)
                item[key] = torch.from_numpy(value.copy())
            else:
                item[key] = torch.tensor(data[base])
        item.update(plan["padding"])
        item["task"] = self.meta.tasks.iloc[int(item["task_index"].item())].name
        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  repo_id={self.repo_id},\n"
            f"  uri={self._db_uri},\n"
            f"  episodes={self.num_episodes} selected / {self.meta.total_episodes} total,\n"
            f"  frames={self.num_frames},\n"
            f")"
        )
